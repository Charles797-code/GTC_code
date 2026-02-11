import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevON import RevON


class ResBlock(nn.Module):
    def __init__(self, seq_len, d_ff, dropout=0.1):
        super(ResBlock, self).__init__()
        self.norm = nn.LayerNorm(seq_len)
        self.linear1 = nn.Linear(seq_len, d_ff)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, seq_len)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x + residual


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.use_day_index = configs.use_day_index
        self.use_hour_index = configs.use_hour_index

        self.temperature = getattr(configs, 'contrastive_temp', 0.1)
        self.e_layers = getattr(configs, 'e_layers', 8)
        self.a_layers = getattr(configs, 'a_layers', 0)
        self.b_layers = getattr(configs, 'b_layers', 0)
        self.d_ff = getattr(configs, 'd_ff', 2048)
        self.dropout = getattr(configs, 'dropout', 0.1)

        self.emb_len_hour = configs.hour_length
        self.emb_len_day = configs.day_length

        self.emb_hour_i = nn.Parameter(torch.randn(self.emb_len_hour, self.enc_in, self.seq_len) * 0.01)
        self.emb_day_i = nn.Parameter(torch.randn(self.emb_len_day, self.enc_in, self.seq_len) * 0.01)

        self.emb_hour_c = nn.Parameter(torch.randn(self.emb_len_hour, self.enc_in, self.seq_len) * 0.01)
        self.emb_day_c = nn.Parameter(torch.randn(self.emb_len_day, self.enc_in, self.seq_len) * 0.01)

        self.revon_i = RevON(num_features=self.enc_in, affine=True)
        self.revon_c = RevON(num_features=self.enc_in, affine=True)

        self.total_channels_i = self.enc_in
        if self.use_hour_index: self.total_channels_i += (2 * self.enc_in)
        if self.use_day_index: self.total_channels_i += (2 * self.enc_in)

        self.feature_fusion_fine = nn.Linear(self.total_channels_i, self.enc_in)
        self.feature_fusion_coarse = nn.Linear(self.total_channels_i, self.enc_in)

        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.upsample_conv = nn.Sequential(
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(self.enc_in, self.enc_in, kernel_size=1)
        )

        self.backbone_fine = nn.ModuleList(
            [ResBlock(self.seq_len, self.d_ff, self.dropout) for _ in range(self.a_layers)])
        self.backbone_coarse = nn.ModuleList(
            [ResBlock(self.seq_len // 2, self.d_ff, self.dropout) for _ in range(self.b_layers)])
        self.backbone_shared = nn.ModuleList(
            [ResBlock(self.seq_len, self.d_ff, self.dropout) for _ in range(self.e_layers)])

        self.proj = nn.Linear(self.seq_len, self.pred_len)

    def alignment_loss(self, features_i, features_c):
        batch_size = features_i.shape[0]
        f_i = F.normalize(features_i.reshape(batch_size, -1), dim=1)
        f_c = F.normalize(features_c.reshape(batch_size, -1), dim=1)
        logits = torch.matmul(f_i, f_c.T) / self.temperature
        labels = torch.arange(batch_size, device=features_i.device)
        return F.cross_entropy(logits, labels)

    def forward(self, x, x_m, hour_index, day_index=None, x_gt=None, **kwargs):
        if self.use_revin:
            x = self.revon_i(x, "norm", x_m)
            if x_gt is not None:
                x_gt = self.revon_c(x_gt, "norm", None)

        x_freq = torch.fft.rfft(x, dim=1)
        keep_freq = x_freq.shape[1] // 4
        x_freq_low = torch.zeros_like(x_freq)
        x_freq_low[:, :keep_freq, :] = x_freq[:, :keep_freq, :]
        x_reconstructed = torch.fft.irfft(x_freq_low, dim=1, n=x.shape[1])
        x_completed = x_m * x + (1 - x_m) * x_reconstructed

        x_i_in = x_completed.permute(0, 2, 1)
        x_c_in = (x_gt if x_gt is not None else x).permute(0, 2, 1)

        idx_h = hour_index % self.emb_len_hour
        idx_d = day_index % self.emb_len_day if self.use_day_index else None

        eh_i, eh_c = self.emb_hour_i[idx_h], self.emb_hour_c[idx_h]
        ed_i = self.emb_day_i[idx_d] if self.use_day_index else None
        ed_c = self.emb_day_c[idx_d] if self.use_day_index else None

        feat_c_list = [x_c_in]
        if self.use_hour_index: feat_c_list.append(eh_c)
        if self.use_day_index: feat_c_list.append(ed_c)
        feat_c = torch.cat(feat_c_list, dim=1)

        feat_i_align_list = [x_i_in]
        if self.use_hour_index: feat_i_align_list.append(eh_i)
        if self.use_day_index: feat_i_align_list.append(ed_i)
        feat_i_align = torch.cat(feat_i_align_list, dim=1)

        if x.shape[0] > 1:
            align_loss = self.alignment_loss(feat_i_align, feat_c)
        else:
            align_loss = F.mse_loss(feat_i_align, feat_c)

        x_i_total = torch.cat([feat_i_align, eh_c, ed_c] if self.use_day_index and self.use_hour_index else
                              [feat_i_align, eh_c] if self.use_hour_index else [feat_i_align], dim=1)

        xf = x_i_total.permute(0, 2, 1)
        xf = self.feature_fusion_fine(xf).permute(0, 2, 1)
        for block in self.backbone_fine: xf = block(xf)

        xc = self.downsample(x_i_total)
        xc = xc.permute(0, 2, 1)
        xc = self.feature_fusion_coarse(xc).permute(0, 2, 1)
        for block in self.backbone_coarse: xc = block(xc)

        xc_up = F.interpolate(xc, size=self.seq_len, mode='linear', align_corners=False)
        xc_up = self.upsample_conv(xc_up)
        fused = xf + xc_up

        for block in self.backbone_shared:
            fused = block(fused)

        y = self.proj(fused).permute(0, 2, 1)
        if self.use_revin:
            y = self.revon_i(y, "denorm")

        return y, align_loss