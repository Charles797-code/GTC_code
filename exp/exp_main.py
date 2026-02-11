from data_provider.data_factory import data_provider, _get_full_data
from exp.exp_basic import Exp_Basic
from models import Autoformer, DLinear, PatchTST, GTC
from utils.polynomial import (chebyshev_torch, hermite_torch, laguerre_torch,
                              leg_torch)
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

        if args.add_noise and args.noise_amp > 0:
            seq_len = args.pred_len
            cutoff_freq_percentage = args.noise_freq_percentage
            cutoff_freq = int((seq_len // 2 + 1) * cutoff_freq_percentage)
            if args.auxi_mode == "rfft":
                low_pass_mask = torch.ones(seq_len // 2 + 1)
                low_pass_mask[-cutoff_freq:] = 0.
            else:
                raise NotImplementedError
            self.mask = low_pass_mask.reshape(1, -1, 1).to(self.device)
        else:
            self.mask = None

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'DLinear': DLinear,
            'PatchTST': PatchTST,
            'GTC': GTC,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, x_m):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index, x_m, x_c) in enumerate(
                    vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                hour_index = hour_index.int().to(self.device)
                if torch.all(day_index != -1):
                    day_index = day_index.int().to(self.device)

                x_m = x_m.int().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'GTC' in self.args.model:
                            outputs, _ = self.model(batch_x, x_m, hour_index, day_index)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'GTC' in self.args.model:
                        outputs, _ = self.model(batch_x, x_m, hour_index, day_index)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.ckpoint, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            max_memory = 0
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index, x_m, x_c) in enumerate(
                    train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                hour_index = hour_index.int().to(self.device)
                if torch.all(day_index != -1):
                    day_index = day_index.int().to(self.device)

                x_m = x_m.int().to(self.device)
                x_c = x_c.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        align_loss = 0

                        if 'GTC' in self.args.model:
                            outputs, align_loss = self.model(batch_x, x_m, hour_index, day_index, x_c)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0

                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_slice = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss_imputed = criterion(outputs, batch_y_slice)

                        loss = loss_imputed

                        if 'GTC' in self.args.model:
                            align_weight = getattr(self.args, 'align_rate', 0.6)
                            loss = loss_imputed + (align_weight * align_loss)

                        train_loss.append(loss.item())
                else:
                    align_loss = 0
                    if 'GTC' in self.args.model:
                        outputs, align_loss = self.model(batch_x, x_m, hour_index, day_index, x_c)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)

                    f_dim = -1 if self.args.features == 'MS' else 0

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_slice = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss_imputed = criterion(outputs, batch_y_slice)
                    loss = loss_imputed

                    if 'GTC' in self.args.model:
                        align_weight = getattr(self.args, 'align_rate', 1)
                        loss = loss_imputed + (align_weight * align_loss)

                    if self.args.rec_lambda and self.args.rec_lambda > 0:
                        loss_rec = criterion(outputs, batch_y_slice)
                        loss += self.args.rec_lambda * loss_rec
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_rec: {loss_rec.item()}")

                    if self.args.auxi_lambda and self.args.auxi_lambda > 0:
                        if self.args.auxi_mode == "fft":
                            loss_auxi = torch.fft.fft(outputs, dim=1) - torch.fft.fft(batch_y_slice, dim=1)
                        elif self.args.auxi_mode == "rfft":
                            if self.args.auxi_type == 'complex':
                                loss_auxi = torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y_slice, dim=1)
                            elif self.args.auxi_type == 'complex-phase':
                                loss_auxi = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y_slice,
                                                                                              dim=1)).angle()
                            elif self.args.auxi_type == 'complex-mag-phase':
                                loss_auxi_mag = (
                                        torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y_slice, dim=1)).abs()
                                loss_auxi_phase = (torch.fft.rfft(outputs, dim=1) - torch.fft.rfft(batch_y_slice,
                                                                                                   dim=1)).angle()
                                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])
                            elif self.args.auxi_type == 'phase':
                                loss_auxi = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y_slice,
                                                                                                    dim=1).angle()
                            elif self.args.auxi_type == 'mag':
                                loss_auxi = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y_slice,
                                                                                                  dim=1).abs()
                            elif self.args.auxi_type == 'mag-phase':
                                loss_auxi_mag = torch.fft.rfft(outputs, dim=1).abs() - torch.fft.rfft(batch_y_slice,
                                                                                                      dim=1).abs()
                                loss_auxi_phase = torch.fft.rfft(outputs, dim=1).angle() - torch.fft.rfft(batch_y_slice,
                                                                                                          dim=1).angle()
                                loss_auxi = torch.stack([loss_auxi_mag, loss_auxi_phase])

                        if self.mask is not None:
                            loss_auxi *= self.mask

                        if self.args.auxi_loss == "MAE":
                            loss_auxi = loss_auxi.abs().mean()
                        elif self.args.auxi_loss == "MSE":
                            loss_auxi = (loss_auxi.abs() ** 2).mean()

                        loss += self.args.auxi_lambda * loss_auxi
                        if (i + 1) % 100 == 0:
                            print(f"\tloss_auxi: {loss_auxi.item()}")

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                current_memory = torch.cuda.max_memory_allocated() / 1024 ** 2
                max_memory = max(max_memory, current_memory)

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, x_m)
            test_loss = self.vali(test_data, test_loader, criterion, x_m)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        print(f"Max Memory (MB): {max_memory}")

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./ckpoint/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index, x_m, x_c) in enumerate(
                    test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                hour_index = hour_index.int().to(self.device)
                if torch.all(day_index != -1):
                    day_index = day_index.int().to(self.device)

                x_m = x_m.int().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'GTC' in self.args.model:
                            outputs, _ = self.model(batch_x, x_m, hour_index, day_index)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'GTC' in self.args.model:
                        outputs, _ = self.model(batch_x, x_m, hour_index, day_index)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    np.savetxt(os.path.join(folder_path, str(i) + '.txt'), pd)
                    np.savetxt(os.path.join(folder_path, str(i) + 'true.txt'), gt)

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.ckpoint, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, hour_index, day_index, x_c) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                hour_index = hour_index.int().to(self.device)
                if torch.all(day_index != -1):
                    day_index = day_index.int().to(self.device)

                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
                    batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'GTC' in self.args.model:
                            outputs, _ = self.model(batch_x, torch.ones_like(batch_x), hour_index, day_index)
                        elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'GTC' in self.args.model:
                        outputs, _ = self.model(batch_x, torch.ones_like(batch_x), hour_index, day_index)
                    elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'TST'}):
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return