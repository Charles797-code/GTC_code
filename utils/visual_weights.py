import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


def visualize_embeddings_paper_quality(model, save_dir='./results/vis/'):
    """
    生成论文级别的 Student vs Teacher 嵌入对比图 (PDF格式)
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 设置论文绘图风格
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    plt.rcParams['pdf.fonttype'] = 42  # 确保字体可编辑
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'serif'  # 使用衬线字体更像论文

    # 获取模型中的嵌入参数 (假设模型被 DataParallel 包裹，需使用 .module)
    if hasattr(model, 'module'):
        base_model = model.module
    else:
        base_model = model

    # 提取权重并转为 numpy
    # 形状: [Num_Embeddings, Channel, Seq_Len] -> 我们需要展平后两维进行对比
    # 或者取 Mean/Max 代表该时间点的整体特征

    # ------------------ 处理 Hour Embedding ------------------
    if hasattr(base_model, 'emb_hour_s') and hasattr(base_model, 'emb_hour_t'):
        # [24, C, L] -> [24, C*L] (用于PCA和相似度)
        h_s = base_model.emb_hour_s.detach().cpu().numpy()
        h_t = base_model.emb_hour_t.detach().cpu().numpy()

        # 展平处理
        h_s_flat = h_s.reshape(h_s.shape[0], -1)
        h_t_flat = h_t.reshape(h_t.shape[0], -1)

        _plot_comparison(h_s_flat, h_t_flat, "Hour", save_dir)

    # ------------------ 处理 Day Embedding ------------------
    if hasattr(base_model, 'emb_day_s') and hasattr(base_model, 'emb_day_t'):
        d_s = base_model.emb_day_s.detach().cpu().numpy()
        d_t = base_model.emb_day_t.detach().cpu().numpy()

        if d_s is not None and d_t is not None:
            # Day 往往比较多，可能有30-31个，同理处理
            d_s_flat = d_s.reshape(d_s.shape[0], -1)
            d_t_flat = d_t.reshape(d_t.shape[0], -1)
            _plot_comparison(d_s_flat, d_t_flat, "Day", save_dir)


def _plot_comparison(emb_s, emb_t, name, save_dir):
    """
    核心绘图逻辑：包含 PCA 投影、相似度矩阵和热力图
    """
    num_points = emb_s.shape[0]

    # --- 图 1: PCA 投影对比 (展示分布对齐情况) ---
    pca = PCA(n_components=2)
    # Fit on Teacher, Transform both (以Teacher为基准空间)
    emb_t_pca = pca.fit_transform(emb_t)
    emb_s_pca = pca.transform(emb_s)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制 Teacher 轨迹 (虚线圆环)
    ax.plot(emb_t_pca[:, 0], emb_t_pca[:, 1], 'o--', color='#34495e', alpha=0.6, label='Teacher Trajectory',
            linewidth=1)
    # 绘制 Student 轨迹 (实线圆环)
    ax.plot(emb_s_pca[:, 0], emb_s_pca[:, 1], '^-', color='#e74c3c', alpha=0.8, label='Student Trajectory',
            linewidth=1.5)

    # 标注起点和部分点
    for i in range(0, num_points, max(1, num_points // 12)):  # 每隔几个点标一下数字
        ax.text(emb_t_pca[i, 0], emb_t_pca[i, 1], str(i), fontsize=8, color='#34495e')
        ax.text(emb_s_pca[i, 0], emb_s_pca[i, 1], str(i), fontsize=8, color='#e74c3c', fontweight='bold')

    ax.set_title(f'PCA Projection of {name} Embeddings (Distillation Alignment)', fontsize=14, pad=20)
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)
    sns.despine()

    plt.savefig(os.path.join(save_dir, f'{name}_PCA_Comparison.pdf'), bbox_inches='tight')
    plt.close()

    # --- 图 2: 余弦相似度矩阵 (Cosine Similarity Matrix) ---
    # 计算 S 和 T 之间的相似度矩阵
    # Normalize
    norm_s = emb_s / np.linalg.norm(emb_s, axis=1, keepdims=True)
    norm_t = emb_t / np.linalg.norm(emb_t, axis=1, keepdims=True)
    similarity_matrix = np.dot(norm_s, norm_t.T)  # [N, N]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(similarity_matrix, cmap="RdBu_r", center=0,
                xticklabels=5, yticklabels=5, square=True, cbar_kws={"shrink": .8})

    ax.set_xlabel('Teacher Index', fontsize=12)
    ax.set_ylabel('Student Index', fontsize=12)
    ax.set_title(f'{name} Embedding Cosine Similarity (Student vs Teacher)', fontsize=14, pad=15)

    plt.savefig(os.path.join(save_dir, f'{name}_Similarity_Matrix.pdf'), bbox_inches='tight')
    plt.close()

    # --- 图 3: 原始特征热力图对比 (Feature Heatmap) ---
    # 为了可视化，我们只取前 20-50 个主要特征维度（或者PCA降维后的前几维，这里直接用原始维度的平均值如果维度过大）
    # 这里我们对 Channel * SeqLen 进行降维或者切片展示

    # 策略：如果特征太长，仅展示 Mean Pooling 后的 Channel 变化
    # 这里简单展示前 50 个 Flatten 特征的变化
    disp_dim = min(100, emb_s.shape[1])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 统一 Scale
    vmin = min(emb_s[:, :disp_dim].min(), emb_t[:, :disp_dim].min())
    vmax = max(emb_s[:, :disp_dim].max(), emb_t[:, :disp_dim].max())

    sns.heatmap(emb_t[:, :disp_dim].T, ax=ax1, cmap="viridis", vmin=vmin, vmax=vmax, cbar=False)
    ax1.set_ylabel('Feature Dim (Partial)', fontsize=10)
    ax1.set_title(f'Teacher {name} Representations', fontsize=12, loc='left')

    sns.heatmap(emb_s[:, :disp_dim].T, ax=ax2, cmap="viridis", vmin=vmin, vmax=vmax,
                cbar_kws={"orientation": "horizontal", "pad": 0.2, "label": "Activation Value"})
    ax2.set_ylabel('Feature Dim (Partial)', fontsize=10)
    ax2.set_xlabel(f'{name} Index', fontsize=12)
    ax2.set_title(f'Student {name} Representations', fontsize=12, loc='left')

    plt.suptitle(f'Raw Feature Comparison: {name}', fontsize=14, y=0.95)
    plt.savefig(os.path.join(save_dir, f'{name}_Heatmap_Comparison.pdf'), bbox_inches='tight')
    plt.close()

    print(f"Paper-quality visualizations for {name} saved to {save_dir}")