# t_jepa_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math


# -----------------------------------------------------------------------------
# 1. 基础组件：Transformer Encoder Block
# -----------------------------------------------------------------------------
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # I-JEPA/T-JEPA 通常推荐 Pre-Norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        # self.norm = nn.LayerNorm(dim) # LayerNorm 已经在 EncoderLayer 内部处理了

    def forward(self, x):
        return self.encoder(x)


# -----------------------------------------------------------------------------
# 2. 核心模型：T-JEPA for Shield Machine Data
# -----------------------------------------------------------------------------
class TJEPA(nn.Module):
    def __init__(self, num_features=9, embed_dim=128, depth=4, heads=4, predictor_depth=2):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # --- A. Embedding Layer ---
        # 针对数值型表格数据，通常对每个特征使用独立的 Linear 层进行映射
        self.feature_embeds = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])

        # Positional Embedding (用于区分这是第几个特征)
        # 形状: (1, num_features + 1, embed_dim)  --> +1 是为了 [REG] token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_features + 1, embed_dim))
        self._init_pos_embed()

        # [REG] Token (防止坍塌的关键)
        self.reg_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # [MASK] Token (用于 Predictor 的输入占位符)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- B. Context Encoder (f_theta) ---
        self.context_encoder = TransformerEncoder(embed_dim, depth, heads, embed_dim * 4)

        # --- C. Target Encoder (f_bar_theta) ---
        # 结构与 Context Encoder 完全相同，初始权重也相同
        self.target_encoder = copy.deepcopy(self.context_encoder)
        # Target Encoder 不参与梯度更新 (Stop Gradient)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # --- D. Predictor (g_phi) ---
        self.predictor = TransformerEncoder(embed_dim, predictor_depth, heads, embed_dim * 4)
        # Predictor 输出层投影 (映射回 latent space)
        self.predictor_proj = nn.Linear(embed_dim, embed_dim)

    def _init_pos_embed(self):
        # 使用正弦位置编码初始化，或者直接随机初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def embed_data(self, x):
        """
        输入 x: (Batch, Num_Features)
        输出: (Batch, Num_Features, Embed_Dim)
        """
        b, n = x.shape
        embeddings = []
        for i in range(n):
            feat = x[:, i:i + 1]  # (B, 1)
            emb = self.feature_embeds[i](feat)  # (B, E)
            embeddings.append(emb)

        x_emb = torch.stack(embeddings, dim=1)  # (B, N, E)
        return x_emb

    def forward_target(self, x):
        """
        Target Encoder 前向传播：处理完整的、未遮挡的数据
        """
        with torch.no_grad():  # 确保不反传梯度
            b = x.shape[0]
            x_emb = self.embed_data(x)

            # 1. 拼接 [REG] token 到末尾
            reg = self.reg_token.expand(b, -1, -1)  # (B, 1, E)
            x_emb = torch.cat([x_emb, reg], dim=1)  # (B, N+1, E)

            # 2. 加上位置编码
            x_emb = x_emb + self.pos_embed

            # 3. 通过 Target Encoder
            h_target = self.target_encoder(x_emb)

            return h_target  # (B, N+1, E)

    def forward_context(self, x, mask_indices):
        """
        Context Encoder 前向传播：处理被遮挡的数据
        mask_indices: (B, N) 布尔矩阵，False 表示被遮挡(drop)，True 表示保留
        """
        b, n = x.shape
        x_emb = self.embed_data(x)

        context_batch = []

        # 这里的实现采用了“Drop Token”的方式，即 Transformer 只看到未遮挡的 Token
        # 这比用 mask token 替换更符合 MAE/JEPA 的做法

        for i in range(b):
            # 获取当前样本保留的特征索引
            keep_idx = torch.nonzero(mask_indices[i]).squeeze()
            if keep_idx.ndim == 0: keep_idx = keep_idx.unsqueeze(0)

            # 取出保留的 Embedding
            curr_emb = x_emb[i, keep_idx, :]  # (L_keep, E)

            # 加上对应的位置编码 (关键！否则网络不知道这是哪个特征)
            # pos_embed 前 N 个对应 N 个特征，最后一个对应 REG
            curr_pos = self.pos_embed[0, keep_idx, :]
            curr_emb = curr_emb + curr_pos

            # 拼接 [REG] Token (REG 始终保留，不参与 Mask)
            # REG 的位置编码是 pos_embed 的最后一个
            reg_emb = self.reg_token[0] + self.pos_embed[0, -1, :]
            curr_emb = torch.cat([curr_emb, reg_emb], dim=0)  # (L_keep+1, E)

            context_batch.append(curr_emb)

        # Pack成 Tensor (假设每个 Batch Mask 数量一致，简化处理)
        # 如果 Mask 比例随机，这里需要 Padding。为演示简单，假设 batch 内 mask 数量一致。
        h_context_input = torch.stack(context_batch, dim=0)

        # 通过 Context Encoder
        h_context = self.context_encoder(h_context_input)

        return h_context  # (B, L_keep+1, E)

    def forward_predictor(self, h_context, mask_indices):
        """
        Predictor 前向传播：尝试从 Context 恢复 Target 中被遮挡的部分
        """
        # T-JEPA 的 Predictor 输入是 Context Output + Mask Tokens (对应需要预测的位置)
        # 这里为了简化，我们让 Predictor 接收 (Context + Mask_Tokens) 并尝试恢复完整序列
        # 或者更高效的做法：只预测被 Mask 的位置。

        # 简单实现：构建一个包含 Context 和 Mask Token 的完整序列
        # 根据 mask_indices 还原顺序

        b, context_len, e = h_context.shape
        n_feat = self.num_features

        predictor_input = torch.zeros(b, n_feat + 1, e, device=h_context.device)

        for i in range(b):
            keep_mask = mask_indices[i]  # (N,) boolean

            # 1. 填入 Context (Unmasked) 部分
            # 注意 h_context 最后一个是 REG，我们要把它放到最后
            context_feats = h_context[i, :-1, :]  # (L_keep, E)
            context_reg = h_context[i, -1, :]  # (E,)

            # 找到 keep 的位置索引
            keep_idx = torch.nonzero(keep_mask).squeeze()
            predictor_input[i, keep_idx, :] = context_feats

            # 2. 填入 Mask Token (Masked) 部分
            mask_idx = torch.nonzero(~keep_mask).squeeze()
            if mask_idx.ndim == 0 and mask_idx.numel() > 0: mask_idx = mask_idx.unsqueeze(0)

            if mask_idx.numel() > 0:
                # Mask Token 也需要加上对应的位置编码
                mask_tokens = self.mask_token + self.pos_embed[0, mask_idx, :]
                predictor_input[i, mask_idx, :] = mask_tokens

            # 3. 填入 REG Token
            predictor_input[i, -1, :] = context_reg

        # 通过 Predictor
        h_pred = self.predictor(predictor_input)
        h_pred = self.predictor_proj(h_pred)

        return h_pred  # (B, N+1, E)

    # --- EMA 更新 (论文核心) ---
    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        """
        param_target = momentum * param_target + (1 - momentum) * param_context
        """
        for param_q, param_k in zip(self.context_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(momentum).add_((1.0 - momentum) * param_q.detach().data)