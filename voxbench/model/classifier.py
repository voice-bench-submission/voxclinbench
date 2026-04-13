"""
VoiceDiseaseModel: 7-branch CNN-Transformer multi-task classifier.

Architecture overview
─────────────────────
  Branch A  (spec)   : EfficientNet-B0 (pretrained, 3-ch) → 1280-d
  Branch B  (mfcc)   : ResNet18 (pretrained, 1-ch avg)    → 512-d
  Branch C  (mel)    : ResNet18 (pretrained, 1-ch avg)    → 512-d
  Branch D  (ppg)    : 3 × Conv1D + stats pool            → 512-d
  Branch E  (ema)    : 3 × Conv1D + stats pool            → 256-d
  Branch F  (pros)   : 3 × Conv1D + stats pool            → 128-d
  Branch G  (static) : 2-layer MLP                        → 64-d

Fusion
  Project each branch → d_model
  Add learnable modality embedding
  Prepend CLS token
  2-layer Pre-LN TransformerEncoder (src_key_padding_mask for missing modalities)
  Disease-specific attention: each head queries over the 8 transformer tokens
    (CLS + 7 modality tokens) with a learned query vector, so heads for
    physiological diseases can attend heavily to spectral tokens while psychiatric
    heads can upweight prosodic / static tokens independently.
"""
import torch
import torch.nn as nn

from voxbench.config import MODALITY_KEYS, N_STATIC_FEATURES
from voxbench.model.branches import BranchD, BranchE, BranchF, BranchG


class VoiceDiseaseModel(nn.Module):
    # Output dim of each branch (D/E/F doubled by mean+std stats pooling)
    BRANCH_DIMS = [1280, 512, 512, 512, 256, 128, 64]  # A B C D E F G

    def __init__(
        self,
        n_diseases:     int,
        d_model:        int   = 512,
        nhead:          int   = 8,
        n_layers:       int   = 2,
        dropout_fusion: float = 0.3,
        dropout_head:   float = 0.5,
        head_hidden_dim:int   = 256,
        modality_dropout_prob: float = 0.0,
        n_static_features: int = N_STATIC_FEATURES,
    ):
        super().__init__()
        import torchvision.models as tvm
        self.modality_dropout_prob = float(modality_dropout_prob)

        # Branch A: EfficientNet-B0 (pretrained on ImageNet)
        enet = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.branch_a      = enet.features
        self.branch_a_pool = nn.AdaptiveAvgPool2d(1)

        # Branch B/C: separate ResNet18 backbones (pretrained, adapted to 1-channel input)
        rnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        with torch.no_grad():
            new_w = rnet.conv1.weight.mean(dim=1, keepdim=True)
        rnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            rnet.conv1.weight.copy_(new_w)
        rnet.fc = nn.Identity()
        self.branch_c = rnet

        mfcc_net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        with torch.no_grad():
            mfcc_w = mfcc_net.conv1.weight.mean(dim=1, keepdim=True)
        mfcc_net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            mfcc_net.conv1.weight.copy_(mfcc_w)
        mfcc_net.fc = nn.Identity()
        self.branch_b = mfcc_net

        # Branches D–G
        self.branch_d = BranchD(dropout_fusion)
        self.branch_e = BranchE(dropout_fusion)
        self.branch_f = BranchF(dropout_fusion)
        self.branch_g = BranchG(n_static_features, dropout_fusion)

        # Per-branch projection into the shared token space
        self.proj = nn.ModuleList([nn.Linear(d, d_model) for d in self.BRANCH_DIMS])

        # Learnable modality-type embeddings
        self.modality_embed = nn.Embedding(len(MODALITY_KEYS), d_model)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer encoder (Pre-LN for training stability)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_fusion,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Per-disease attention queries (each disease attends over transformer tokens)
        self.head_queries = nn.Parameter(torch.zeros(n_diseases, d_model))
        nn.init.normal_(self.head_queries, std=0.02)
        self._head_scale = d_model ** -0.5

        # Per-disease binary classification heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, head_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_head),
                nn.Linear(head_hidden_dim, 1),
            )
            for _ in range(n_diseases)
        ])

    def _empty_branch_out(
        self,
        batch_size: int,
        branch_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.BRANCH_DIMS[branch_idx],
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        spec:         torch.Tensor,              # [B, 1, 201, T]
        mfcc:         torch.Tensor,              # [B, 1,  60, T]
        mel:          torch.Tensor,              # [B, 1,  60, T]
        ppg:          torch.Tensor,              # [B, 40, T]
        ema:          torch.Tensor,              # [B, 12, T]
        pros:         torch.Tensor,              # [B, 3,  T]
        static:       torch.Tensor,              # [B, N_STATIC_FEATURES]
        available:    torch.Tensor,              # [B, 7] bool
        act_len_ppg:  torch.Tensor | None = None,
        act_len_ema:  torch.Tensor | None = None,
        act_len_pros: torch.Tensor | None = None,
    ) -> torch.Tensor:                           # [B, n_diseases]
        B, device = spec.size(0), spec.device
        available_eff = available

        # Training-time modality dropout discourages the transformer from
        # over-relying on spec / mel tokens and improves robustness when a
        # subset of branches carries the disease signal.
        if self.training and self.modality_dropout_prob > 0:
            keep = (torch.rand_like(available.float()) > self.modality_dropout_prob)
            available_eff = available & keep
            empty_rows = ~available_eff.any(dim=1)
            if empty_rows.any():
                available_eff = available_eff.clone()
                available_eff[empty_rows] = available[empty_rows]

        def _branch_enabled(mod_idx: int) -> bool:
            return bool(available_eff[:, mod_idx].any().item())

        # Skip whole branches when the batch has no usable examples for that modality.
        branch_outs = []
        if _branch_enabled(0):
            fa = self.branch_a_pool(self.branch_a(spec.repeat(1, 3, 1, 1))).flatten(1)
        else:
            fa = self._empty_branch_out(B, 0, device, spec.dtype)
        branch_outs.append(fa)

        if _branch_enabled(1):
            branch_outs.append(self.branch_b(mfcc))
        else:
            branch_outs.append(self._empty_branch_out(B, 1, device, mfcc.dtype))

        if _branch_enabled(2):
            branch_outs.append(self.branch_c(mel))
        else:
            branch_outs.append(self._empty_branch_out(B, 2, device, mel.dtype))

        if _branch_enabled(3):
            branch_outs.append(self.branch_d(ppg, act_len=act_len_ppg))
        else:
            branch_outs.append(self._empty_branch_out(B, 3, device, ppg.dtype))

        if _branch_enabled(4):
            branch_outs.append(self.branch_e(ema, act_len=act_len_ema))
        else:
            branch_outs.append(self._empty_branch_out(B, 4, device, ema.dtype))

        if _branch_enabled(5):
            branch_outs.append(self.branch_f(pros, act_len=act_len_pros))
        else:
            branch_outs.append(self._empty_branch_out(B, 5, device, pros.dtype))

        if _branch_enabled(6):
            branch_outs.append(self.branch_g(static))
        else:
            branch_outs.append(self._empty_branch_out(B, 6, device, static.dtype))

        # Project, embed, and zero-out missing modalities
        tokens = []
        for i, (proj, out) in enumerate(zip(self.proj, branch_outs)):
            t = proj(out) + self.modality_embed.weight[i]        # [B, D]
            tokens.append(t * available_eff[:, i].float().unsqueeze(1))

        # Prepend CLS token and run transformer
        token_seq = torch.stack(tokens, dim=1)                   # [B, 7, D]
        x = torch.cat([self.cls_token.expand(B, -1, -1), token_seq], dim=1)  # [B, 8, D]

        # CLS is always attended; missing modality tokens are ignored
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device),
            ~available_eff,
        ], dim=1)  # [B, 8]

        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Disease-specific attention over all 8 tokens (CLS + 7 modality slots).
        # Each disease head has its own learned query; CLS is never masked so
        # softmax always has at least one finite value even when modalities are missing.
        #   scores : [B, n_diseases, 8]
        #   weights: [B, n_diseases, 8]  (softmax over tokens)
        #   context: [B, n_diseases, D]
        scores  = torch.einsum("nd,btd->bnt", self.head_queries, x) * self._head_scale
        scores  = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)                       # [B, n_d, 8]
        context = torch.einsum("bnt,btd->bnd", weights, x)           # [B, n_d, D]

        return torch.cat(
            [h(context[:, i]) for i, h in enumerate(self.heads)], dim=1
        )  # [B, n_diseases]
