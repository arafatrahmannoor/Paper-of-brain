# ============================================================================
# BRAIN TUMOR CLASSIFICATION — v2: DUPLICATE-AWARE (GROUPED) CV + ABLATIONS
# ============================================================================
# Changes from v1 (code/brain_tumor_cv.py); every change is a CONFIG switch:
# - CV_MODE="grouped": StratifiedGroupKFold. All offline-augmented copies of one
#   slice (G_710.jpg, G_710_RO_.jpg, ...) and pHash near-duplicates share a group,
#   so they never fall on both sides of a split (outer folds AND inner validation).
#   CV_MODE="image" reproduces the v1 image-level split.
# - SELECT_ON="val_loss": checkpoint selection on validation cross-entropy (v1 used
#   validation accuracy, which saturates at ~99% on ~770 images).
#   Validation runs with a fixed RNG seed, so MC-dropout noise in the gate is identical
#   every epoch and does not make the selection random.
# - TTA / augmentation fixes: on greyscale MRI (R=G=B) saturation and hue changes are
#   exact no-ops, so SaturationTTA is replaced by a 1.05x zoom view and
#   saturation/hue jitter is set to 0. All views use the same Resize as validation.
# - ABLATION switch: full | image_only | graph_only | concat | no_recal |
#   no_trans | no_uncert | single_gcn.
# - Extra outputs: fold assignment (paths), best epoch + training history,
#   per-image gate weights / uncertainties, parameter count, FLOPs and latency.
# - Patient IDs are unavailable, so patient-level independence cannot be guaranteed;
#   grouping removes the augmented-copy and near-duplicate leakage that IS detectable.
# - For Kaggle 12-hour sessions, run ONE fold at a time:
#       FOLD_TO_RUN = 0   # Fold 1
#       FOLD_TO_RUN = 1   # Fold 2
#       ...
#       FOLD_TO_RUN = 4   # Fold 5
#   Set FOLD_TO_RUN = None only if you intentionally want all 5 folds in one run.
# ============================================================================

import os
import gc
import copy
import json
import time
import random
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings("ignore")

SEED = 42


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


def install_torch_geometric():
    try:
        import torch_geometric  # noqa: F401
        print("✅ torch_geometric already installed")
        return
    except ImportError:
        pass

    print("📦 Installing torch_geometric...")
    import torch
    tv = torch.__version__.split("+")[0]
    cv = torch.version.cuda
    cu = "cu" + cv.replace(".", "")[:3] if cv else "cpu"
    commands = [
        f"pip install -q torch-scatter -f https://data.pyg.org/whl/torch-{tv}+{cu}.html",
        f"pip install -q torch-sparse -f https://data.pyg.org/whl/torch-{tv}+{cu}.html",
        "pip install -q torch-geometric",
    ]
    for cmd in commands:
        subprocess.check_call(cmd.split())
    print("✅ torch_geometric successfully installed")


install_torch_geometric()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch_geometric.data import Data, Batch
from torch_geometric.nn import (
    GCNConv,
    GATConv,
    TransformerConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
import re
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = os.environ.get(
    "BT_DATA_PATH",
    "/kaggle/input/datasets/arafatrahmann/my-data221/Epic and CSCR hospital Dataset",
)
TRAIN_FOLDER = "Train"
TEST_FOLDER = "Test"

OUTPUT_DIR = Path(os.environ.get("BT_OUTPUT_DIR", "/kaggle/working/brain_tumor_results_v2"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 30
BATCH_SIZE = 36
EARLY_STOP_PATIENCE = 5
VAL_SPLIT = 0.08
NUM_WORKERS = 4

TTA_AUGMENTS = 10
MC_PASSES = 20
MC_DROPOUT = 0.20
AUX_LOSS_WEIGHT = 0.15

# 0..4 = one fold. None = all folds.
FOLD_TO_RUN = 0

CV_MODE = "grouped"          # "grouped" (duplicate-aware, recommended) or "image" (v1 behaviour)
USE_PHASH = True             # also merge perceptual-hash near-duplicates into one group
PHASH_MAX_HAMMING = 5        # 64-bit pHash; <=5 differing bits = near-duplicate
SELECT_ON = "val_loss"       # "val_loss" (recommended) or "val_acc" (v1 behaviour)
ABLATION = "full"            # full | image_only | graph_only | concat | no_recal | no_trans | no_uncert | single_gcn
MEASURE_COMPLEXITY = os.environ.get("BT_COMPLEXITY", "1") == "1"  # params, FLOPs, latency (single pass vs full TTA x MC pipeline)
PRETRAINED = os.environ.get("BT_PRETRAINED", "1") == "1"

if os.environ.get("BT_SMOKE") == "1":   # tiny CPU smoke test only
    EPOCHS, BATCH_SIZE, NUM_WORKERS, TTA_AUGMENTS, MC_PASSES = 2, 8, 0, 3, 3
    FOLD_TO_RUN = int(os.environ.get("BT_FOLD", "0"))
    ABLATION = os.environ.get("BT_ABLATION", ABLATION)

RUN_TAG = f"{CV_MODE}_{ABLATION}"

print("\n" + "=" * 80)
print("BRAIN TUMOR CLASSIFICATION FRAMEWORK")
print("=" * 80)
print(f"Epochs               : {EPOCHS}")
print(f"Batch size           : {BATCH_SIZE}")
print(f"Early-stop patience  : {EARLY_STOP_PATIENCE}")
print(f"TTA                   : {TTA_AUGMENTS}-way")
print(f"MC-dropout passes     : {MC_PASSES}")
print(f"MC-dropout p          : {MC_DROPOUT}")
print(f"Fold to run           : {FOLD_TO_RUN if FOLD_TO_RUN is not None else 'ALL'}")
print(f"Seed                  : {SEED}")
print(f"CV mode               : {CV_MODE} (pHash merge: {USE_PHASH}, max Hamming {PHASH_MAX_HAMMING})")
print(f"Checkpoint selection  : {SELECT_ON}")
print(f"Ablation              : {ABLATION}")
print("=" * 80 + "\n")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ============================================================================
# CALIBRATION METRICS
# ============================================================================
def calculate_uncertainty_metrics(probs, targets, num_bins=10):
    """Equal-width ECE and multiclass Brier score."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == targets

    ece = 0.0
    boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    for i in range(num_bins):
        lo, hi = boundaries[i], boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences > lo) & (confidences <= hi)
        prop = np.mean(in_bin)
        if prop > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += np.abs(bin_conf - bin_acc) * prop

    num_classes = probs.shape[1]
    one_hot = np.eye(num_classes, dtype=np.float64)[targets]
    brier = np.mean(np.sum((probs - one_hot) ** 2, axis=1))
    return float(ece), float(brier)


# ============================================================================
# HIERARCHICAL MULTI-SCALE GRAPH PYRAMID
# ============================================================================
class GraphPyramid:
    SCALES = [4, 8, 16]

    def create_graph(self, pil_img):
        img = pil_img.convert("L").resize((224, 224))
        img = np.asarray(img, dtype=np.float32)

        features, edges = [], []
        node_offset = 0

        for scale_idx, grid_size in enumerate(self.SCALES):
            patch_h = 224 // grid_size
            patch_w = 224 // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    patch = img[
                        i * patch_h:(i + 1) * patch_h,
                        j * patch_w:(j + 1) * patch_w,
                    ]
                    features.append(self._extract_features(patch, i, j, grid_size))

            # 8-neighbour edges within scale
            for i in range(grid_size):
                for j in range(grid_size):
                    curr = node_offset + i * grid_size + j
                    for di, dj in [
                        (-1, -1), (-1, 0), (-1, 1),
                        (0, -1),            (0, 1),
                        (1, -1),  (1, 0),   (1, 1),
                    ]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < grid_size and 0 <= nj < grid_size:
                            nbr = node_offset + ni * grid_size + nj
                            edges.append([curr, nbr])

            # Bidirectional parent-child edges between consecutive scales
            if scale_idx > 0:
                prev_grid = self.SCALES[scale_idx - 1]
                prev_offset = sum(s * s for s in self.SCALES[:scale_idx])
                ratio = grid_size // prev_grid
                for i in range(grid_size):
                    for j in range(grid_size):
                        curr = node_offset + i * grid_size + j
                        parent = prev_offset + (i // ratio) * prev_grid + (j // ratio)
                        edges.append([curr, parent])
                        edges.append([parent, curr])

            node_offset += grid_size * grid_size

        return Data(
            x=torch.tensor(features, dtype=torch.float32),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        )

    @staticmethod
    def _extract_features(patch, row, col, grid_size):
        mean, std = patch.mean(), patch.std()
        pmin, pmax = patch.min(), patch.max()
        median, var = np.median(patch), patch.var()
        gx = np.gradient(patch, axis=0)
        gy = np.gradient(patch, axis=1)
        grad_mag = np.sqrt(gx**2 + gy**2)
        p25, p50, p75, p90 = [np.percentile(patch, p) for p in [25, 50, 75, 90]]
        total = patch.size
        bright = (patch > 128).sum() / total
        dark = (patch < 64).sum() / total
        above_mean = (patch > mean).sum() / total
        mid_range = ((patch > 64) & (patch < 192)).sum() / total
        very_bright = (patch > 192).sum() / total
        norm_row, norm_col = row / grid_size, col / grid_size
        dist_center = np.sqrt((norm_row - 0.5) ** 2 + (norm_col - 0.5) ** 2)

        return [
            mean / 255.0,
            std / 255.0,
            pmin / 255.0,
            pmax / 255.0,
            median / 255.0,
            var / (255.0**2),
            p25 / 255.0,
            p50 / 255.0,
            p75 / 255.0,
            p90 / 255.0,
            np.abs(gx).mean() / 255.0,
            np.abs(gy).mean() / 255.0,
            grad_mag.mean() / 255.0,
            bright,
            dark,
            above_mean,
            mid_range,
            very_bright,
            norm_row,
            norm_col,
            dist_center,
            (row + col) / (2.0 * grid_size),
            grid_size / 16.0,
            1.0 / grid_size,
        ]


# ============================================================================
# MULTI-PATH GNN
# ============================================================================
class MultiPathGNN(nn.Module):
    def __init__(self, in_channels=24, hidden_dim=128, out_channels=384, gcn_only=False):
        super().__init__()
        self.gcn_only = gcn_only  # ablation: GCN path only, no learned path mixing
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.trans1 = TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=0.1)
        self.bn_gcn1 = nn.BatchNorm1d(hidden_dim)
        self.bn_gat1 = nn.BatchNorm1d(hidden_dim)
        self.bn_trans1 = nn.BatchNorm1d(hidden_dim)

        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.1)
        self.trans2 = TransformerConv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=0.1)
        self.bn_gcn2 = nn.BatchNorm1d(hidden_dim)
        self.bn_gat2 = nn.BatchNorm1d(hidden_dim)
        self.bn_trans2 = nn.BatchNorm1d(hidden_dim)
        self.bn_residual = nn.BatchNorm1d(hidden_dim)

        self.alpha = nn.Parameter(torch.zeros(3))
        self.beta = nn.Parameter(torch.zeros(3))

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(0.2),
        )

    def forward(self, x, edge_index, batch):
        x = self.input_proj(x)
        w1 = F.softmax(self.alpha, dim=0)
        w2 = F.softmax(self.beta, dim=0)

        h_gcn1 = F.gelu(self.bn_gcn1(self.gcn1(x, edge_index)))
        if self.gcn_only:
            h1 = h_gcn1
        else:
            h_gat1 = F.gelu(self.bn_gat1(self.gat1(x, edge_index)))
            h_tr1 = F.gelu(self.bn_trans1(self.trans1(x, edge_index)))
            h1 = w1[0] * h_gcn1 + w1[1] * h_gat1 + w1[2] * h_tr1
        h1 = F.dropout(h1, p=0.2, training=self.training)

        h_gcn2 = F.gelu(self.bn_gcn2(self.gcn2(h1, edge_index)))
        if self.gcn_only:
            h2 = h_gcn2
        else:
            h_gat2 = F.gelu(self.bn_gat2(self.gat2(h1, edge_index)))
            h_tr2 = F.gelu(self.bn_trans2(self.trans2(h1, edge_index)))
            h2 = w2[0] * h_gcn2 + w2[1] * h_gat2 + w2[2] * h_tr2
        h2 = F.dropout(self.bn_residual(h2 + h1), p=0.15, training=self.training)

        pooled = torch.cat(
            [
                global_mean_pool(h2, batch),
                global_max_pool(h2, batch),
                global_add_pool(h2, batch),
            ],
            dim=1,
        )
        return self.output_head(pooled)


# ============================================================================
# EFFICIENTNET-B3 CNN BACKBONE
# ============================================================================
class CNNBackbone(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1 if PRETRAINED else None)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(768, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.projection(self.features(x))


# ============================================================================
# DYNAMIC FEATURE RECALIBRATION
# ============================================================================
class DynamicRecalibration(nn.Module):
    def __init__(self, channels=384):
        super().__init__()
        self.cg_g = nn.Sequential(
            nn.Linear(channels, channels // 16), nn.GELU(),
            nn.Linear(channels // 16, channels), nn.Sigmoid(),
        )
        self.cg_i = nn.Sequential(
            nn.Linear(channels, channels // 16), nn.GELU(),
            nn.Linear(channels // 16, channels), nn.Sigmoid(),
        )
        self.sg = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LayerNorm(channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid(),
        )
        self.cmg = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.ln_g = nn.LayerNorm(channels)
        self.ln_i = nn.LayerNorm(channels)

    def forward(self, g_f, i_f):
        g_c = g_f * self.cg_g(g_f)
        i_c = i_f * self.cg_i(i_f)
        g_s = g_c * self.sg(g_c)
        i_s = i_c * self.sg(i_c)
        cross_weight = self.cmg(torch.cat([g_s, i_s], dim=1))
        g_out = self.ln_g(g_s * cross_weight + g_f)
        i_out = self.ln_i(i_s * cross_weight + i_f)
        return g_out, i_out


# ============================================================================
# CROSS-MODAL TRANSFORMER
# ============================================================================
class CrossModalTransformer(nn.Module):
    """Two modality tokens [GNN, CNN] jointly attend to one another."""

    def __init__(self, dim=384, num_layers=3, num_heads=8, dropout=0.05):
        super().__init__()
        self.modality_embedding = nn.Parameter(torch.zeros(1, 2, dim))
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": nn.MultiheadAttention(
                    embed_dim=dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "ln1": nn.LayerNorm(dim),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim * 4, dim),
                ),
                "ln2": nn.LayerNorm(dim),
            })
            for _ in range(num_layers)
        ])

    def forward(self, g, i):
        x = torch.stack([g, i], dim=1) + self.modality_embedding
        for layer in self.layers:
            attn_out, _ = layer["attn"](x, x, x, need_weights=False)
            x = layer["ln1"](x + attn_out)
            x = layer["ln2"](x + layer["ffn"](x))
        return x[:, 0, :], x[:, 1, :]


# ============================================================================
# TRUE MC-DROPOUT UNCERTAINTY-GUIDED GATING
# ============================================================================
class UncertaintyGuidedGate(nn.Module):
    def __init__(self, dim=384, num_classes=4, mc_passes=20, dropout_p=0.20):
        super().__init__()
        self.mc_passes = mc_passes
        self.dropout_p = dropout_p

        self.g_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_classes)
        )
        self.i_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_classes)
        )
        self.log_tau = nn.Parameter(torch.tensor(0.0))

    def _mc_uncertainty(self, features, head):
        # [T, B, D]
        mc_features = features.unsqueeze(0).expand(self.mc_passes, -1, -1)
        # Active even in model.eval(); trained in the same way during training.
        mc_features = F.dropout(mc_features, p=self.dropout_p, training=True)
        mc_logits = head(mc_features)
        mc_probs = F.softmax(mc_logits.float(), dim=-1)
        mean_probs = mc_probs.mean(dim=0)
        uncertainty = mc_probs.var(dim=0, unbiased=False).mean(dim=1, keepdim=True)
        return mean_probs, uncertainty

    def forward(self, g_f, i_f):
        g_mean_probs, u_g = self._mc_uncertainty(g_f, self.g_head)
        i_mean_probs, u_i = self._mc_uncertainty(i_f, self.i_head)

        uncertainties = torch.cat([u_g, u_i], dim=1)
        uncertainties_norm = uncertainties / (
            uncertainties.mean(dim=1, keepdim=True) + 1e-8
        )
        tau = F.softplus(self.log_tau) + 1e-6
        weights = F.softmax(-uncertainties_norm / tau, dim=1)

        g_weighted = g_f * weights[:, 0:1]
        i_weighted = i_f * weights[:, 1:2]

        # Auxiliary supervised heads
        g_aux_logits = self.g_head(g_f)
        i_aux_logits = self.i_head(i_f)

        return (
            g_weighted,
            i_weighted,
            g_aux_logits,
            i_aux_logits,
            u_g,
            u_i,
            weights,
            g_mean_probs,
            i_mean_probs,
        )


# ============================================================================
# COMPLETE MODEL
# ============================================================================
ABLATIONS = ("full", "image_only", "graph_only", "concat", "no_recal", "no_trans", "no_uncert", "single_gcn")


class BrainTumorClassifier(nn.Module):
    def __init__(self, num_classes=4, feature_dim=384, ablation="full"):
        super().__init__()
        assert ablation in ABLATIONS, ablation
        self.ablation = ablation
        single = ablation in ("image_only", "graph_only")
        fused = ablation not in ("image_only", "graph_only", "concat")
        if ablation != "image_only":
            self.g_enc = MultiPathGNN(out_channels=feature_dim, gcn_only=(ablation == "single_gcn"))
        if ablation != "graph_only":
            self.i_enc = CNNBackbone(out_channels=feature_dim)
        if fused and ablation != "no_recal":
            self.recal = DynamicRecalibration(channels=feature_dim)
        if fused and ablation != "no_trans":
            self.trans = CrossModalTransformer(dim=feature_dim, num_layers=3, num_heads=8)
        if fused:
            # no_uncert keeps the auxiliary heads (same losses) but fixes the gate at 0.5 / 0.5
            self.uncert = UncertaintyGuidedGate(
                dim=feature_dim,
                num_classes=num_classes,
                mc_passes=MC_PASSES,
                dropout_p=MC_DROPOUT,
            )
        self.clf = nn.Sequential(
            nn.Linear(feature_dim * (1 if single else 2), 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, gx, gei, gb, image, return_aux=False):
        a = self.ablation
        g_aux_logits = i_aux_logits = u_g = u_i = weights = g_mean_probs = i_mean_probs = None
        if a == "image_only":
            logits = self.clf(self.i_enc(image))
        elif a == "graph_only":
            logits = self.clf(self.g_enc(gx, gei, gb))
        else:
            g = self.g_enc(gx, gei, gb)
            i = self.i_enc(image)
            if a != "concat":
                if hasattr(self, "recal"):
                    g, i = self.recal(g, i)
                if hasattr(self, "trans"):
                    g, i = self.trans(g, i)
                if a == "no_uncert":
                    g_aux_logits = self.uncert.g_head(g)
                    i_aux_logits = self.uncert.i_head(i)
                    weights = torch.full((g.size(0), 2), 0.5, device=g.device, dtype=g.dtype)
                    g, i = g * 0.5, i * 0.5
                else:
                    (
                        g,
                        i,
                        g_aux_logits,
                        i_aux_logits,
                        u_g,
                        u_i,
                        weights,
                        g_mean_probs,
                        i_mean_probs,
                    ) = self.uncert(g, i)
            logits = self.clf(torch.cat([g, i], dim=1))

        if return_aux:
            return (
                logits,
                g_aux_logits,
                i_aux_logits,
                u_g,
                u_i,
                weights,
                g_mean_probs,
                i_mean_probs,
            )
        return logits


# ============================================================================
# DATASET
# ============================================================================
class BrainTumorDataset(Dataset):
    """
    The exact same transformed PIL image is used for graph construction and CNN input,
    so the two modalities remain aligned during training and TTA.
    """

    def __init__(
        self,
        root_dirs,
        graph_builder,
        pil_transform=None,
        tensor_transform=None,
    ):
        self.root_dirs = list(root_dirs)
        self.graph_builder = graph_builder
        self.pil_transform = pil_transform
        self.tensor_transform = tensor_transform

        self.classes = sorted([
            d for d in os.listdir(self.root_dirs[0])
            if os.path.isdir(os.path.join(self.root_dirs[0], d))
        ])
        self.c2i = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        for root_dir in self.root_dirs:
            for cls in self.classes:
                class_dir = os.path.join(root_dir, cls)
                if not os.path.isdir(class_dir):
                    continue
                for name in sorted(os.listdir(class_dir)):
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(class_dir, name), self.c2i[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")

        if self.pil_transform is not None:
            img = self.pil_transform(img)

        graph = self.graph_builder.create_graph(img)
        if self.tensor_transform is not None:
            image_tensor = self.tensor_transform(img)
        else:
            image_tensor = transforms.ToTensor()(img)

        return graph, image_tensor, label

    def get_labels(self):
        return [label for _, label in self.samples]


def collate_batch(batch):
    graphs, images, labels = zip(*batch)
    return (
        Batch.from_data_list(graphs),
        torch.stack(images),
        torch.tensor(labels, dtype=torch.long),
    )


# ============================================================================
# SHARED AUGMENTATION / PREPROCESSING
# ============================================================================
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)
TENSOR_TRANSFORM = transforms.Compose([transforms.ToTensor(), NORMALIZE])
VAL_PIL_TRANSFORM = transforms.Compose([transforms.Resize((224, 224))])


def get_train_pil_transform(epoch, max_epochs):
    progress = min(epoch / max(max_epochs * 0.5, 1), 1.0)
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.10 + 0.25 * progress,
            contrast=0.10 + 0.25 * progress,
            # v2: saturation/hue removed (exact no-ops on greyscale MRI stored as RGB)
        ),
    ])


# ============================================================================
# DETERMINISTIC 10-WAY TTA
# ============================================================================
RESIZE = transforms.Resize((224, 224))  # v2: identical resampling to training/validation


class IdentityTTA:
    def __call__(self, img):
        return RESIZE(img)


class HFlipTTA:
    def __call__(self, img):
        return TF.hflip(RESIZE(img))


class RotateTTA:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(RESIZE(img), self.angle, fill=0)


class BrightnessTTA:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_brightness(RESIZE(img), self.factor)


class ContrastTTA:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        return TF.adjust_contrast(RESIZE(img), self.factor)


class ZoomTTA:
    """v2 replacement for SaturationTTA (a no-op on greyscale): centre zoom by `factor`."""

    def __init__(self, factor):
        self.size = int(round(224 * factor))

    def __call__(self, img):
        return TF.center_crop(TF.resize(img, [self.size, self.size]), [224, 224])


class HFlipBrightnessTTA:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, img):
        img = TF.hflip(RESIZE(img))
        return TF.adjust_brightness(img, self.factor)


def get_tta_transforms():
    return [
        IdentityTTA(),
        HFlipTTA(),
        RotateTTA(+5),
        RotateTTA(-5),
        BrightnessTTA(0.90),
        BrightnessTTA(1.10),
        ContrastTTA(0.90),
        ContrastTTA(1.10),
        ZoomTTA(1.05),
        HFlipBrightnessTTA(1.10),
    ][:TTA_AUGMENTS]


# ============================================================================
# TRAINING / VALIDATION
# ============================================================================
def train_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, use_amp):
    model.train()
    total_correct, total_samples, total_loss = 0, 0, 0.0

    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"Epoch {epoch:2d}", leave=False, ncols=100)

    for graph_batch, images, labels in pbar:
        graph_batch = graph_batch.to(device)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            (
                logits,
                g_aux_logits,
                i_aux_logits,
                _,
                _,
                _,
                _,
                _,
            ) = model(
                graph_batch.x,
                graph_batch.edge_index,
                graph_batch.batch,
                images,
                return_aux=True,
            )

            main_loss = F.cross_entropy(
                torch.clamp(logits.float(), -20, 20),
                labels,
                label_smoothing=0.03,
            )
            if g_aux_logits is not None:
                g_aux_loss = F.cross_entropy(
                    torch.clamp(g_aux_logits.float(), -20, 20),
                    labels,
                    label_smoothing=0.03,
                )
                i_aux_loss = F.cross_entropy(
                    torch.clamp(i_aux_logits.float(), -20, 20),
                    labels,
                    label_smoothing=0.03,
                )
                aux_loss = 0.5 * (g_aux_loss + i_aux_loss)
                loss = main_loss + AUX_LOSS_WEIGHT * aux_loss
            else:  # single-modality / concat ablations have no auxiliary heads
                loss = main_loss

        if not torch.isfinite(loss):
            print("⚠️ Non-finite loss detected; skipping batch.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)
        pbar.set_postfix({
            "acc": f"{total_correct / max(total_samples, 1):.4f}",
            "loss": f"{total_loss / max(total_samples, 1):.4f}",
        })

    return (
        total_correct / max(total_samples, 1),
        total_loss / max(total_samples, 1),
    )


def validate(model, loader, device, use_amp):
    """Returns (accuracy, mean cross-entropy). The RNG is re-seeded so the gate's
    MC-dropout masks are identical every epoch (v1 validation was stochastic)."""
    model.eval()
    total_correct, total_samples, total_nll = 0, 0, 0.0
    fork_devices = [device] if device.type == "cuda" else []

    with torch.random.fork_rng(devices=fork_devices), torch.inference_mode():
        torch.manual_seed(SEED)
        for graph_batch, images, labels in loader:
            graph_batch = graph_batch.to(device)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(
                    graph_batch.x,
                    graph_batch.edge_index,
                    graph_batch.batch,
                    images,
                )

            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_nll += F.cross_entropy(logits.float(), labels, reduction="sum").item()
            total_samples += labels.size(0)

    return total_correct / max(total_samples, 1), total_nll / max(total_samples, 1)


# ============================================================================
# TTA EVALUATION
# ============================================================================
def evaluate_fold(model, test_ds, test_idx, labels_array, classes, device, fold, use_amp):
    model.eval()
    tta_transforms = get_tta_transforms()
    all_probs, all_gate = [], []
    test_indices = test_idx.tolist() if isinstance(test_idx, np.ndarray) else list(test_idx)
    test_subset = Subset(test_ds, test_indices)

    for tta_idx, tta_transform in enumerate(tta_transforms):
        print(f"TTA {tta_idx + 1}/{len(tta_transforms)}... ", end="", flush=True)
        test_subset.dataset.pil_transform = tta_transform

        test_loader = DataLoader(
            test_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=make_generator(SEED + 1000 + fold * 100 + tta_idx),
        )

        probs_this_tta, gate_this_tta = [], []
        fork_devices = [device] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices), torch.inference_mode():
            torch.manual_seed(SEED + 1000 + fold * 100 + tta_idx)
            for graph_batch, images, _ in test_loader:
                graph_batch = graph_batch.to(device)
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    out = model(
                        graph_batch.x,
                        graph_batch.edge_index,
                        graph_batch.batch,
                        images,
                        return_aux=True,
                    )
                logits, weights, u_g, u_i = out[0], out[5], out[3], out[4]
                probs_this_tta.append(F.softmax(logits.float(), dim=1).cpu().numpy())
                n = logits.size(0)
                gate = np.full((n, 4), np.nan)
                if weights is not None:
                    gate[:, 0:2] = weights.float().cpu().numpy()
                if u_g is not None:
                    gate[:, 2] = u_g.float().cpu().numpy().ravel()
                    gate[:, 3] = u_i.float().cpu().numpy().ravel()
                gate_this_tta.append(gate)

        all_probs.append(np.vstack(probs_this_tta))
        all_gate.append(np.vstack(gate_this_tta))
        print("✓")
        del test_loader
        gc.collect()

    final_probs = np.mean(np.stack(all_probs, axis=0), axis=0)
    gate_mean = np.mean(np.stack(all_gate, axis=0), axis=0)  # w_g, w_i, u_g, u_i averaged over TTA views
    preds = final_probs.argmax(axis=1)
    trues = labels_array[test_idx]

    test_acc = float((preds == trues).mean())
    precision, recall, f1, _ = precision_recall_fscore_support(
        trues, preds, average="macro", zero_division=0
    )
    ece_score, brier_score = calculate_uncertainty_metrics(final_probs, trues, num_bins=10)
    cm = confusion_matrix(trues, preds)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    report = classification_report(
        trues,
        preds,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": test_acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "ece": ece_score,
        "brier": brier_score,
        "cm": cm,
        "cm_norm": cm_norm,
        "report": report,
        "final_probs": final_probs,
        "gate": gate_mean,
        "preds": preds,
        "trues": trues,
    }


# ============================================================================
# SAVE OUTPUTS
# ============================================================================
def save_fold_outputs(result, full_dataset, test_idx, fold, extra=None):
    fold_num = fold + 1
    metrics = {
        "fold": fold_num,
        "accuracy": result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "ece": result["ece"],
        "brier": result["brier"],
        "num_test_samples": int(len(result["trues"])),
        "classes": list(full_dataset.classes),
        "mc_passes": MC_PASSES,
        "mc_dropout": MC_DROPOUT,
        "tta_augments": TTA_AUGMENTS,
        "seed": SEED,
        "cv_mode": CV_MODE,
        "ablation": ABLATION,
        "select_on": SELECT_ON,
        **(extra or {}),
    }

    with open(OUTPUT_DIR / f"{RUN_TAG}_fold_{fold_num}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(OUTPUT_DIR / f"{RUN_TAG}_fold_{fold_num}_classification_report.json", "w", encoding="utf-8") as f:
        json.dump(result["report"], f, indent=2)

    np.savetxt(
        OUTPUT_DIR / f"{RUN_TAG}_fold_{fold_num}_confusion_matrix.csv",
        result["cm"], delimiter=",", fmt="%d"
    )
    np.savetxt(
        OUTPUT_DIR / f"{RUN_TAG}_fold_{fold_num}_confusion_matrix_normalized.csv",
        result["cm_norm"], delimiter=",", fmt="%.8f"
    )

    pred_csv = OUTPUT_DIR / f"{RUN_TAG}_fold_{fold_num}_predictions.csv"
    with open(pred_csv, "w", encoding="utf-8") as f:
        header = [
            "sample_index", "path", "true_index", "true_class",
            "pred_index", "pred_class"
        ] + [f"prob_{c}" for c in full_dataset.classes] + ["w_graph", "w_image", "u_graph", "u_image", "group"]
        f.write(",".join(header) + "\n")

        for row_idx, ds_idx in enumerate(test_idx):
            path, true_idx = full_dataset.samples[int(ds_idx)]
            pred_idx = int(result["preds"][row_idx])
            probs = result["final_probs"][row_idx]
            values = [
                str(int(ds_idx)),
                '"' + path.replace('"', '""') + '"',
                str(int(true_idx)),
                full_dataset.classes[int(true_idx)],
                str(pred_idx),
                full_dataset.classes[pred_idx],
            ] + [f"{float(p):.10f}" for p in probs] + [
                "" if np.isnan(v) else f"{float(v):.8f}" for v in result["gate"][row_idx]
            ] + [str(full_dataset.groups[int(ds_idx)]) if getattr(full_dataset, "groups", None) is not None else ""]
            f.write(",".join(values) + "\n")

    cm = result["cm"]
    cm_norm = result["cm_norm"]
    per_class_recall = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=full_dataset.classes, yticklabels=full_dataset.classes,
        ax=axes[0, 0]
    )
    axes[0, 0].set_title(f"Confusion Matrix - Fold {fold_num} (Acc: {result['accuracy']:.4f})")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")

    sns.heatmap(
        cm_norm, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
        xticklabels=full_dataset.classes, yticklabels=full_dataset.classes,
        ax=axes[0, 1]
    )
    axes[0, 1].set_title("Normalized Confusion Matrix")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("True")

    axes[1, 0].bar(range(len(full_dataset.classes)), per_class_recall)
    axes[1, 0].set_xticks(range(len(full_dataset.classes)))
    axes[1, 0].set_xticklabels(full_dataset.classes, rotation=45, ha="right")
    axes[1, 0].set_title("Per-Class Recall")
    axes[1, 0].set_ylim([0.0, 1.0])
    for idx, value in enumerate(per_class_recall):
        axes[1, 0].text(idx, min(value + 0.02, 0.98), f"{value:.4f}", ha="center", va="bottom")

    summary = (
        f"FOLD {fold_num} RESULTS ({RUN_TAG})\n\n"
        f"Accuracy        : {result['accuracy']:.4f}\n"
        f"Macro Precision : {result['macro_precision']:.4f}\n"
        f"Macro Recall    : {result['macro_recall']:.4f}\n"
        f"Macro F1        : {result['macro_f1']:.4f}\n"
        f"ECE             : {result['ece']:.4f}\n"
        f"Brier Score     : {result['brier']:.4f}\n\n"
        f"MC passes       : {MC_PASSES}\n"
        f"TTA             : {TTA_AUGMENTS}-way\n"
        f"Seed            : {SEED}"
    )
    axes[1, 1].text(
        0.5, 0.5, summary, ha="center", va="center",
        fontsize=11, fontfamily="monospace", transform=axes[1, 1].transAxes
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{RUN_TAG}_fold_{fold_num}_results.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)
    print(f"💾 Saved fold {fold_num} outputs to: {OUTPUT_DIR}")


# ============================================================================
# DUPLICATE-AWARE GROUPS
# ============================================================================
AUG_BASE = re.compile(r"^([GMPN])_(\d+)")


_POPCOUNT_LUT = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)


def _popcount64(x):
    """Number of set bits of each uint64 element (numpy>=2 fast path, LUT fallback)."""
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    b = np.ascontiguousarray(x).view(np.uint8).reshape(*x.shape, 8)
    return _POPCOUNT_LUT[b].sum(axis=-1, dtype=np.int64)


def build_groups(samples, use_phash=True, max_hamming=5):
    """Group id per sample: offline-augmented copies (same G/M/P/N_### stem) share a group;
    with use_phash, images whose 64-bit perceptual hashes differ in <= max_hamming bits are
    merged as well (union-find). Writes groups.csv for auditing."""
    n = len(samples)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    stems = {}
    for idx, (path, label) in enumerate(samples):
        m = AUG_BASE.match(os.path.basename(path))
        if m:
            key = (label, m.group(1), int(m.group(2)))
            if key in stems:
                union(idx, stems[key])
            else:
                stems[key] = idx
    n_stem_unions = n - len({find(i) for i in range(n)})

    hashes = None
    cross_label_pairs = []
    if use_phash:
        import imagehash
        print("🔎 Computing perceptual hashes...", flush=True)
        hashes = np.zeros(n, dtype=np.uint64)
        for idx, (path, _) in enumerate(samples):
            with Image.open(path) as im:
                h = imagehash.phash(im.convert("L"))
            hashes[idx] = np.uint64(int(str(h), 16))
        labels_arr = np.array([lab for _, lab in samples])
        for start in range(0, n, 512):
            block = hashes[start:start + 512]
            dist = _popcount64(np.bitwise_xor(block[:, None], hashes[None, :]))
            ii, jj = np.nonzero(dist <= max_hamming)
            for a, b in zip(ii + start, jj):
                if a < b:
                    union(int(a), int(b))
                    if labels_arr[a] != labels_arr[b]:
                        cross_label_pairs.append((int(a), int(b), int(dist[a - start, b])))

    roots = [find(i) for i in range(n)]
    remap = {r: k for k, r in enumerate(sorted(set(roots)))}
    groups = np.array([remap[r] for r in roots], dtype=np.int64)
    sizes = np.bincount(groups)
    print(f"✅ Groups: {len(sizes)} for {n} images | merged by filename stem: {n_stem_unions} | "
          f"largest group: {sizes.max()} | groups with >1 image: {(sizes > 1).sum()}")
    if cross_label_pairs:
        print(f"⚠️ {len(cross_label_pairs)} near-duplicate pairs carry DIFFERENT labels (possible label noise); "
              f"see near_duplicate_label_conflicts.csv")
    if sizes.max() > 0.01 * n:
        print("⚠️ A very large group formed (>1% of images) - consider lowering PHASH_MAX_HAMMING.")

    with open(OUTPUT_DIR / "groups.csv", "w", encoding="utf-8") as f:
        f.write("sample_index,path,label,group,phash\n")
        for idx, (path, label) in enumerate(samples):
            ph = f"{int(hashes[idx]):016x}" if hashes is not None else ""
            f.write(f'{idx},"{path}",{label},{groups[idx]},{ph}\n')
    with open(OUTPUT_DIR / "near_duplicate_label_conflicts.csv", "w", encoding="utf-8") as f:
        f.write("path_a,label_a,path_b,label_b,hamming\n")
        for a, b, d in cross_label_pairs:
            f.write(f'"{samples[a][0]}",{samples[a][1]},"{samples[b][0]}",{samples[b][1]},{d}\n')
    return groups


def make_splits(labels, groups):
    if CV_MODE == "grouped":
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        return list(skf.split(np.zeros(len(labels)), labels, groups))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    return list(skf.split(np.zeros(len(labels)), labels))


def make_val_split(train_val_idx, labels, groups, fold):
    if CV_MODE == "grouped":
        n_inner = max(2, int(round(1.0 / VAL_SPLIT)))  # ~8% of the training groups
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=SEED + fold)
        tr, va = next(inner.split(np.zeros(len(train_val_idx)), labels[train_val_idx], groups[train_val_idx]))
        return train_val_idx[tr], train_val_idx[va]
    return train_test_split(
        train_val_idx,
        test_size=VAL_SPLIT,
        stratify=labels[train_val_idx],
        random_state=SEED + fold,
    )


def measure_complexity(model, sample_path, device):
    """Trainable parameters, FLOPs of one forward pass, and latency per image for
    (a) one forward pass and (b) the full 10-view TTA pipeline incl. graph construction."""
    out = {"params": int(sum(p.numel() for p in model.parameters()))}
    model.eval()
    gb_builder = GraphPyramid()
    with Image.open(sample_path) as im:
        img = im.convert("RGB")

    def prepare(view):
        pil = view(img)
        g = gb_builder.create_graph(pil)
        b = Batch.from_data_list([g]).to(device)
        return b, TENSOR_TRANSFORM(pil).unsqueeze(0).to(device)

    b, x = prepare(IdentityTTA())

    def fwd():
        with torch.inference_mode():
            return model(b.x, b.edge_index, b.batch, x)

    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            fwd()
        out["gflops_single_pass"] = fc.get_total_flops() / 1e9
    except Exception as exc:  # FLOP counting is best-effort
        out["gflops_single_pass"] = None
        print(f"FLOP count unavailable: {exc}")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(5):
        fwd()
    sync()
    t0 = time.perf_counter()
    for _ in range(30):
        fwd()
    sync()
    out["ms_single_forward"] = (time.perf_counter() - t0) / 30 * 1000

    t0 = time.perf_counter()
    for _ in range(30):
        gb_builder.create_graph(IdentityTTA()(img))
    out["ms_graph_construction_cpu"] = (time.perf_counter() - t0) / 30 * 1000

    views = get_tta_transforms()
    sync()
    t0 = time.perf_counter()
    for _ in range(5):
        for view in views:
            b, x = prepare(view)
            fwd()
    sync()
    out["ms_full_pipeline"] = (time.perf_counter() - t0) / 5 * 1000
    out["tta_views"] = len(views)
    out["mc_passes"] = MC_PASSES
    out["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    print("⏱️ Complexity:", json.dumps(out, indent=2))
    return out


# ============================================================================
# MAIN
# ============================================================================
def main():
    script_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"🖥️ Device: {device}\n")

    graph_builder = GraphPyramid()
    train_root = os.path.join(DATA_PATH, TRAIN_FOLDER)
    test_root = os.path.join(DATA_PATH, TEST_FOLDER)

    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Train folder not found: {train_root}")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test folder not found: {test_root}")

    # Merge the predefined Train/Test folders only to perform image-level
    # Stratified 5-Fold CV over the complete dataset.
    # This does NOT establish patient-level independence.
    full_dataset = BrainTumorDataset(
        root_dirs=[train_root, test_root],
        graph_builder=graph_builder,
        pil_transform=VAL_PIL_TRANSFORM,
        tensor_transform=TENSOR_TRANSFORM,
    )
    labels = np.asarray(full_dataset.get_labels(), dtype=np.int64)

    print(f"✅ Total images: {len(full_dataset)}")
    print(f"✅ Classes: {full_dataset.classes}\n")

    groups = None
    if CV_MODE == "grouped":
        groups = build_groups(full_dataset.samples, use_phash=USE_PHASH, max_hamming=PHASH_MAX_HAMMING)
    full_dataset.groups = groups
    splits = make_splits(labels, groups)

    # Save the complete fold assignment once (identical in every session for the same settings)
    with open(OUTPUT_DIR / f"fold_assignment_{CV_MODE}.csv", "w", encoding="utf-8") as f:
        f.write("sample_index,path,label,test_fold\n")
        test_fold = np.zeros(len(labels), dtype=np.int64)
        for k, (_, te) in enumerate(splits):
            test_fold[te] = k + 1
        for idx, (path, label) in enumerate(full_dataset.samples):
            f.write(f'{idx},"{path}",{label},{test_fold[idx]}\n')
    if groups is not None:
        for k, (tv, te) in enumerate(splits):
            shared = np.intersect1d(groups[tv], groups[te]).size
            print(f"Fold {k + 1}: test {len(te)} images, groups shared with training: {shared}")
            assert shared == 0, "group leakage between train and test"
    fold_results = []

    for fold, (train_val_idx, test_idx) in enumerate(splits):
        if FOLD_TO_RUN is not None and fold != FOLD_TO_RUN:
            continue

        print("\n" + "=" * 80)
        print(f"🚀 STARTING FOLD {fold + 1}/5")
        print("=" * 80)

        train_idx, val_idx = make_val_split(train_val_idx, labels, groups, fold)
        if groups is not None:
            assert np.intersect1d(groups[train_idx], groups[val_idx]).size == 0
        print(f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

        train_ds = copy.deepcopy(full_dataset)
        val_ds = copy.deepcopy(full_dataset)
        test_ds = copy.deepcopy(full_dataset)
        train_ds.pil_transform = get_train_pil_transform(1, EPOCHS)
        val_ds.pil_transform = VAL_PIL_TRANSFORM
        test_ds.pil_transform = VAL_PIL_TRANSFORM

        train_subset = Subset(train_ds, train_idx.tolist())
        val_subset = Subset(val_ds, val_idx.tolist())

        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_batch,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=make_generator(SEED + fold * 10 + 1),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_batch,
            num_workers=NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            worker_init_fn=seed_worker,
            generator=make_generator(SEED + fold * 10 + 2),
        )

        model = BrainTumorClassifier(
            num_classes=len(full_dataset.classes), feature_dim=384, ablation=ABLATION
        ).to(device)

        attention_params, other_params = [], []
        for name, param in model.named_parameters():
            if any(k in name.lower() for k in ["gat", "trans", "attn"]):
                attention_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": 3e-4},
                {"params": attention_params, "lr": 6e-5},
            ],
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[3e-4, 6e-5],
            epochs=EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.10,
            anneal_strategy="cos",
        )
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        best_val_acc = -1.0
        best_score = -float("inf")
        patience = 0
        best_model_state = None
        best_epoch = None
        history = []

        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()
            # persistent_workers=False ensures this updated transform is used each epoch.
            train_ds.pil_transform = get_train_pil_transform(epoch, EPOCHS)

            train_acc, train_loss = train_epoch(
                model, train_loader, optimizer, scheduler,
                scaler, device, epoch, use_amp
            )
            val_acc, val_loss = validate(model, val_loader, device, use_amp)
            elapsed_min = (time.time() - epoch_start) / 60.0
            history.append({"epoch": epoch, "train_acc": train_acc, "train_loss": train_loss,
                             "val_acc": val_acc, "val_loss": val_loss, "minutes": elapsed_min})
            score = -val_loss if SELECT_ON == "val_loss" else val_acc

            if score > best_score:
                best_score = score
                best_val_acc = val_acc
                best_epoch = epoch
                patience = 0
                best_model_state = copy.deepcopy(model.state_dict())
                print(
                    f"✓ Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | "
                    f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | {elapsed_min:.1f}m"
                )
            else:
                patience += 1
                if epoch % 5 == 0 or patience >= EARLY_STOP_PATIENCE:
                    print(
                        f"  Epoch {epoch:2d} | Train Acc: {train_acc:.4f} | "
                        f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f} | Pat: {patience}/{EARLY_STOP_PATIENCE}"
                    )

            if patience >= EARLY_STOP_PATIENCE:
                print("⏹️ Early stopping triggered.")
                break

            if epoch % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if best_model_state is None:
            raise RuntimeError("No valid model checkpoint was produced.")

        print(
            f"\n✅ Fold {fold + 1} training complete. "
            f"Best epoch: {best_epoch}, Best Val Acc: {best_val_acc:.4f}\n"
        )

        model_path = OUTPUT_DIR / f"{RUN_TAG}_model_fold_{fold + 1}.pth"
        torch.save(best_model_state, model_path)
        model.load_state_dict(best_model_state)

        result = evaluate_fold(
            model=model,
            test_ds=test_ds,
            test_idx=test_idx,
            labels_array=labels,
            classes=full_dataset.classes,
            device=device,
            fold=fold,
            use_amp=use_amp,
        )

        print(f"\n🎯 FOLD {fold + 1} RESULTS")
        print(f"Accuracy        : {result['accuracy']:.4f}")
        print(f"Macro Precision : {result['macro_precision']:.4f}")
        print(f"Macro Recall    : {result['macro_recall']:.4f}")
        print(f"Macro F1        : {result['macro_f1']:.4f}")
        print(f"ECE             : {result['ece']:.4f}")
        print(f"Brier Score     : {result['brier']:.4f}\n")

        extra = {
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "epochs_run": len(history),
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "history": history,
        }
        if ABLATION in ("full", "no_uncert", "no_recal", "no_trans", "single_gcn"):
            extra["tau"] = float(F.softplus(model.uncert.log_tau).item())
        if hasattr(model, "g_enc") and ABLATION != "single_gcn":
            extra["softmax_alpha"] = F.softmax(model.g_enc.alpha.detach().float(), 0).tolist()
            extra["softmax_beta"] = F.softmax(model.g_enc.beta.detach().float(), 0).tolist()
        if MEASURE_COMPLEXITY:
            extra["complexity"] = measure_complexity(model, full_dataset.samples[int(test_idx[0])][0], device)
        save_fold_outputs(result, full_dataset, test_idx, fold, extra=extra)
        fold_results.append({
            "fold": fold + 1,
            "accuracy": result["accuracy"],
            "macro_precision": result["macro_precision"],
            "macro_recall": result["macro_recall"],
            "macro_f1": result["macro_f1"],
            "ece": result["ece"],
            "brier": result["brier"],
        })

        del model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate only if all five folds were run in this same execution.
    if FOLD_TO_RUN is None and len(fold_results) == 5:
        def arr(key):
            return np.array([r[key] for r in fold_results], dtype=np.float64)

        accs = arr("accuracy")
        precs = arr("macro_precision")
        recalls = arr("macro_recall")
        f1s = arr("macro_f1")
        eces = arr("ece")
        briers = arr("brier")

        summary = {
            "accuracy_mean": float(accs.mean()),
            "accuracy_std": float(accs.std(ddof=1)),
            "macro_precision_mean": float(precs.mean()),
            "macro_precision_std": float(precs.std(ddof=1)),
            "macro_recall_mean": float(recalls.mean()),
            "macro_recall_std": float(recalls.std(ddof=1)),
            "macro_f1_mean": float(f1s.mean()),
            "macro_f1_std": float(f1s.std(ddof=1)),
            "ece_mean": float(eces.mean()),
            "ece_std": float(eces.std(ddof=1)),
            "brier_mean": float(briers.mean()),
            "brier_std": float(briers.std(ddof=1)),
        }

        print("\n" + "=" * 80)
        print("FINAL 5-FOLD CROSS-VALIDATION RESULTS")
        print("=" * 80)
        print(f"Accuracy        : {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}")
        print(f"Macro Precision : {summary['macro_precision_mean']:.4f} ± {summary['macro_precision_std']:.4f}")
        print(f"Macro Recall    : {summary['macro_recall_mean']:.4f} ± {summary['macro_recall_std']:.4f}")
        print(f"Macro F1        : {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f}")
        print(f"ECE             : {summary['ece_mean']:.4f} ± {summary['ece_std']:.4f}")
        print(f"Brier Score     : {summary['brier_mean']:.4f} ± {summary['brier_std']:.4f}")
        print("=" * 80)

        with open(OUTPUT_DIR / f"{RUN_TAG}_five_fold_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    total_hours = (time.time() - script_start) / 3600.0
    print(f"\n⏱️ Total runtime: {total_hours:.2f} hours")
    print(f"📁 Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
