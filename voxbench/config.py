"""
Bridge2AI-Voice — constants, hyperparameters, and tier configurations.
Edit this file to change diseases, modalities, or training settings.
"""

# Modality keys — order is fixed; indices must stay consistent across all code.
# MFCC is restored as an explicit experimental branch so we can test whether
# the MARVEL-style cepstral inductive bias closes the tier-2 unified gap.
MODALITY_KEYS = ["spec", "mfcc", "mel", "ppg", "ema", "pros", "static"]

# Number of modalities; stored in HDF5 for schema versioning.
N_MODALITIES = len(MODALITY_KEYS)

T_MAX = 300  # ~6 s at 50 fps; PPG uses same after ÷2 downsampling

# Static feature dimension: 131 openSMILE/Praat scalars + 6 macro prosodic scalars
# (phonation_ratio, mean/std voiced run, mean/std silence run, n_pauses_per_100f).
# Update this constant if the static feature set changes; it propagates to BranchG.
N_STATIC_FEATURES = 137

# ── Disease lists ─────────────────────────────────────────────────────────────

# Core 12 diseases used in legacy tiers 1–5
CORE_DISEASES = [
    "parkinsons",
    "airway_stenosis",
    "adhd",
    "ptsd",
    "laryngeal_dystonia",
    "cognitive_impairment",
    "vf_paralysis",
    "depression",
    "psychiatric_history",
    "benign_lesions",
    "mtd",
    "chronic_cough",
]

# Full ordered list of all 21 diagnosis labels
DISEASE_LIST = [
    "adhd",
    "airway_stenosis",
    "als",
    "anxiety",
    "benign_lesions",
    "bipolar_disorder",
    "cognitive_impairment",
    "control",
    "copd_asthma",
    "depression",
    "glottic_insufficiency",
    "laryngeal_cancer",
    "laryngeal_dystonia",
    "laryngitis",
    "mtd",
    "parkinsons",
    "precancerous_lesions",
    "psychiatric_history",
    "ptsd",
    "chronic_cough",
    "vf_paralysis",
]

DISEASE_FILES = {
    "adhd":                 "adhd_adult.tsv",
    "airway_stenosis":      "airway_stenosis.tsv",
    "als":                  "amyotrophic_lateral_sclerosis.tsv",
    "anxiety":              "anxiety.tsv",
    "benign_lesions":       "benign_lesions.tsv",
    "bipolar_disorder":     "bipolar_disorder.tsv",
    "cognitive_impairment": "cognitive_impairment.tsv",
    "control":              "control.tsv",
    "copd_asthma":          "copd_and_asthma.tsv",
    "depression":           "depression.tsv",
    "glottic_insufficiency":"glottic_insufficiency.tsv",
    "laryngeal_cancer":     "laryngeal_cancer.tsv",
    "laryngeal_dystonia":   "laryngeal_dystonia.tsv",
    "laryngitis":           "laryngitis.tsv",
    "mtd":                  "muscle_tension_dysphonia.tsv",
    "parkinsons":           "parkinsons_disease.tsv",
    "precancerous_lesions": "precancerous_lesions.tsv",
    "psychiatric_history":  "psychiatric_history.tsv",
    "ptsd":                 "ptsd_adult.tsv",
    "chronic_cough":        "unexplained_chronic_cough.tsv",
    "vf_paralysis":         "unilateral_vocal_fold_paralysis.tsv",
}

# ── Tier configs ──────────────────────────────────────────────────────────────
# 4-tier design:
#   Tier 1 — data-rich + easy to predict (physiological / structural diseases
#             with strong acoustic biomarkers, ≥40 participants each)
#   Tier 2 — Tier 1 + data-rich but hard (psychiatric diseases that require
#             semantic / language cues that voice features alone barely capture)
#   Tier 3 — Tier 1 + data-sparse but easy (rare structural disorders whose
#             physiology is well-captured by acoustics but train size is tiny)
#   Tier 4 — all 21 diagnosis labels
TIER_DISEASES = {
    1: [
        "parkinsons", "airway_stenosis", "laryngeal_dystonia", "vf_paralysis",
        "cognitive_impairment", "benign_lesions", "mtd", "chronic_cough",
    ],
    2: [
        "parkinsons", "airway_stenosis", "laryngeal_dystonia", "vf_paralysis",
        "cognitive_impairment", "benign_lesions", "mtd", "chronic_cough",
        "adhd", "ptsd", "psychiatric_history", "depression",
    ],
    3: [
        "parkinsons", "airway_stenosis", "laryngeal_dystonia", "vf_paralysis",
        "cognitive_impairment", "benign_lesions", "mtd", "chronic_cough",
        "precancerous_lesions", "glottic_insufficiency", "laryngitis",
        "laryngeal_cancer", "als", "copd_asthma",
    ],
    4: DISEASE_LIST,   # all 21 diagnosis labels
}

# Modalities enabled per tier.  All 6 modalities are enabled for every tier.
_ALL_MODS = {k: True for k in MODALITY_KEYS}
TIER_MODALITIES = {
    1: _ALL_MODS,
    2: _ALL_MODS,
    3: _ALL_MODS,
    4: _ALL_MODS,
}

# ── Hyperparameters + paths ───────────────────────────────────────────────────
CONFIG = {
    # Paths (inside Modal volume mounted at /data)
    "data_root":    "/data/physionet.org/files/b2ai-voice/3.0.0",
    "hdf5_path":    "/data/b2ai_voice_v3.h5",
    "ckpt_dir":     "/data/checkpoints",

    # Architecture
    "d_model":              512,
    "nhead":                8,
    "n_transformer_layers": 2,
    "dropout_fusion":       0.3,
    "dropout_head":         0.5,
    "head_hidden_dim":      256,
    "modality_dropout_prob":0.10,

    # Optimiser
    "batch_size":           128,
    "lr_backbone":          1e-4,   # EfficientNet / ResNet pretrained weights
    "lr_new":               3e-4,   # Transformer, new branches, heads
    "lr_warmup_epochs":     10,     # Linear LR warmup before cosine decay
    "cosine_t_max":         80,     # Cosine annealing period (epochs)
    "accumulation_steps":   1,      # Effective batch = batch_size × steps = 128
    "weight_decay":         1e-4,
    "grad_clip":            1.0,

    # Training loop
    "max_epochs":           150,
    "patience":             30,
    "ema_decay":            0.998,  # EMA weight tracking for eval/checkpoint
    "seed":                 42,

    # Loss
    "pos_weight_min":       1.0,
    "pos_weight_max":       30.0,
    "label_smoothing":      0.05,   # BCE targets: 0→0.025, 1→0.975
    "mixup_alpha":          0.2,    # MixUp beta; 0 = disabled; applied after warmup
    "focal_gamma":          1.0,    # Focal loss exponent; 0 = standard weighted BCE

    # Numerical stability
    "norm_std_floor":       1e-3,
    "feature_clip":         12.0,
    "amp_enabled":          False,  # BF16 AMP; off by default for stability

    # Resume / cache
    "auto_resume":          False,  # Avoid inheriting stale checkpoints by default
    "use_cached_norm_stats":False,

    # Per-disease task filtering for AUROC (empty = use all recordings)
    "task_filter_keywords": {},

    # Early-stopping macro AUROC: exclude diseases with fewer than this many
    # positive val patients (prevents rare-disease noise from T3/T4 driving
    # premature stopping; 0 = include all diseases as before)
    "es_min_val_patients":  10,

    # Augmentation (training only; applied to all 2-D spectral features)
    "augmentation": {
        "n_time_masks":    2,
        "time_mask_ratio": 0.20,
        "n_freq_masks":    2,
        "freq_mask_ratio": 0.15,
        "noise_std":       0.05,    # Gaussian noise added to mel
    },

    # Data split
    "train_frac":           0.70,
    "val_frac":             0.15,

    # Preprocessing (always build HDF5 with all modalities)
    "preprocess_modalities": {k: True for k in MODALITY_KEYS},
    "T_MAX":                T_MAX,
    "n_workers":            12,
}

TRAIN_PROFILES = {
    "default": {},
    "fast_main": {
        "amp_enabled": True,
        "ema_decay": 0.0,
        "use_cached_norm_stats": True,
        "n_workers": 8,
    },
    "fast_main_debug": {
        "amp_enabled": True,
        "ema_decay": 0.0,
        "use_cached_norm_stats": True,
        "n_workers": 6,
        "max_epochs": 20,
        "patience": 8,
    },
}
