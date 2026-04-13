"""Configuration constants for Constitutional BioGuard."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CONSTITUTION_DIR = PROJECT_ROOT / "constitution"
CONSTITUTION_FILE = CONSTITUTION_DIR / "biosafety_constitution.yaml"
CONFIGS_DIR = PROJECT_ROOT / "configs"

DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_AUGMENTED = DATA_DIR / "augmented"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_EXTERNAL = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = RESULTS_DIR / "reports"

# ── BioThreat-Eval paths (external validation data) ─────────────────────────
import os as _os
BIOTHREAT_EVAL_DIR = Path(
    _os.environ.get(
        "BIOTHREAT_EVAL_DIR",
        str(PROJECT_ROOT.parent.parent / "New_Science" / "biothreat_eval"),
    )
)
BIOTHREAT_QUERY_BANK = BIOTHREAT_EVAL_DIR / "data" / "raw" / "query_bank.json"
BIOTHREAT_EVAL_RESULTS = BIOTHREAT_EVAL_DIR / "data" / "processed"

# ── Constitution ─────────────────────────────────────────────────────────────
NSABB_CATEGORIES = [
    "enhance_harm",
    "disrupt_immunity",
    "confer_resistance",
    "increase_stability",
    "alter_host_range",
    "enhance_susceptibility",
    "generate_reconstruct",
]
MIN_RULES_PER_CATEGORY = 6
TARGET_RULES_PER_CATEGORY = 8  # aim for 8-10 per category

# ── Synthetic Data Generation ────────────────────────────────────────────────
EXAMPLES_PER_RULE_PERMITTED = 15
EXAMPLES_PER_RULE_RESTRICTED = 15
EXAMPLES_PER_RULE_BOUNDARY = 5
GENERATION_MODEL = "claude-sonnet-4-20250514"   # core synthetic examples (quality matters)
AUGMENT_MODEL = "claude-haiku-4-5-20251001"     # translation, formality, benign (simpler tasks)
GENERATION_TEMPERATURE = 0.8
GENERATION_MAX_TOKENS = 6000
GENERATION_RPM = 40  # requests per minute

# ── Augmentation ─────────────────────────────────────────────────────────────
AUGMENT_LANGUAGES = ["es", "fr", "zh", "ar", "ru"]
AUGMENT_JAILBREAK_TEMPLATES = 10
AUGMENT_FORMALITY_LEVELS = 3  # casual, technical, formal

# ── Training (DeBERTa) ──────────────────────────────────────────────────────
DEBERTA_MODEL_NAME = "microsoft/deberta-v3-base"
DEBERTA_MAX_SEQ_LENGTH = 512
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ── Evaluation ───────────────────────────────────────────────────────────────
TARGET_FPR = 0.02  # < 2% false positive rate
TARGET_F1 = 0.90
TARGET_ASR = 0.15  # mean attack success rate < 15%
OVERREFUSAL_HOLDOUT_RATIO = 0.20  # reserve benign-only examples for FPR eval

# ── Figures ──────────────────────────────────────────────────────────────────
FIGURE_DPI = 300
FIGURE_FONT_SIZE = 10
