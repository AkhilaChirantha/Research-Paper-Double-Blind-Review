from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_DATASET_PATH = REPO_ROOT / "data" / "PeerRead Set" / "peerread_features.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "peerread_acceptance_model.json"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"


DECISION_THRESHOLDS = {
    "accept": 0.65,
    "reject": 0.35,
}
