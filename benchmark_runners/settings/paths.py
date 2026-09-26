from pathlib import Path

# .../sxz4github
REPO_ROOT = Path(__file__).resolve().parents[1]

# .../SciDoc
WORKSPACE_ROOT = REPO_ROOT.parent

# .../SciDoc/data
DATA_ROOT = WORKSPACE_ROOT / "data"

# .../SciDoc/data/pdfs
PDF_ROOT = DATA_ROOT / "pdfs"

# .../SciDoc/models
MODEL_ROOT = WORKSPACE_ROOT / "models"
