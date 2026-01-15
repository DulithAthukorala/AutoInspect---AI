import os
from pathlib import Path
from huggingface_hub import hf_hub_download


def ensure_weights() -> str:
    """
    Ensure model weights exist locally.
    Priority:
    1) If MODEL_PATH exists -> use it
    2) Else download from HF_REPO_ID + HF_FILENAME into /app/models (volume-cached)
    """
    repo_id = os.getenv("HF_REPO_ID")
    filename = os.getenv("HF_FILENAME", "best.pt")

    local_path = Path(os.getenv("MODEL_PATH", f"/app/models/{filename}"))

    # if someone sets MODEL_PATH to a dir by mistake
    if local_path.exists() and local_path.is_dir():
        local_path = local_path / filename

    # already there (volume cached)
    if local_path.exists():
        return str(local_path)

    if not repo_id:
        raise RuntimeError("HF_REPO_ID not set and weights not found locally.")

    local_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_path.parent),
        local_dir_use_symlinks=False,
    )

    return str(Path(downloaded))
