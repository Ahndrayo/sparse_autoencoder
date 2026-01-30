# --- helper: create analysis_data/<timestamp>_run-XXX ---
import re, time
from pathlib import Path

def make_analysis_run_dir(repo_root: str) -> Path:
    base = Path(repo_root) / "analysis_data"
    base.mkdir(parents=True, exist_ok=True)

    # discover next run id
    ids = []
    for p in base.glob("*_run-*"):
        m = re.search(r"_run-(\d+)", p.name)
        if m:
            try:
                ids.append(int(m.group(1)))
            except ValueError:
                pass
    next_id = (max(ids) + 1) if ids else 1
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")  # filesystem-safe

    run_dir = base / f"{ts}_run-{next_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


__all__ = ["make_analysis_run_dir"]
