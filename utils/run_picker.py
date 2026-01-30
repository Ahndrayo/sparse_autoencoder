import re
from pathlib import Path


def get_run_by_id(analysis_dir: Path, run_id: int) -> Path:
    """
    Find a run folder by its ID, ignoring any suffix after the ID.

    Args:
        analysis_dir: Path to analysis_data directory
        run_id: Run number (e.g., 81 for run-081)

    Returns:
        Path to the run folder

    Raises:
        FileNotFoundError: If no matching run is found
    """
    pattern = f"*_run-{run_id:03d}*"
    matches = list(analysis_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError(
            f"No run folder found with ID {run_id} in {analysis_dir}"
        )

    if len(matches) > 1:
        exact = [m for m in matches if m.name.endswith(f"_run-{run_id:03d}")]
        if exact:
            return exact[0]
        print(
            f"Warning: Multiple folders found for run-{run_id:03d}, using {matches[0].name}"
        )

    return matches[0]


def get_latest_run(analysis_dir: Path) -> Path:
    """
    Get the most recently modified run folder.

    Args:
        analysis_dir: Path to analysis_data directory

    Returns:
        Path to the latest run folder

    Raises:
        FileNotFoundError: If no run folders exist
    """
    run_folders = []
    for folder in analysis_dir.iterdir():
        if folder.is_dir() and "_run-" in folder.name:
            match = re.search(r"_run-(\d+)", folder.name)
            if match:
                run_folders.append(folder)

    if not run_folders:
        raise FileNotFoundError(f"No run folders found in {analysis_dir}")

    return max(run_folders, key=lambda p: p.stat().st_mtime)


__all__ = ["get_run_by_id", "get_latest_run"]
