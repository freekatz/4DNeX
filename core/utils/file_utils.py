import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union

from accelerate.logging import get_logger

from core.constants import LOG_LEVEL, LOG_NAME


logger = get_logger(LOG_NAME, LOG_LEVEL)


def find_files(dir: Union[str, Path], prefix: str = "step-") -> List[str]:
    if not isinstance(dir, Path):
        dir = Path(dir)
    if not dir.exists():
        return []
    entries = os.listdir(dir.as_posix())
    entries = [c for c in entries if c.startswith(prefix)]
    # Extract trailing integer from names like "step-000200" or "checkpoint-200"
    entries = sorted(entries, key=lambda x: int(x.rsplit("-", 1)[-1]))
    entries = [dir / c for c in entries]
    return entries


def delete_files(dirs: Union[str, List[str], Path, List[Path]]) -> None:
    if not isinstance(dirs, list):
        dirs = [dirs]
    dirs = [Path(d) if isinstance(d, str) else d for d in dirs]
    logger.info(f"Deleting files: {dirs}")
    for dir in dirs:
        if not dir.exists():
            continue
        shutil.rmtree(dir, ignore_errors=True)


def string_to_filename(s: str) -> str:
    return (
        s.replace(" ", "-")
        .replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(",", "-")
        .replace(";", "-")
        .replace("!", "-")
        .replace("?", "-")
    )
