"""Entry point for Vercel (setup: README.md). On your own computer: `python -m ffcalc serve`.

Vercel loads `app` from here. `python app.py` is Vercel's build step: it learns past seasons
for the test league's scoring, so the first visit is quick.
"""

import os
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEAGUE = "1400221038771437568"

if not os.access(HERE, os.W_OK):  # on Vercel the app folder is read-only; only /tmp is writable
    # shortcut: every fresh copy refetches caches past their age limit (the players list after
    # a day), so Sleeper may see a few fetches a day; add shared storage if usage grows
    work = Path(tempfile.gettempdir()) / "ffcalc"
    if (HERE / "data").is_dir() and not work.exists():
        # copytree keeps file times, so caches made at build time still expire on schedule
        shutil.copytree(HERE / "data", work)
        for folder, _, _ in os.walk(work):  # copytree also copies the bundle's read-only mode
            os.chmod(folder, 0o755)
    os.environ.setdefault("FFCALC_DATA", str(work))
os.environ.setdefault("FFCALC_HOSTED", "1")
os.environ.setdefault("FFCALC_HEAVY_JOBS", "1")  # 2 GB per copy on Vercel; a grade peaks ~1.1 GB

from ffcalc.api import app  # noqa: E402

__all__ = ["app"]

if __name__ == "__main__":
    from ffcalc import engine

    try:
        print(f"tables: {engine.prepare(LEAGUE).tables.source}")
    except Exception as e:  # a data source being down must not fail the deploy
        print(f"history builds on first visit instead ({e})")
