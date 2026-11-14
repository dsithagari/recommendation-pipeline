"""Dependency-aware orchestration entrypoint; schedule externally using cron/Task Scheduler."""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
# A single orchestrator enforces stage ordering, abort-on-failure, and run logging.
subprocess.run([sys.executable,str(ROOT/'src'/'pipeline.py'),'--stage','all'],cwd=ROOT,check=True)
