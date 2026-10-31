"""Run a repository v10 GPU experiment with an explicit module allowlist."""
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'src'), str(root)]
allowed = {'eval.run_trajectory_v10', 'training.run_neural_trajectory_v10', 'training.run_uniform_memory_v10'}
if len(sys.argv) < 2 or sys.argv[1] not in allowed:
    raise SystemExit('Specify an allowed v10 evaluation module')
module = sys.argv.pop(1)
runpy.run_module(module, run_name='__main__')
