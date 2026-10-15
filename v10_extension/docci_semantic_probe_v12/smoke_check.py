"""CPU smoke acceptance BEFORE main; does not relax any text quality gate."""
import sys
from pathlib import Path
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT)); sys.path.insert(0,str(EXT.parent))
from docci_semantic_probe_v12.report import verify
from docci_semantic_probe_v12.core import atomic_json,digest

if __name__=='__main__':
    root=Path(sys.argv[1]).resolve(strict=True); result=verify(root,smoke_only=True)
    atomic_json(root/'SMOKE_VERIFIED.json',dict(independent_CPU_verification=True,
        smoke_run_identity_sha256=digest(root/'smoke/run_identity.json'),smoke_weights_NEVER_reused=True,
        tasks=list(result['task_results']['smoke']),main_training_NOT_certified=True))
    print('Authentic saved-parent smoke and CPU verification passed; main must start again from approved parents.',flush=True)
