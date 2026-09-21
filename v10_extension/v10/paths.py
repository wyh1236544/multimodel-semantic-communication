"""Write boundary for the in-project release. Never create an external path."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_path(value, root=None):
    root = Path(root or PROJECT_ROOT).resolve()
    path = Path(os.path.abspath(str(value)))
    try:
        parts = path.relative_to(root).parts
        path.resolve().relative_to(root)
    except ValueError:
        raise ValueError('V10 writes must stay inside multimodel: '+str(path))
    if not parts:
        raise ValueError('The project root cannot be an output target')
    current = root
    for part in parts:
        current = current/part
        if current.is_symlink():
            raise ValueError('Refusing a symlink in a writable path: '+str(current))
    return path


def local_environment(root=None):
    """Child-process-only settings; no shell startup or global config edits."""
    root = Path(root or PROJECT_ROOT).resolve()
    cache = write_path(root/'.cache/v10', root)
    env = dict(os.environ)
    env.pop('SUDO_PW', None)
    folders = {
        'TMPDIR': cache/'tmp', 'TMP': cache/'tmp', 'TEMP': cache/'tmp',
        'HF_HOME': cache/'huggingface', 'HF_HUB_CACHE': cache/'huggingface/hub',
        'HUGGINGFACE_HUB_CACHE': cache/'huggingface/hub',
        'TRANSFORMERS_CACHE': cache/'huggingface/transformers',
        'HF_MODULES_CACHE': cache/'huggingface/modules',
        'TORCH_HOME': cache/'torch', 'TORCH_EXTENSIONS_DIR': cache/'torch/extensions',
        'TORCHINDUCTOR_CACHE_DIR': cache/'torch/inductor',
        'TRITON_CACHE_DIR': cache/'triton', 'CUDA_CACHE_PATH': cache/'cuda',
        'XDG_CACHE_HOME': cache, 'MPLCONFIGDIR': cache/'matplotlib',
    }
    for name, folder in folders.items():
        write_path(folder, root).mkdir(parents=True, exist_ok=True)
        env[name] = str(folder)
    env.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', PYTHONUNBUFFERED='1',
               PYTHONDONTWRITEBYTECODE='1')
    return env
