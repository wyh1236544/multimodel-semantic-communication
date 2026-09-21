# V6.1 server adaptation

This bundle targets the server package set that already contains:

- `torch==1.12.0`
- `torchvision==0.13.0`
- `torchaudio==0.12.0`
- NumPy, SciPy, Pillow, scikit-image, PyYAML, matplotlib, tqdm, pandas,
  safetensors, and the other basic utilities used by the model files.

## Install the missing V6.1 dependencies

Activate the server environment first, then run:

```bash
# Existing server environment: keep its current huggingface-hub version.
python -m pip install -r requirements_v61_existing_server.txt
python check_v61_environment.py
```

For a clean isolated environment, use `requirements_v61_torch112.txt`; that
file pins `huggingface-hub` to a version aligned with the older Diffusers stack.

The new dependencies are:

- `diffusers==0.20.2`: Stable Diffusion `AutoencoderKL` VAE;
- `transformers==4.30.2`: CLIP text and image models;
- `accelerate==0.21.0`: Hugging Face model/runtime support;
- `huggingface-hub==0.16.4`: version compatible with the pinned stack;
- `tokenizers==0.13.3` and `regex`: tokenizer dependencies.

The file intentionally does not reinstall the existing Torch stack. If a new
virtual environment is created without access to system site packages, install
the matching Torch/vision/audio trio separately before this requirements file.

## Model weights

The VAE and CLIP checkpoints are not included here. Configure each model path
to a local Hugging Face cache or download them on the server first. The VAE
path must contain a Diffusers-compatible `AutoencoderKL` checkpoint.

## Scope limitation

The uploaded archive contains model files, result files, and reports, but not
the complete `v61_gate_train.py`, evaluation scripts, configuration tree,
dataset loader, and loss modules. Therefore this bundle adapts the available
model layer and environment entry point; it is not yet a complete standalone
training project. The complete source directory is required to produce a fully
runnable training package.
