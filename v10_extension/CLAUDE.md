# V10 project instructions

Latest scope (2026-09-20): only modify and run inside the existing `multimodel` project. New code lives in `v10_extension/`; start with `bash run_v10.sh` from the parent project root. Never create a sibling project or edit global Codex/Claude/shell/Conda settings. Preserve root legacy code and artifacts; this entry applies only to the extension. Deployment instructions: `../README_V10_INPLACE.md`.

Before substantive work, explicitly read `docs/engineering-habits.md`, `docs/HANDOFF.md`, `docs/DESIGN.md`, and `README_V10.md`. Preserve V9 source, weights and results. Use unique run directories, bounded jobs, independent verification and a shared handoff. Respect host/user ownership and current authorization. Do not use test images for fitting or tuning. Claude Code entry loading has not been independently tested.
