# Project Claude Guidelines

All Claude Code sessions working on this project must strictly comply with the engineering habits and multi-agent collaboration protocol:

1. **Read Guidelines First**: Before undertaking substantive work, explicitly read:
   - `AGENT_ENGINEERING_GUIDELINES.md` (General engineering & git synchronization rules)
   - `v10_extension/docs/HANDOFF.md` (Current task status & next steps)
   - `v10_extension/docs/DESIGN.md` (Communication physics & architecture specs)
2. **Non-destructive Inplace Extension**:
   - Do NOT overwrite or delete legacy V6~V9 code, weights, configurations, or experimental outputs.
   - Work within `v10_extension/` or designated new modules. Output experiments to unique directories (`experiment_results/v10_*`).
3. **Git & GitHub Synchronization**:
   - Follow atomic semantic commits.
   - If offline / network restricted, generate a copy-pasteable bash snippet for the human host with exact files, commit message, and push instructions.
4. **Physical Reality & Scientific Honesty**:
   - Mock and CPU dry-runs do not prove real wireless/optical communication gains.
   - Respect channel physics, UEP constraints, and test split isolation.
