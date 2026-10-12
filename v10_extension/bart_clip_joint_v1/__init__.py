"""Additive, offline CLIP/BART C/I/T candidate, not a verified improvement."""
VERSION = 'V10-bart-clip-joint-v1'
CANDIDATES = ('J0_no_clip_input', 'J1_joint', 'J2_separate')
ARMS = tuple(c + '_' + p for c in CANDIDATES for p in ('clean_continue', 'fso'))
