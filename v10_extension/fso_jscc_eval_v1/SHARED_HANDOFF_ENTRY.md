## 2026-10-11 frozen-weight expanded FSO JSCC evaluation V1

Additive fso_jscc_eval_v1 evaluates the EXACT four completed fso_jscc_v1 screen
weights. NO optimizer/retraining, no environment upgrade, physical GPU1 only.
128 fixed validation-prefix images: pilot32 vs additional96 reported separately.
Additional means unused in that 32-image screen, not guaranteed unseen in all
historical experiments. No final test/calibration split is used. One training
replicate; channel draws are independent trials, not combined transmissions.

R=1, unchanged token/symbol budget, AWG64GSa/s, 4SPS, RRC.1 and V3 receiver.
Proxy25,088 plus waveform6,144 =31,232 rows. Six physical points, two independent
draws. Predeclared primary: M3-M2 PSNR on new96, per-image average over
moderate_p24/p21/p18 and draws, image-cluster bootstrap. Failures retained;
pilot replay, strict delta identity, before/after file and full model state hashes
checked. Optical power, offline projection SNR, pilot SNR, energy, text metrics,
images/constellations included. No analog BER or oracle text side channel.

Public slot/samples/peak-envelope limits match, not exact actual launch energy.
This is validation confirmation of FSO-vs-AWGN fine tuning, NOT proof of joint-vs-
separate, analog-vs-digital superiority, independent semantic fidelity or hardware.
Local NumPy/source/archive tests passed; Torch/real checkpoints/GPU unexecuted
locally (Torch unavailable). User-started launcher gates CPU tests and two-image
frozen-weight smoke before expanded evaluation; displays measured time projection.
Default 6h safety cap is NOT ETA. No server task launched by this code delivery.
Return v10_fso_jscc_eval_v1_records.tar.gz. See fso_jscc_eval_v1/README.md.
