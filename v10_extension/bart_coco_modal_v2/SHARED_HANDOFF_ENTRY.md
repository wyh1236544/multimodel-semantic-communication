## 2026-10-12 — additive original COCO caption and single-modality BART V2

User requested original short-text dataset and independent image/text transmission
on current CLIP+BART+VAE architecture, source archive, commands and GitHub save.
New bart_coco_modal_v2 only; old source/results/weights/LN labels/split preserved.
Official captions_train2017.json/captions_val2017.json must exist on offline server;
read-only data-check checks original image IDs, selects minimum annotation ID,
independent hashed caption views. No final test/calibration business access.

Default independently trained image_only C32/I224=2048complex and text_only
C32/T32=512complex; optional joint full2304. Opposite source never enters encoder;
text-only never opens image pixels; image-only legacy B4 receives constant dummy
text, not actual caption. Same registered V1 bridge, different used parameters;
frozen CLIP/BART/VAE/exact B4. RX public mode/pilot/compact features only, no GT,
source mask/length/UTF8 bypass. Decoded text-to-image path remains deferred.

Clean training ONLY then real compact full V3 optical waveforms, not new noise
training or sliced2304 proxy bank. 64GSa/s/4SPS/16GBaud/RRC.1/R1, probe/preamble/
pilot/tails/guards all charged; 2048/512/2304 payload uses12026/5114/13178samples.
Same peak/slot caps, not equal actual energy or equal-resource causal comparison.
No invented analog BER/CRC/bitrate; untransmitted task metrics null.

Text short T32 TRAIN diagnostic8/1200forward/200warmup, then own C+T diagnostic
same budget; private failure does not forbid C+T. Actual mode gate WER<=.20 and
zero/shuffle margins>=.10; reset initial bridge for256x4 clean main if passes.
Validation16 quality gate WER<=.60 +same margins, no checkpoint selection.
Image clean main independent of negative text quality gate; engineering failures
stop and preserve FAILED. Same-grad/no-grad actual forward tolerance.0002 audited
before each stage, no silently changing kernels or suppressing errors.

Each evaluated mode112rows(clean+6FSO)x16; fixed first4 image/text/constellation
examples, all erased frames counted. Actual C-zero/C-shuffle receive sensitivity
controls not independently trained best-private baselines. COMPLETE only all
requested modes evaluated, PARTIAL_COMPLETE negative text quality stop, SMOKE
only execution. Report recomputes text metrics and verifies matrix/resources,
training groups/frozen updates, hashes; weights stay server, records archive small.

Source/install/restore checks +21 new nonTorch tests +49 inherited NumPy tests
passed locally; new11 Torch/HF contracts and real pretrained GPU gates SERVER
only, not locally executed. PhysicalGPU1 free>=16GiB/reserve4GiB, existing env
unchanged. Plan20-60min UNMEASURED, finite2h cap not ETA. No server job launched.
Default launch.py pipeline experiment_results/v10_bart_coco_modal_v2_01
--max-hours2 (space arguments per README). Send v10_bart_coco_modal_v2_records.tar.gz
even PARTIAL; missing originalCOCO JSON/engineering failure send exact log first.
HANDOFF exact-byte backup before append. See ../bart_coco_modal_v2/README.md.
