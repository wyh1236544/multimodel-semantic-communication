"""Two-image diagnosis only. Frozen V23 inputs; no training or expansion."""
import hashlib
import json
import math
from pathlib import Path
import re
import time

FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parents[1]
SCHEMA = 'docci-caption-calibration-v24'
QUESTIONS = {
    'object': 'Question: What is the main object in this image? Answer:',
    'color': 'Question: What color is the main object in this image? Answer:',
    'count': 'Question: How many main objects are visible in this image? Answer:',
    'relation': 'Question: Where is the main object relative to its surroundings? Answer:',
}
PROTOCOL = dict(schema=SCHEMA, images=2, repeats=1, snr_db=[None, 0, -5],
    caption_max_new_tokens=64, qa_max_new_tokens=32, seed=42,
    source_role='privileged diagnosis, NOT receiver input',
    caption_prompt=None, qa_questions=QUESTIONS, do_sample=False, num_beams=1,
    no_repeat_ngram_size=0, repetition_penalty=1.0, min_new_tokens=0,
    training_updates=0, semantic_success=False, facts='PENDING',
    image_frame_symbols=4128, text_frame_symbols=9248, assisted_frame_symbols=13376,
    new_transmissions=0, channel='inherited AWGN; clean is no added noise, not original image',
    negative_snr='OOD frozen inference', automatic_next_stage=False)

def read(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.load(f)

def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()

def local(root, name):
    p = Path(name)
    if not p.parts or p.is_absolute() or '..' in p.parts:
        raise ValueError('Unsafe relative file')
    result = Path(root)/p
    if any(x.is_symlink() for x in [result]+list(result.parents)):
        raise ValueError('Symlink paths forbidden')
    if Path(root).resolve() not in result.resolve().parents:
        raise ValueError('Escaped file')
    return result

def safe(path):
    p = Path(path).absolute()
    if PROJECT.resolve() not in p.resolve().parents or '..' in p.parts:
        raise ValueError('New output must be inside current multimodel project')
    if any(x.is_symlink() for x in [p]+list(p.parents)):
        raise ValueError('Symlink output forbidden')
    return p

def atomic(path, value):
    p = Path(path)
    if p.exists():
        raise FileExistsError('Existing output preserved: '+str(p))
    tmp = p.with_name(p.name+'.writing')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    tmp.rename(p)

def line(f, value):
    f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')
    f.flush()

def source_check():
    m = read(FOLDER/'MANIFEST.json')
    for base, files in ((FOLDER, m['files']), (PROJECT, m['dependencies'])):
        for n, h in files.items():
            if sha(local(base, n)) != h:
                raise ValueError('Pinned source differs: '+n)
    return m

def verify_assets(lock, budget):
    """One streaming full read, checked against BOTH seal and official blob/LFS IDs."""
    from docci_evidence_v23.download_caption import FILES, REPO, REVISION
    if lock.get('kind') != 'caption' or lock.get('origin') != REPO or lock.get('revision') != REVISION:
        raise ValueError('Wrong pinned official caption assets')
    folder = Path(lock['tree']['path']).resolve(strict=True)
    expected = set(FILES) | {'DOWNLOAD_RECEIPT.json'}
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if set(lock['tree']['files']) != expected or actual != expected:
        raise ValueError('Unexpected/missing model files')
    last = time.monotonic()
    for name in sorted(expected):
        budget(); path = local(folder, name); size = path.stat().st_size
        spec = FILES.get(name)
        if spec and size != spec[0]: raise ValueError('Official model size differs '+name)
        official = hashlib.sha1(('blob '+str(size)+'\0').encode()) if spec and spec[1] == 'git' else hashlib.sha256()
        checksum = hashlib.sha256(); count = 0
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                checksum.update(chunk); official.update(chunk); count += len(chunk)
                if time.monotonic()-last >= 10:
                    print('V24 HASH {} {:.1f}%'.format(name,100*count/max(1,size)),flush=True)
                    budget(); last = time.monotonic()
        if checksum.hexdigest() != lock['tree']['files'][name] or (spec and official.hexdigest() != spec[2]):
            raise ValueError('Sealed/official model bytes differ '+name)
    receipt = read(folder/'DOWNLOAD_RECEIPT.json')
    if receipt.get('repo') != REPO or receipt.get('revision') != REVISION or receipt.get('state') != 'COMPLETE':
        raise ValueError('Official download receipt differs')
    if set(receipt['files']) != set(FILES) or any(receipt['files'][n] != lock['tree']['files'][n] for n in FILES):
        raise ValueError('Receipt/seal file hashes differ')
    return folder

def key(eid, snr, control, task):
    return digest([eid, snr, control, task])

def make_plan(index, data, stage):
    """Source text, lengths and received text never enter these model requests."""
    if stage not in ('caption', 'qa') or index['profile'] != 'smoke':
        raise ValueError('Only existing two-image smoke is authorized here')
    examples = index['examples']
    ids = [r['example_id'] for r in examples]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError('Expected exactly two distinct development examples')
    by = {(r['example_id'], r['snr_db'], r['repeat']): r for r in data}
    if len(data) != 6 or set(by) != {(e, s, 0) for e in ids for s in (None, 0, -5)}:
        raise ValueError('Wrong frozen two-image matrix')
    tasks = {'caption': None} if stage == 'caption' else QUESTIONS
    plan = []
    for ex in examples:
        eid = ex['example_id']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', eid):
            raise ValueError('Unsafe example ID')
        for task, prompt in tasks.items():
            plan.append(dict(case_id=key(eid, None, 'source_diagnostic', task),
                example_id=eid, snr_db=None, control='source_diagnostic', task=task,
                prompt=prompt, privileged=True, donor_id=None,
                source_file=ex['source_image'], image_sha256=ex['source_image_sha256']))
        for snr in (None, 0, -5):
            own = by[eid, snr, 0]
            if own['donor_id'] not in ids or own['donor_id'] == eid:
                raise ValueError('Mismatch must use another real example')
            other = by[own['donor_id'], snr, 0]
            for control, r in (('received', own), ('wrong_image', other), ('gray_image', None)):
                for task, prompt in tasks.items():
                    plan.append(dict(case_id=key(eid, snr, control, task),
                        example_id=eid, snr_db=snr, control=control, task=task,
                        prompt=prompt, privileged=False,
                        donor_id=own['donor_id'] if control == 'wrong_image' else None,
                        source_file=None if r is None else r['received_image'],
                        image_sha256=None if r is None else r['received_image_sha256']))
    return plan

def generation_kwargs(stage):
    return dict(max_new_tokens=64 if stage == 'caption' else 32, do_sample=False,
        num_beams=1, num_return_sequences=1, return_dict_in_generate=True,
        output_scores=True, min_length=0, min_new_tokens=0,
        repetition_penalty=1.0, no_repeat_ngram_size=0,
        forced_bos_token_id=None, forced_eos_token_id=None,
        bad_words_ids=None, suppress_tokens=None, begin_suppress_tokens=None,
        remove_invalid_values=False, renormalize_logits=False)

def processor_kwargs(image, prompt):
    # Omit text altogether for native image captioning; text='' is not substituted.
    value = dict(images=image, return_tensors='pt')
    if prompt is not None:
        value.update(text=prompt, truncation=False)
    return value

def token_audit(returned, steps, eos_ids, cap):
    if (not isinstance(steps, int) or not 1 <= steps <= cap or
            len(returned) <= steps or any(type(x) is not int or x < 0 for x in returned)):
        raise ValueError('Invalid generation sequence/score alignment')
    prefix, new = returned[:-steps], returned[-steps:]
    eos = [i for i, x in enumerate(new) if x in eos_ids]
    if eos and eos[0] != len(new)-1:
        raise ValueError('Unexpected tokens after EOS in single greedy sequence')
    if not eos and steps != cap:
        raise ValueError('Unexplained early termination')
    return dict(returned_token_ids=returned, returned_prefix_ids=prefix,
        new_token_ids=new, new_token_count=steps, natural_EOS=bool(eos),
        capacity_stop=not bool(eos), first_new_EOS_position=eos[0] if eos else None)

def text_audit(text):
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    pairs = sum(a == b for a, b in zip(words, words[1:]))
    grams = [tuple(words[i:i+8]) for i in range(max(0, len(words)-7))]
    return dict(empty=not bool(text.strip()), words=len(words),
        adjacent_repeat_fraction=pairs/max(1, len(words)-1),
        duplicate_8gram_fraction=0.0 if not grams else 1-len(set(grams))/len(grams))

def review_keys(plan):
    return [r['case_id'] for r in plan if r['control'] == 'source_diagnostic' or
        (r['control'] == 'received' and r['snr_db'] is None)]

def require_review(caption_root, decision, verified):
    review = read(decision)
    expected = set(review_keys(read(Path(caption_root)/'PLAN.json')))
    if (review.get('caption_complete_sha256') != sha(Path(caption_root)/'COMPLETE.json') or
            review.get('decision') != 'APPROVE_QA_ONLY' or
            not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()):
        raise ValueError('Explicit human review of this exact caption run required')
    items = review.get('items', [])
    if len(items) != 4 or {x['case_id'] for x in items} != expected:
        raise ValueError('Review must cover both originals and both clean receivers')
    for r in items:
        if any(r.get(k) is not True for k in ('on_topic', 'main_object_supported', 'no_long_loop')):
            raise ValueError('Basic visual review failed; do not auto-promote')
    if verified['empty_basic_outputs']:
        raise ValueError('Empty basic captions cannot authorize QA')
    return review

class Budget:
    def __init__(self, minutes):
        if not math.isfinite(minutes) or not 0 < minutes <= 120:
            raise ValueError('Explicit budget in (0,120] minutes required')
        self.deadline = time.monotonic()+minutes*60
        self.gpu = False
    def __call__(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Budget exceeded; preserve partial records, no restart')
        if self.gpu:
            import torch
            if torch.cuda.mem_get_info()[0] < 4*2**30:
                raise RuntimeError('Runtime GPU free reserve below 4 GiB; no auto changes')
