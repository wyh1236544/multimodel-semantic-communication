"""Same source-based text metrics for every method; exact and normalized separate."""
import re
import unicodedata
from business_link.metrics import text_metrics


def distance(a, b):
    old = list(range(len(b)+1))
    for i, item in enumerate(a, 1):
        new = [i]
        for j, other in enumerate(b, 1):
            new.append(min(new[-1]+1, old[j]+1, old[j-1]+int(item != other)))
        old = new
    return old[-1]


def words(text):
    # Lowercase/Unicode compatibility normalize, remove punctuation, NOT semantics.
    return re.findall(r'\w+', unicodedata.normalize('NFKC', text).casefold(), re.UNICODE)


def caption_metrics(source, received):
    a, b = words(source), words(received)
    errors = distance(a, b)
    raw = text_metrics(source, received)
    return dict(raw, wer=raw['text_wer'], cer=raw['text_cer'], text_exact=source == received,
                normalized_word_errors=errors, normalized_source_words=len(a),
                normalized_wer=errors/max(len(a), 1),
                normalization='NFKC+casefold+Unicode_word_tokens; raw WER/CER also reported')


def task_metrics(row, plan):
    image_ok = row['psnr'] >= plan['image_psnr_target_db']
    text_ok = row['normalized_wer'] <= plan['normalized_wer_target']
    return dict(image_task_success=bool(image_ok), text_task_success=bool(text_ok),
                joint_task_success=bool(image_ok and text_ok),
                task_thresholds_are_public_proxies_not_universal_quality_limits=True)
