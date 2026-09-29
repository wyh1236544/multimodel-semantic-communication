"""Deterministic cross-image Common donors for single-image evaluation."""


def cyclic_donor_map(image_ids):
    ids = list(image_ids)
    if len(ids) < 2 or len(ids) != len(set(ids)):
        raise ValueError('Shuffled Common requires at least two distinct image IDs')
    return dict(zip(ids, ids[1:] + ids[:1]))
