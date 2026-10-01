"""Exact digital text metrics. Failed transfer is scored as empty, not omitted."""
def edit_distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def text_metrics(source, received):
    return {'text_cer': edit_distance(source, received)/max(len(source), 1),
            'text_wer': edit_distance(source.split(), received.split())/max(len(source.split()), 1)}
