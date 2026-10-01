"""Dependency-free SVG constellation artifacts: all saved conditions, not winners."""
import html
from pathlib import Path


def constellation_svg(records, destination):
    panels = []
    for record in records:
        if record['purpose'] not in ('text', 'features'):
            continue
        points = record['rx']
        limit = max(1.5, min(10., max((abs(v) for p in points for v in p), default=1.5)))
        shapes = []
        for x, y in points:
            shapes.append('<circle cx="%.2f" cy="%.2f" r="1.3" fill="#2458b6" opacity=".35"/>' %
                          (180+min(max(x/limit, -1), 1)*145, 190-min(max(y/limit, -1), 1)*145))
        for x, y in set(tuple(p) for p in record['tx']):
            if record['policy'] == 'analog':
                break
            shapes.append('<circle cx="%.2f" cy="%.2f" r="4" fill="none" stroke="#d33935"/>' %
                          (180+x/limit*145, 190-y/limit*145))
        title = html.escape(record['purpose']+' / '+record['policy'])
        panels.append('<g transform="translate(%d,0)"><text x="15" y="24">%s</text>'
            '<rect x="30" y="40" width="300" height="300" fill="none" stroke="#999"/>'
            '<path d="M30 190H330 M180 40V340" stroke="#bbb"/>'
            '%s<text x="35" y="365">Blue: RX; red: TX alphabet; axis +/- %.2f</text></g>' %
            (len(panels)*380, title, ''.join(shapes), limit))
    if panels:
        data = '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="390" font-family="sans-serif" font-size="12">%s</svg>' % (380*len(panels), ''.join(panels))
        with Path(destination).open('x', encoding='utf-8') as f:
            f.write(data)
