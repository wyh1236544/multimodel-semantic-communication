"""Before-rejection AND delivered constellation panels, offline diagnostics."""
import html
from pathlib import Path


def constellation_svg(records, destination):
    panels=[]
    for record in records:
        for field,label in (("rx","Equalized BEFORE gate"),("delivered_rx","Delivered AFTER gate")):
            if field not in record: continue
            points=record[field]
            limit=max(1.5,min(10.,max((abs(v) for p in points for v in p),default=1.5)))
            shapes=[]
            for x,y in points:
                shapes.append('<circle cx="%.2f" cy="%.2f" r="1.2" fill="#2458b6" opacity=".35"/>' %
                    (180+min(max(x/limit,-1),1)*145,190-min(max(y/limit,-1),1)*145))
            for x,y in set(tuple(p) for p in record["tx"]):
                shapes.append('<circle cx="%.2f" cy="%.2f" r="3" fill="none" stroke="#d33935"/>' %
                    (180+min(max(x/limit,-1),1)*145,190-min(max(y/limit,-1),1)*145))
            title=html.escape(record["purpose"]+" / "+record["policy"]+" / "+label)
            status=html.escape("lock="+str(record.get("receiver_locked"))+" "+str(record.get("rejected_reasons",[])))
            panels.append('<g transform="translate(%d,0)"><text x="12" y="24">%s</text><rect x="30" y="40" width="300" height="300" fill="none" stroke="#999"/><path d="M30 190H330 M180 40V340" stroke="#bbb"/>%s<text x="15" y="365">%s; axis +/- %.2f (clipped for display)</text></g>' %
                (len(panels)*440,title,"".join(shapes),status,limit))
    if not panels: raise ValueError("No constellation records")
    document='<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="390" font-family="sans-serif" font-size="10">%s</svg>' % (440*len(panels),"".join(panels))
    with Path(destination).open("x",encoding="utf-8") as stream: stream.write(document)

