"""Reuse historical optical DSP, but transmit/charge the ACTUAL new frame size."""
from docci_modal_v9.channel import clean,noisy,physical_config
from docci_reconstruction_v10.core import layout,geometry


def resource(c,arm,physical=None):
    from docci_modal_v9.channel import resource as old
    branch='text' if arm.startswith('T') else 'image'
    res=old(layout(c,arm),branch,physical); res.update(geometry(c,arm)); return res


def send(c,arm,physical,condition,path,seed,wire):
    from docci_modal_v9.channel import send as old
    branch='text' if arm.startswith('T') else 'image'
    y,stats,transport=old(layout(c,arm),branch,physical,condition,path,seed,wire)
    stats['resources']=resource(c,arm,physical)
    return y,stats,transport
