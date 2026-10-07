"""Bounded B2B signal-chain diagnostics, three alphabets and isolated impairments."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from v10.paths import write_path
from optical_realism_v3.hardware import load, PRESETS
from optical_realism_v3.spec import load_plan, resources
from optical_realism_v3.transport import RealisticTransport
from optical_link_v1.config import load as load_optical
from optical_realism_v3.constellation import constellation_svg
from physical_link.digital import modulate, demodulate


def run(output, hardware_path=None, optical_path=None):
    directory=write_path(output,EXT.parent)
    if directory.exists(): raise FileExistsError('Preserve old B2B output')
    directory.mkdir(parents=True); rows=[]; started=time.monotonic()
    for preset in PRESETS:
        plan,_=load_plan(hardware_path=hardware_path,preset=preset)
        _,cases,_=load_optical(optical_path) if optical_path else load_optical(); c=cases['good']
        res=resources(c,plan)
        for order in (4,16,0):
            if time.monotonic()-started>600: raise TimeoutError('10-minute diagnostic cap')
            rng=np.random.default_rng(6)
            bits=rng.integers(0,2,12000*int(np.log2(order)),dtype=np.uint8) if order else None
            source=modulate(bits,order) if order else (rng.normal(size=12000)+1j*rng.normal(size=12000))/2**.5
            t=RealisticTransport(c,1,2,-9,res['slot_samples'],20,plan['hardware'])
            # Explicit optical back-to-back bypass: not a claimed 1-km FSO result.
            # Fixed 30-dB B2B attenuator prevents overdriving EDFA/ADC; not a fitted fading draw.
            t.path_draw.update(field_real=10**(-30/20),field_imag=0.,aperture_irradiance_gain=.001,coupled_irradiance_gain=.001)
            t.distance=0.
            y,event=t.burst(source,'features',policy='analog' if not order else 'qpsk_conv' if order==4 else '16qam_conv')
            hard=(demodulate(y,order,event['equalized_noise_variance'])<0).astype(np.uint8) if order else None
            row=dict(preset=preset,alphabet='continuous_gaussian' if not order else str(order)+'QAM',
                b2b_field_bypass=True,b2b_attenuator_db=30.,synthetic_payload_not_neural=True,hardware=plan['hardware'],
                uncoded_ber=float(np.mean(hard!=bits)) if order else None,**event)
            rows.append(row)
            print('{} {} lock={} EVM={:.4f} BER={}'.format(preset,row['alphabet'],event['receiver_locked'],event['evm_squared']**.5,row['uncoded_ber']),flush=True)
            constellation_svg(t.constellations,directory/(preset+'_'+row['alphabet']+'.svg'))
            (directory/'b2b.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
    # Loose declared functional gates, not performance/hardware claims.
    passed=all(r['receiver_locked'] and r['evm_squared']<.10 and
               (r['uncoded_ber'] is None or r['uncoded_ber']<.02) for r in rows)
    if not passed: raise RuntimeError('B2B functional gate failed; preserve output, DO NOT launch neural evaluation')
    from optical_realism_v3.report import digest
    (directory/'COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),wall_seconds=time.monotonic()-started,
        b2b_functional_gate=True,not_hardware_calibrated=True,b2b_sha256=digest(directory/'b2b.json')),indent=2))
    return directory


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('output')
    p.add_argument('--hardware-config'); p.add_argument('--optical-config')
    a=p.parse_args(); run(a.output,a.hardware_config,a.optical_config)
