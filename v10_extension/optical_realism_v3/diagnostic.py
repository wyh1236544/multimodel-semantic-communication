"""Same-observation V2/V3 replay on synthetic payloads; NOT neural/GPU evidence."""
import argparse
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
EXT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(EXT))
from v10.paths import write_path
from v10.io import atomic_json
from business_link.engine import identity_seed
from optical_link_v1.config import load as optical
from optical_realism_v3.spec import load_plan,resources
from optical_realism_v3.transport import RealisticTransport
from optical_realism_v2.dsp import receive as old_receive
from optical_realism_v2.hardware import load as old_hardware
from optical_realism_v3.report import digest


def run(output,draws=4,hardware_path=None,optical_path=None):
    if type(draws) is not int or not 1<=draws<=4: raise ValueError("1..4 bounded channel observations")
    d=write_path(output,EXT.parent)
    if d.exists(): raise FileExistsError("Preserve previous receiver diagnostics")
    d.mkdir(parents=True); rows=[]; started=time.monotonic()
    for preset in ("ideal","combined"):
        plan,_=load_plan(hardware_path=hardware_path,preset=preset)
        _,cases,_=optical(optical_path) if optical_path else optical()
        old_h=old_hardware(preset=preset)
        # Retain V2 receiver thresholds, use the same publicly declared pilot layout.
        old_h.update(pilot_interval=plan["hardware"]["pilot_interval"],pilot_block=plan["hardware"]["pilot_block"])
        for case,target in (("good",-9.),("moderate",-24.),("stress",-9.)):
            c=cases[case]; c["waveform"]["policy"]="qpsk_conv"; resource=resources(c,plan)
            for draw in range(draws):
                if time.monotonic()-started>600: raise TimeoutError("10-minute receiver diagnostic cap")
                rng=np.random.default_rng(1729+draw)
                x=rng.normal(size=(288,16)).astype(np.float32)
                t=RealisticTransport(c,identity_seed(1729,draw,case,"field"),
                    identity_seed(1729,draw,case,"noise"),target,resource["slot_samples"],20,plan["hardware"])
                t.capture_diagnostic_observation=True
                t.probe()
                restored,rel,stats=t.continuous(x,resource["analog_repeats"],"lmmse")
                source=(x.ravel()[::2]+1j*x.ravel()[1::2])/math.sqrt(2)
                before,_,diagnostic,*_=old_receive(t.last_diagnostic_observation,t.pulse,
                    len(source)*resource["analog_repeats"],old_h)
                oldrel=1/(1+diagnostic["equalized_noise_variance"]/resource["analog_repeats"]) if diagnostic["receiver_locked"] else 0.
                combined=before.reshape(resource["analog_repeats"],-1).mean(0)*oldrel
                old_restored=np.stack([combined.real,combined.imag],1).ravel()*math.sqrt(2)
                oldmse=float(np.mean((old_restored-x.ravel())**2))
                event=t.events[-1]
                row=dict(preset=preset,case=case,reference_launch_dbm=target,draw=draw,
                    same_adc_observation=True,synthetic_payload_not_A=True,
                    v2_locked=diagnostic["receiver_locked"],v3_locked=event["receiver_locked"],
                    v2_mse=oldmse,v3_mse=stats["analog_feature_mse"],
                    v2_cfo_hz=diagnostic["estimated_cfo_hz"],v3_cfo_hz=event["estimated_cfo_hz"],
                    expected_cfo_hz=-event["simulator_lo_offset_hz"],
                    v2_reliability=oldrel,v3_reliability=rel,
                    v3_effective_variance=stats["analog_effective_combined_variance"],
                    v3_false_lock_mse_over_one=bool(event["receiver_locked"] and stats["analog_feature_mse"]>1),
                    waveform_noise_only_snr_db=event["waveform_noise_only_snr_db"],
                    v2_rejection_reasons=diagnostic["receiver_rejection_reasons"],
                    v3_rejection_reasons=event["receiver_rejection_reasons"])
                rows.append(row)
                print("{} {} draw={} old/new lock={}/{} MSE={:.4f}/{:.4f}".format(
                    preset,case,draw,row["v2_locked"],row["v3_locked"],oldmse,row["v3_mse"]),flush=True)
                atomic_json(d/"diagnostic.json",rows)
    groups=[]
    for preset in ("ideal","combined"):
        for case in ("good","moderate","stress"):
            sub=[r for r in rows if r["preset"]==preset and r["case"]==case]
            groups.append(dict(preset=preset,case=case,observations=len(sub),
                v2_locked=sum(r["v2_locked"] for r in sub),v3_locked=sum(r["v3_locked"] for r in sub),
                mean_v2_mse=float(np.mean([r["v2_mse"] for r in sub])),
                mean_v3_mse=float(np.mean([r["v3_mse"] for r in sub])),
                v3_locked_mse_over_one=sum(r["v3_false_lock_mse_over_one"] for r in sub)))
    atomic_json(d/"summary.json",dict(groups=groups,
        synthetic_only=True,not_hardware_calibrated=True,not_neural_performance=True))
    atomic_json(d/"COMPLETE.json",dict(rows=len(rows),wall_seconds=time.monotonic()-started,
        version="V10-optical-realism-v3-receiver-diagnostic",synthetic_only=True,
        diagnostic_sha256=digest(d/"diagnostic.json"),summary_sha256=digest(d/"summary.json")))
    return groups


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output"); p.add_argument("--draws",type=int,default=4)
    p.add_argument("--hardware-config"); p.add_argument("--optical-config")
    a=p.parse_args()
    print(json.dumps(run(a.output,a.draws,a.hardware_config,a.optical_config),indent=2))
