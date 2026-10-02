"""Common time/launch-power ceilings; existing sampled SI optics unchanged.

A transmitter-only launch-power regulator changes laser power using the known
MZM waveform and TX EDFA model. It never reads received data, fading, SNR or
payload correctness. Receiver pilot equalization needs no hidden source scale.
"""
import copy
import math
import numpy as np
from optical_link_v1.transport import OpticalTransport
from optical_link_v1.physics import modulate_field, power, amplifier, dbm_to_w, w_to_dbm
from physical_link.waveform import preamble, dac
from final_compare_v1.spec import sampled_count


def regulated_config(converted, config, target_dbm, laser_cap_dbm):
    c = copy.deepcopy(config)
    c["laser"]["power_dbm"] = 0.
    mzm_at_one_mw = power(modulate_field(converted, c, c["waveform"]["fs_hz"]))
    if mzm_at_one_mw <= 0:
        raise ValueError("Cannot regulate a zero modulator output")
    ed = c["tx_edfa"]
    requested = 10**(ed["gain_db"]/10) if ed["enabled"] else 1.
    launch_loss = 10**(-c["path"]["tx_optics_loss_db"]/10)
    wanted = dbm_to_w(target_dbm)
    laser_dbm = w_to_dbm(wanted / (mzm_at_one_mw/.001 * requested * launch_loss))
    if laser_dbm is None or not -100 <= laser_dbm <= laser_cap_dbm:
        raise ValueError("Launch target infeasible under laser cap; change public plan, not silently clamp")
    c["laser"]["power_dbm"] = laser_dbm
    after = mzm_at_one_mw * 10**(laser_dbm/10)
    stage = amplifier(after, 0., ed, c["waveform"]["fs_hz"], c["laser"]["wavelength_m"])
    achieved = stage["signal_output_w"]*launch_loss
    if abs(achieved/wanted-1) > 1e-9:
        raise ValueError("TX EDFA limiting prevents common launch target")
    return c, laser_dbm


class BudgetTransport(OpticalTransport):
    def __init__(self, config, field_seed, noise_seed, target_dbm, slot_samples, laser_cap_dbm):
        super().__init__(copy.deepcopy(config), field_seed, noise_seed)
        self.public_config = copy.deepcopy(config)
        self.target_dbm = float(target_dbm)
        self.slot_samples = int(slot_samples)
        self.laser_cap_dbm = float(laser_cap_dbm)

    def burst(self, symbols, purpose, sequence=0, attempt=0, policy="probe"):
        samples = sampled_count(len(symbols), self.public_config["waveform"])
        used = sum(e["awg_samples"] + e["guard_samples"] for e in self.events)
        if used + samples > self.slot_samples:
            raise ValueError("Common resource budget exceeded BEFORE transmission")
        frame = np.concatenate([preamble(256), np.asarray(symbols, complex)])
        wave = self.pulse.shape(frame)*math.sqrt(self.pulse.sps)
        converted, _, _ = dac(wave, self.pulse.sps, self.settings["dac_bits"])
        config, laser = regulated_config(converted, self.public_config, self.target_dbm, self.laser_cap_dbm)
        previous = self.optical_config
        try:
            self.optical_config = config
            data, event = super().burst(symbols, purpose, sequence, attempt, policy)
        finally:
            self.optical_config = previous
        actual = event["power_stages"]["launch"]["signal_w"]
        target = dbm_to_w(self.target_dbm)
        if abs(actual/target-1) > 1e-8:
            raise ValueError("Actual launch signal power differs from common target")
        event.update(launch_target_signal_dbm=self.target_dbm, regulated_laser_dbm=laser,
            launch_power_relative_error=abs(actual/target-1),
            launch_signal_energy_j=actual*event["waveform_seconds"])
        return data, event

    def totals(self):
        result = super().totals()
        used = result["awg_samples"]+result["guard_samples"]
        actual = result["launch_signal_energy_j"]
        cap = dbm_to_w(self.target_dbm)*self.slot_samples/self.fs
        if used > self.slot_samples or actual > cap*(1+1e-8):
            raise ValueError("Post-transmission time/energy ceiling violation")
        result.update(slot_samples=self.slot_samples, slot_seconds=self.slot_samples/self.fs,
            used_samples=used, unused_slot_samples=self.slot_samples-used,
            launch_target_signal_dbm=self.target_dbm, signal_energy_budget_j=cap,
            launch_signal_energy_j=actual, energy_budget_utilization=actual/cap,
            normalized_sample_time_utilization=used/self.slot_samples,
            same_resource_contract="same bandwidth and average active launch SIGNAL power; common time/energy ceilings, not identical spent energy",
            resource_feasible=True, ideal_ack_arq=False, ack_simulated=False,
            bytes_for_analog_are_undefined=True)
        policy=self.public_config['waveform']['policy']
        result.update(awg_sample_rate=self.fs,sps=self.sps,symbol_rate_baud=self.pulse.rs,
            rolloff=self.pulse.alpha,occupied_complex_bandwidth_hz=(1+self.pulse.alpha)*self.pulse.rs,
            digital_gross_modulation_bps=self.pulse.rs*(2 if policy=='qpsk_conv' else 4),
            configured_digital_policy=policy,wire_fec='none' if policy.endswith('_none') else 'K7 terminated convolutional, 1024-bit padded blocks',
            configured_digital_bit_repetitions=2 if policy=='16qam_repeat2' else 1,
            bit_rate_for_continuous_features_undefined=True)
        return result

    def continuous(self, values, repeats, equalizer):
        x = np.asarray(values, np.float32)
        if x.shape != (288, 16) or not np.isfinite(x).all() or repeats < 1:
            raise ValueError("Expected C32/I224/T32 features and positive public repetition")
        pair = x.ravel().reshape(-1, 2)
        source = (pair[:, 0]+1j*pair[:, 1])/math.sqrt(2)
        symbols = np.tile(source, repeats)
        received, event = self.burst(symbols, "features", policy="analog")
        combined = received.reshape(repeats, len(source)).mean(0)
        variance = event["equalized_noise_variance"]/repeats
        reliability = float(1/(1+variance))
        if equalizer == "lmmse":
            combined = combined*reliability
        restored = np.stack([combined.real, combined.imag], 1).ravel()*math.sqrt(2)
        if not np.isfinite(restored).all():
            raise FloatingPointError("Nonfinite continuous reconstruction")
        stats = dict(self.totals(), feature_object_crc_accepted=None,
            raw_utf8_wire_bytes=0, analog_repeats=int(repeats), analog_source_complex_symbols=len(source),
            analog_source_reals=x.size, source_quantization_mse=0., source_clip_fraction=0.,
            semantic_body_bytes=None, feature_object_wire_bytes=None,
            analog_combined_estimated_snr_db=(-10*math.log10(variance) if variance > 0 else None),
            analog_reliability=reliability, analog_equalizer=equalizer,
            analog_feature_mse=float(np.mean((restored.reshape(x.shape)-x)**2)),
            analog_repeat_note="time/energy charged; same constant field, no free fading diversity; variance/R is an approximate noise model",
            pre_fec_ber=None, post_fec_ber=None, packet_error_rate=None,
            crc_note="continuous semantic features have no binary CRC/BER; availability is NOT certified delivery")
        return restored.astype(np.float32).reshape(x.shape), reliability, stats
