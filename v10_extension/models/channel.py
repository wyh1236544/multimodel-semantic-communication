from typing import Optional, Union

import torch
from torch import Tensor, nn


class PowerNormalization(nn.Module):
    """Normalize every sample so its average latent power is one."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, latent: Tensor) -> Tensor:
        dims = tuple(range(1, latent.ndim))
        mean_power = latent.pow(2).mean(dim=dims, keepdim=True)
        return latent / torch.sqrt(mean_power + self.eps)


class AWGNChannel(nn.Module):
    """Add white Gaussian noise using SNR measured in dB."""

    def forward(
        self,
        latent: Tensor,
        snr_db: Union[float, int, Tensor],
        standard_noise: Optional[Tensor] = None,
    ) -> Tensor:
        snr = torch.as_tensor(snr_db, dtype=latent.dtype, device=latent.device)
        if snr.ndim == 1:
            snr = snr.view(-1, *([1] * (latent.ndim - 1)))
        snr_linear = torch.pow(10.0, snr / 10.0)
        noise_std = torch.rsqrt(snr_linear)
        if standard_noise is None:
            standard_noise = torch.randn_like(latent)
        elif standard_noise.shape != latent.shape:
            raise ValueError(
                f"Noise shape {tuple(standard_noise.shape)} does not match "
                f"latent shape {tuple(latent.shape)}"
            )
        standard_noise = standard_noise.to(device=latent.device, dtype=latent.dtype)
        return latent + standard_noise * noise_std


class FadingChannel(nn.Module):
    """Block fading with perfect receiver CSI and zero-forcing equalization."""

    def __init__(
        self,
        fading_type: str,
        gamma_gamma_alpha: float = 4.0,
        gamma_gamma_beta: float = 2.0,
        min_gain: float = 1e-3,
    ):
        super().__init__()
        self.fading_type = fading_type
        self.gamma_gamma_alpha = float(gamma_gamma_alpha)
        self.gamma_gamma_beta = float(gamma_gamma_beta)
        self.min_gain = float(min_gain)

    def sample_gain(self, latent: Tensor) -> Tensor:
        shape = (latent.size(0),) + (1,) * (latent.ndim - 1)
        if self.fading_type == "rayleigh":
            real = torch.randn(shape, device=latent.device, dtype=latent.dtype)
            imag = torch.randn(shape, device=latent.device, dtype=latent.dtype)
            return torch.sqrt((real.square() + imag.square()) / 2.0)
        if self.fading_type == "gamma_gamma":
            sample_dtype = torch.float32
            alpha = torch.full(
                shape, self.gamma_gamma_alpha,
                device=latent.device, dtype=sample_dtype,
            )
            beta = torch.full(
                shape, self.gamma_gamma_beta,
                device=latent.device, dtype=sample_dtype,
            )
            large_scale = torch._standard_gamma(alpha) / self.gamma_gamma_alpha
            small_scale = torch._standard_gamma(beta) / self.gamma_gamma_beta
            return (large_scale * small_scale).to(dtype=latent.dtype)
        raise ValueError(f"Unsupported fading type: {self.fading_type}")

    def forward(
        self,
        latent: Tensor,
        snr_db: Union[float, int, Tensor],
        standard_noise: Optional[Tensor] = None,
        channel_gain: Optional[Tensor] = None,
    ) -> Tensor:
        snr = torch.as_tensor(snr_db, dtype=latent.dtype, device=latent.device)
        if snr.ndim == 1:
            snr = snr.view(-1, *([1] * (latent.ndim - 1)))
        noise_std = torch.rsqrt(torch.pow(10.0, snr / 10.0))
        if standard_noise is None:
            standard_noise = torch.randn_like(latent)
        elif standard_noise.shape != latent.shape:
            raise ValueError(
                f"Noise shape {tuple(standard_noise.shape)} does not match "
                f"latent shape {tuple(latent.shape)}"
            )
        noise = standard_noise.to(device=latent.device, dtype=latent.dtype)
        gain = self.sample_gain(latent) if channel_gain is None else channel_gain
        gain = gain.to(device=latent.device, dtype=latent.dtype)
        expected_shape = (latent.size(0),) + (1,) * (latent.ndim - 1)
        if gain.shape not in (latent.shape, expected_shape):
            raise ValueError(
                f"Channel gain shape {tuple(gain.shape)} must be "
                f"{expected_shape} or {tuple(latent.shape)}"
            )
        received = gain * latent + noise * noise_std
        return received / gain.clamp_min(self.min_gain)


class SemanticChannel(nn.Module):
    """Power normalization followed by AWGN, Rayleigh, or Gamma-Gamma fading."""

    def __init__(
        self,
        channel_type: str = "awgn",
        gamma_gamma_alpha: float = 4.0,
        gamma_gamma_beta: float = 2.0,
    ):
        super().__init__()
        self.power_normalization = PowerNormalization()
        self.configure(
            channel_type,
            gamma_gamma_alpha=gamma_gamma_alpha,
            gamma_gamma_beta=gamma_gamma_beta,
        )

    def configure(
        self,
        channel_type: str,
        gamma_gamma_alpha: float = 4.0,
        gamma_gamma_beta: float = 2.0,
    ) -> None:
        channel_type = channel_type.lower().replace("fso", "gamma_gamma")
        if channel_type == "awgn":
            self.channel = AWGNChannel()
        elif channel_type in {"rayleigh", "gamma_gamma"}:
            self.channel = FadingChannel(
                channel_type,
                gamma_gamma_alpha=gamma_gamma_alpha,
                gamma_gamma_beta=gamma_gamma_beta,
            )
        else:
            raise ValueError(
                "channel_type must be awgn, rayleigh, gamma_gamma, or fso"
            )
        self.channel_type = channel_type

    def forward(
        self,
        latent: Tensor,
        snr_db: Union[float, int, Tensor],
        standard_noise: Optional[Tensor] = None,
        channel_gain: Optional[Tensor] = None,
    ) -> Tensor:
        normalized = self.power_normalization(latent)
        if self.channel_type == "awgn":
            return self.channel(normalized, snr_db, standard_noise=standard_noise)
        return self.channel(
            normalized,
            snr_db,
            standard_noise=standard_noise,
            channel_gain=channel_gain,
        )
