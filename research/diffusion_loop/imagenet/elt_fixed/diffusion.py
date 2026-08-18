from __future__ import annotations

import math

import torch


def shifted_cosine_logsnr(
    timestep: torch.Tensor,
    *,
    image_resolution: int = 256,
    noise_resolution: int = 64,
    logsnr_min: float = -15.0,
    logsnr_max: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shifted cosine log-SNR and its analytic derivative with respect to t."""
    if image_resolution <= 0 or noise_resolution <= 0:
        raise ValueError("image and noise resolutions must be positive")
    lower_angle = math.atan(math.exp(-0.5 * logsnr_max))
    upper_angle = math.atan(math.exp(-0.5 * logsnr_min))
    angle_range = upper_angle - lower_angle
    angle = lower_angle + angle_range * timestep
    shift = 2.0 * math.log(noise_resolution / image_resolution)
    logsnr = -2.0 * torch.log(torch.tan(angle)) + shift
    derivative = -2.0 * angle_range / (torch.sin(angle) * torch.cos(angle))
    return logsnr, derivative


def alpha_sigma(
    timestep: torch.Tensor,
    *,
    image_resolution: int = 256,
    noise_resolution: int = 64,
    logsnr_min: float = -15.0,
    logsnr_max: float = 15.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    logsnr, _ = shifted_cosine_logsnr(
        timestep,
        image_resolution=image_resolution,
        noise_resolution=noise_resolution,
        logsnr_min=logsnr_min,
        logsnr_max=logsnr_max,
    )
    return torch.sigmoid(logsnr).sqrt(), torch.sigmoid(-logsnr).sqrt()


def clean_noise_from_v(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a v-prediction into clean-data and noise predictions."""
    while alpha.ndim < noisy.ndim:
        alpha = alpha.unsqueeze(-1)
        sigma = sigma.unsqueeze(-1)
    prediction = prediction.float()
    predicted_clean = alpha * noisy - sigma * prediction
    predicted_noise = sigma * noisy + alpha * prediction
    return predicted_clean, predicted_noise


def ddpm_posterior_step(
    noisy: torch.Tensor,
    prediction: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
    alpha_s: torch.Tensor,
    sigma_s: torch.Tensor,
    *,
    noise: torch.Tensor | None,
) -> torch.Tensor:
    """Ancestral DDPM transition from noisy time t to the cleaner time s."""
    predicted_clean, _ = clean_noise_from_v(noisy, prediction, alpha_t, sigma_t)
    alpha_bar_t = alpha_t.square()
    alpha_bar_s = alpha_s.square()
    incremental_alpha = alpha_bar_t / alpha_bar_s
    incremental_beta = 1.0 - incremental_alpha

    denominator = 1.0 - alpha_bar_t
    clean_coefficient = alpha_s * incremental_beta / denominator
    noisy_coefficient = incremental_alpha.sqrt() * (1.0 - alpha_bar_s) / denominator
    posterior_variance = (1.0 - alpha_bar_s) * incremental_beta / denominator

    while clean_coefficient.ndim < noisy.ndim:
        clean_coefficient = clean_coefficient.unsqueeze(-1)
        noisy_coefficient = noisy_coefficient.unsqueeze(-1)
        posterior_variance = posterior_variance.unsqueeze(-1)
    mean = clean_coefficient * predicted_clean + noisy_coefficient * noisy
    if noise is None:
        return mean
    return mean + posterior_variance.clamp_min(0.0).sqrt() * noise


__all__ = [
    "alpha_sigma",
    "clean_noise_from_v",
    "ddpm_posterior_step",
    "shifted_cosine_logsnr",
]
