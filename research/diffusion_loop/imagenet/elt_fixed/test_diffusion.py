from __future__ import annotations

import copy

import torch

from .diffusion import alpha_sigma, clean_noise_from_v, ddpm_posterior_step
from .model import FixedLoopDiT


def test_shifted_cosine_is_variance_preserving_and_monotonic() -> None:
    timestep = torch.linspace(0.0, 1.0, 101)
    alpha, sigma = alpha_sigma(timestep)
    torch.testing.assert_close(alpha.square() + sigma.square(), torch.ones_like(alpha))
    assert torch.all(alpha[1:] < alpha[:-1])
    assert torch.all(sigma[1:] > sigma[:-1])


def test_v_conversion_recovers_clean_and_noise() -> None:
    generator = torch.Generator().manual_seed(7)
    clean = torch.randn((3, 4, 8, 8), generator=generator)
    noise = torch.randn((3, 4, 8, 8), generator=generator)
    alpha, sigma = alpha_sigma(torch.tensor([0.2, 0.5, 0.8]))
    broadcast_alpha = alpha[:, None, None, None]
    broadcast_sigma = sigma[:, None, None, None]
    noisy = broadcast_alpha * clean + broadcast_sigma * noise
    velocity = broadcast_alpha * noise - broadcast_sigma * clean
    recovered_clean, recovered_noise = clean_noise_from_v(noisy, velocity, alpha, sigma)
    torch.testing.assert_close(recovered_clean, clean, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(recovered_noise, noise, atol=2e-6, rtol=2e-6)


def test_ddpm_step_matches_closed_form_posterior_mean() -> None:
    generator = torch.Generator().manual_seed(11)
    clean = torch.randn((2, 4, 4, 4), generator=generator)
    forward_noise = torch.randn((2, 4, 4, 4), generator=generator)
    t = torch.tensor([0.7, 0.7])
    s = torch.tensor([0.4, 0.4])
    alpha_t, sigma_t = alpha_sigma(t)
    alpha_s, sigma_s = alpha_sigma(s)
    noisy = alpha_t[:, None, None, None] * clean + sigma_t[:, None, None, None] * forward_noise
    velocity = alpha_t[:, None, None, None] * forward_noise - sigma_t[:, None, None, None] * clean

    actual = ddpm_posterior_step(
        noisy, velocity, alpha_t, sigma_t, alpha_s, sigma_s, noise=None
    )
    alpha_bar_t = alpha_t.square()
    alpha_bar_s = alpha_s.square()
    incremental_alpha = alpha_bar_t / alpha_bar_s
    expected = (
        (
            alpha_s * (1.0 - incremental_alpha) / (1.0 - alpha_bar_t)
        )[:, None, None, None]
        * clean
        + (
            incremental_alpha.sqrt() * (1.0 - alpha_bar_s) / (1.0 - alpha_bar_t)
        )[:, None, None, None]
        * noisy
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_selective_activation_checkpointing_preserves_output_and_gradients() -> None:
    torch.manual_seed(23)
    reference = FixedLoopDiT(
        latent_size=4,
        patch_size=2,
        channels=2,
        classes=7,
        hidden_size=32,
        heads=4,
        mlp_ratio=2,
        unique_blocks=2,
        loops=3,
    )
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    checkpointed = copy.deepcopy(reference)
    checkpointed.activation_checkpoint_every = 2

    noisy = torch.randn(3, 2, 4, 4)
    timestep = torch.rand(3)
    labels = torch.tensor([0, 3, 6])
    expected = reference(noisy, timestep, labels)
    actual = checkpointed(noisy, timestep, labels)
    torch.testing.assert_close(actual, expected)

    expected.square().mean().backward()
    actual.square().mean().backward()
    for expected_parameter, actual_parameter in zip(
        reference.parameters(), checkpointed.parameters(), strict=True
    ):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad)
