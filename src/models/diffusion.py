from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DiffusionScheduleConfig:
    num_steps: int
    beta_start: float
    beta_end: float


class DiffusionScheduler:
    def __init__(self, config: DiffusionScheduleConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.device = device or torch.device("cpu")
        self.betas = torch.linspace(config.beta_start, config.beta_end, config.num_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionScheduler":
        return DiffusionScheduler(self.config, device=device)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.config.num_steps, (batch_size,), device=self.device)

    def _alpha_bars_on(self, device: torch.device) -> torch.Tensor:
        if self.alpha_bars.device == device:
            return self.alpha_bars
        return self.alpha_bars.to(device)

    @staticmethod
    def _broadcast_coeff(coeff: torch.Tensor, target_ndim: int) -> torch.Tensor:
        return coeff.view(-1, *([1] * (target_ndim - 1)))

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bars = self._alpha_bars_on(timesteps.device)
        alpha_bar = self._broadcast_coeff(alpha_bars[timesteps], x0.ndim)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise

    def predict_start_from_noise(self, xt: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bars = self._alpha_bars_on(timesteps.device)
        alpha_bar = self._broadcast_coeff(alpha_bars[timesteps], xt.ndim)
        return (xt - torch.sqrt(1.0 - alpha_bar) * noise) / torch.sqrt(torch.clamp(alpha_bar, min=1e-6))

    def velocity_target(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bars = self._alpha_bars_on(timesteps.device)
        alpha_bar = self._broadcast_coeff(alpha_bars[timesteps], x0.ndim)
        return torch.sqrt(alpha_bar) * noise - torch.sqrt(1.0 - alpha_bar) * x0

    def predict_start_from_velocity(self, xt: torch.Tensor, timesteps: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        alpha_bars = self._alpha_bars_on(timesteps.device)
        alpha_bar = self._broadcast_coeff(alpha_bars[timesteps], xt.ndim)
        return torch.sqrt(alpha_bar) * xt - torch.sqrt(1.0 - alpha_bar) * velocity

    def predict_noise_from_velocity(self, xt: torch.Tensor, timesteps: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        alpha_bars = self._alpha_bars_on(timesteps.device)
        alpha_bar = self._broadcast_coeff(alpha_bars[timesteps], xt.ndim)
        return torch.sqrt(alpha_bar) * velocity + torch.sqrt(1.0 - alpha_bar) * xt

    def training_target(self, x0: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, target_type: str) -> torch.Tensor:
        if target_type == "epsilon":
            return noise
        if target_type == "x0":
            return x0
        if target_type == "v":
            return self.velocity_target(x0, noise, timesteps)
        raise ValueError(f"Unsupported diffusion target_type: {target_type}")

    def predict_start_from_model_output(
        self,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        model_output: torch.Tensor,
        target_type: str,
    ) -> torch.Tensor:
        if target_type == "epsilon":
            return self.predict_start_from_noise(xt, timesteps, model_output)
        if target_type == "x0":
            return model_output
        if target_type == "v":
            return self.predict_start_from_velocity(xt, timesteps, model_output)
        raise ValueError(f"Unsupported diffusion target_type: {target_type}")

    def min_snr_weight(self, timesteps: torch.Tensor, gamma: float, target_type: str) -> torch.Tensor:
        if gamma <= 0.0:
            return torch.ones_like(timesteps, dtype=torch.float32)
        alpha_bars = self._alpha_bars_on(timesteps.device)
        alpha_bar = torch.clamp(alpha_bars[timesteps], min=1e-6, max=1 - 1e-6)
        snr = alpha_bar / torch.clamp(1.0 - alpha_bar, min=1e-6)
        gamma_tensor = torch.full_like(snr, float(gamma))
        if target_type == "epsilon":
            return torch.minimum(snr, gamma_tensor) / snr
        return torch.minimum(snr, gamma_tensor)

    def _sampling_indices(self, sample_steps: int) -> torch.Tensor:
        if sample_steps >= self.config.num_steps:
            return torch.arange(self.config.num_steps - 1, -1, -1, device=self.device)
        indices = torch.linspace(self.config.num_steps - 1, 0, sample_steps, device=self.device).round().long()
        return torch.unique(indices, sorted=True).flip(0)

    def _guided_prediction(self, model, sample: torch.Tensor, timesteps: torch.Tensor, model_kwargs, guidance_scale: float) -> torch.Tensor:
        conditional = model(noisy_future=sample, timesteps=timesteps, **model_kwargs)
        if guidance_scale != 1.0:
            unconditional_kwargs = dict(model_kwargs)
            if model_kwargs.get("target_behavior") is not None:
                unconditional_kwargs["target_behavior"] = torch.zeros_like(model_kwargs["target_behavior"])
            if model_kwargs.get("target_difficulty") is not None:
                unconditional_kwargs["target_difficulty"] = torch.zeros_like(model_kwargs["target_difficulty"])
            unconditional_kwargs["condition_force_drop"] = True
            unconditional = model(noisy_future=sample, timesteps=timesteps, **unconditional_kwargs)
            return unconditional + guidance_scale * (conditional - unconditional)
        return conditional

    @torch.no_grad()
    def sample_loop(
        self,
        model,
        shape,
        model_kwargs,
        sample_steps: int,
        guidance_scale: float = 1.0,
        sampler: str = "ddpm",
        ddim_eta: float = 0.0,
        initial_noise: torch.Tensor | None = None,
        target_type: str = "epsilon",
    ) -> torch.Tensor:
        sample = torch.randn(shape, device=self.device) if initial_noise is None else initial_noise.to(self.device).clone()
        indices = self._sampling_indices(sample_steps)
        if sampler not in {"ddpm", "ddim"}:
            raise ValueError(f"Unsupported sampler: {sampler}")
        for step_idx, timestep in enumerate(indices):
            t = torch.full((shape[0],), int(timestep.item()), device=self.device, dtype=torch.long)
            model_prediction = self._guided_prediction(model, sample, t, model_kwargs, guidance_scale)
            predicted_noise = (
                    model_prediction
                    if target_type == "epsilon"
                    else (
                        self.predict_noise_from_velocity(sample, t, model_prediction)
                        if target_type == "v"
                    else (sample - torch.sqrt(self._broadcast_coeff(self._alpha_bars_on(t.device)[t], sample.ndim)) * model_prediction)
                    / torch.sqrt(torch.clamp(1.0 - self._broadcast_coeff(self._alpha_bars_on(t.device)[t], sample.ndim), min=1e-6))
                )
            )
            alpha = self.alphas[timestep]
            alpha_bar = self.alpha_bars[timestep]
            if sampler == "ddpm":
                beta = self.betas[timestep]
                if timestep > 0:
                    noise = torch.randn_like(sample)
                else:
                    noise = torch.zeros_like(sample)
                sample = (sample - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise) / torch.sqrt(alpha) + torch.sqrt(beta) * noise
                continue
            prev_timestep = int(indices[step_idx + 1].item()) if step_idx + 1 < len(indices) else -1
            alpha_bar_prev = torch.tensor(1.0, device=self.device, dtype=sample.dtype) if prev_timestep < 0 else self.alpha_bars[prev_timestep]
            pred_x0 = self.predict_start_from_model_output(sample, t, model_prediction, target_type)
            if float(alpha_bar_prev) <= 0.0:
                sample = pred_x0
                continue
            sigma = float(ddim_eta) * torch.sqrt((1 - alpha_bar_prev) / torch.clamp(1 - alpha_bar, min=1e-6)) * torch.sqrt(
                torch.clamp(1 - alpha_bar / torch.clamp(alpha_bar_prev, min=1e-6), min=0.0)
            )
            noise = torch.randn_like(sample) if prev_timestep >= 0 and float(ddim_eta) > 0.0 else torch.zeros_like(sample)
            direction = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma**2, min=0.0)) * predicted_noise
            sample = torch.sqrt(torch.clamp(alpha_bar_prev, min=1e-6)) * pred_x0 + direction + sigma * noise
        return sample
