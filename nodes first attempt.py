"""
ComfyUI-Spectrum-LTX25: Adaptive Spectral Feature Forecasting for LTX 2.5
Hybrid Release: Ground-truth auto-detection with deterministic manual override and guided presets.
Based on and adapted from ComfyUI-Spectrum-MiniMax-H3:
Copyright (C) 2026 xmarre (https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3)
Ported and adapted for LTX-Video / LTX-2.x models by gavr728 (2026-09-26).

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import time
import torch
import logging

logger = logging.getLogger("SpectrumLTX25")


def solve_affine_chebyshev(
    step_history: list[int],
    step_target: int,
    degree: int = 1,
    ridge_lambda: float = 0.10,
    blend_weight: float = 0.50,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    K = len(step_history)
    if K <= 1:
        return torch.tensor([1.0], dtype=torch.float32, device=device)

    # 1. Local sliding window mapping to [-1, 1]; target anchored at boundary (+1.0)
    s_min = float(step_history[0])
    s_max = float(step_target)
    span = max(s_max - s_min, 1e-4)

    tau_hist = torch.tensor(
        [2.0 * (s - s_min) / span - 1.0 for s in step_history],
        dtype=torch.float32,
        device=device
    )
    tau_tgt = 1.0

    m = min(degree, K - 1, 2)

    Phi = torch.zeros((K, m + 1), dtype=torch.float32, device=device)
    Phi[:, 0] = 1.0
    if m >= 1:
        Phi[:, 1] = tau_hist
    if m >= 2:
        Phi[:, 2] = 2.0 * tau_hist * tau_hist - 1.0

    phi_tgt = torch.zeros((1, m + 1), dtype=torch.float32, device=device)
    phi_tgt[0, 0] = 1.0
    if m >= 1:
        phi_tgt[0, 1] = tau_tgt
    if m >= 2:
        phi_tgt[0, 2] = 2.0 * tau_tgt * tau_tgt - 1.0

    reg = ridge_lambda * torch.eye(m + 1, dtype=torch.float32, device=device)
    A = Phi.T @ Phi + reg
    try:
        sol = torch.linalg.solve(A, phi_tgt.squeeze(0))
        alpha_cheb = Phi @ sol
    except Exception:
        inv = torch.linalg.pinv(A)
        alpha_cheb = (phi_tgt @ inv @ Phi.T).squeeze(0)

    # Affine Normalization: sum(weights) == 1.0 prevents velocity scaling drift
    s_cheb = torch.sum(alpha_cheb)
    if abs(s_cheb.item()) > 1e-4:
        alpha_cheb = alpha_cheb / s_cheb
    else:
        alpha_cheb = torch.zeros(K, dtype=torch.float32, device=device)
        alpha_cheb[-1] = 1.0

    # Local Taylor slope
    gamma_local = torch.zeros(K, dtype=torch.float32, device=device)
    ds = float(step_history[-1] - step_history[-2])
    if ds > 0:
        beta = min(max((step_target - step_history[-1]) / ds, 0.0), 1.0)
        gamma_local[-1] = 1.0 + beta
        gamma_local[-2] = -beta
    else:
        gamma_local[-1] = 1.0

    weights = blend_weight * alpha_cheb + (1.0 - blend_weight) * gamma_local
    total_w = torch.sum(weights)
    if abs(total_w.item()) > 1e-4:
        weights = weights / total_w

    return weights


class HybridSpectrumState:
    def __init__(self, max_history=6, storage="system_ram"):
        self.max_history = max_history
        self.storage = storage

        self.history_steps = {}    # stream_idx -> list of step indices
        self.history_outputs = {}  # stream_idx -> list of stored tensors

        self.current_outer_step = -1
        self.stream_counter = 0
        self.total_calls = 0
        self.last_sigma = None
        self.last_seen_time = 0.0

        self.actual_steps_done = 0
        self.forecast_steps_done = 0
        self.consecutive_skips = 0
        self.step_should_forecast = False
        self.step_start_time = 0.0

    def reset(self):
        self.history_steps.clear()
        self.history_outputs.clear()
        self.current_outer_step = -1
        self.stream_counter = 0
        self.total_calls = 0
        self.last_sigma = None
        self.last_seen_time = 0.0
        self.actual_steps_done = 0
        self.forecast_steps_done = 0
        self.consecutive_skips = 0
        self.step_should_forecast = False
        self.step_start_time = 0.0

    def store_anchor(self, stream_idx, step, output):
        if stream_idx not in self.history_steps:
            self.history_steps[stream_idx] = []
            self.history_outputs[stream_idx] = []

        stored = self._offload(output)
        self.history_steps[stream_idx].append(step)
        self.history_outputs[stream_idx].append(stored)

        if len(self.history_steps[stream_idx]) > self.max_history:
            self.history_steps[stream_idx].pop(0)
            self.history_outputs[stream_idx].pop(0)

    def _offload(self, item):
        if isinstance(item, torch.Tensor):
            if self.storage == "system_ram":
                return item.detach().to("cpu", copy=True)
            return item.detach()
        elif isinstance(item, (list, tuple)):
            return type(item)(self._offload(x) for x in item)
        return item


class SpectrumApplyLTX25:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "passes_per_step": ("INT", {
                    "default": 0, "min": 0, "max": 32, "step": 1,
                    "tooltip": (
                        "0 = Auto-detect (Default. Works automatically for most setups, "
                        "including LTXV Dual CFG Guider with Euler or res_2s).\n\n"
                        "WHEN TO SWITCH TO MANUAL:\n"
                        "Switch to manual only if auto-detection fails (e.g. the console step counter "
                        "advances faster than the progress bar, which can happen with complex MultimodalGuider setups).\n\n"
                        "HOW TO CALCULATE:\n"
                        "Formula: Guider Passes × Sampler Stages\n\n"
                        "1. Sampler Stages:\n"
                        "   • Euler / Euler Ancestral / UniPC / res_2m = 1\n"
                        "   • res_2s / Heun / dpm_2 = 2\n\n"
                        "2. Guider Passes:\n"
                        "   • CFG = 1.0 (Distilled mode) = 1\n"
                        "   • LTXV Dual CFG Guider (or standard CFGGuider) = 2\n"
                        "   • MultimodalGuider (STG = 0) = 4\n"
                        "   • MultimodalGuider (Video STG > 0, Audio STG = 0) = 5\n"
                        "   • MultimodalGuider (Video STG > 0, Audio STG > 0) = 6\n\n"
                        "QUICK PRESETS (Guider × Sampler):\n"
                        "   • LTXV Dual CFG Guider + Euler = 2 (or leave at 0 Auto)\n"
                        "   • LTXV Dual CFG Guider + res_2s = 4 (or leave at 0 Auto)\n"
                        "   • MultimodalGuider (STG = 0) + res_2s = 8\n"
                        "   • MultimodalGuider (Video STG > 0) + res_2s = 10\n"
                        "   • MultimodalGuider (Video + Audio STG > 0) + res_2s = 12"
                    )
                }),
                "blend_weight": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Chebyshev share for video. 0.50 blends 50% polynomial forecast with 50% Taylor slope."
                }),
                "audio_blend_weight": ("FLOAT", {
                    "default": 0.00, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Keep at 0.00 to preserve audio phase coherence and prevent muffling/distortion."
                }),
                "degree": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "1 = Linear trend (recommended & stable), 2 = Quadratic curvature."
                }),
                "ridge_lambda": ("FLOAT", {
                    "default": 0.10, "min": 1e-4, "max": 2.0, "step": 0.01,
                    "tooltip": "Tikhonov regularization parameter; stabilizes the ridge regression solve."
                }),
                "window_size": ("INT", {
                    "default": 2, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Base number of steps to skip per real network evaluation."
                }),
                "flex_window": ("FLOAT", {
                    "default": 0.35, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Incremental expansion of the skip window during mid-generation."
                }),
                "warmup_steps": ("INT", {
                    "default": 3, "min": 1, "max": 20, "step": 1,
                    "tooltip": "Initial real steps to anchor the polynomial trajectory."
                }),
                "tail_actual_steps": ("INT", {
                    "default": 2, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Final steps forced to real DiT evaluations for fine detail preservation."
                }),
                "max_history": ("INT", {
                    "default": 6, "min": 3, "max": 16, "step": 1,
                    "tooltip": "Maximum number of past evaluated anchors to retain per stream."
                }),
                "history_storage": (["system_ram", "vram"], {
                    "default": "system_ram",
                    "tooltip": "Store anchors in system RAM to preserve VRAM on consumer GPUs."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_spectrum"
    CATEGORY = "LTXVideo/acceleration"

    def apply_spectrum(
        self,
        model,
        enabled,
        passes_per_step=0,
        blend_weight=0.50,
        audio_blend_weight=0.00,
        degree=1,
        ridge_lambda=0.10,
        window_size=2,
        flex_window=0.35,
        warmup_steps=3,
        tail_actual_steps=2,
        max_history=6,
        history_storage="system_ram",
    ):
        if not enabled:
            return (model,)

        # Type sanitization against shifted widget values from saved JSON workflows
        if isinstance(passes_per_step, bool) or int(passes_per_step) < 0:
            passes_per_step = 0
        else:
            passes_per_step = int(passes_per_step)

        try:
            degree = min(max(int(degree), 1), 4)
        except Exception:
            degree = 1

        try:
            warmup_steps = max(int(warmup_steps), 1)
        except Exception:
            warmup_steps = 3

        try:
            tail_actual_steps = max(int(tail_actual_steps), 0)
        except Exception:
            tail_actual_steps = 2

        try:
            window_size = max(int(window_size), 1)
        except Exception:
            window_size = 2

        try:
            blend_weight = float(blend_weight)
        except Exception:
            blend_weight = 0.50

        try:
            audio_blend_weight = float(audio_blend_weight)
        except Exception:
            audio_blend_weight = 0.00

        patched_model = model.clone()
        runtime = HybridSpectrumState(max_history=max_history, storage=history_storage)

        def spectrum_wrapper(apply_model, args):
            now = time.time()
            input_x = args["input"]
            timestep = args["timestep"]
            c = args["c"]

            t_val = timestep[0].item() if isinstance(timestep, torch.Tensor) else float(timestep)

            # Auto-detect total scheduled steps from ComfyUI's scheduler
            total_steps = 25
            sample_sigmas = c.get("transformer_options", {}).get("sample_sigmas", None)
            if sample_sigmas is None:
                sample_sigmas = c.get("transformer_options", {}).get("sigmas", None)
            if sample_sigmas is not None and len(sample_sigmas) > 1:
                total_steps = max(len(sample_sigmas) - 1, 1)

            # -------------------------------------------------------------
            # MODE A: MANUAL OVERRIDE (passes_per_step > 0)
            # Used if a specific complex setup desyncs from the progress bar
            # -------------------------------------------------------------
            if passes_per_step > 0:
                is_new_prompt = (
                    runtime.last_sigma is None
                    or t_val > (runtime.last_sigma + 0.05)
                    or (runtime.last_seen_time > 0 and (now - runtime.last_seen_time) > 120.0)
                    or runtime.total_calls >= (total_steps * passes_per_step)
                )
                if is_new_prompt:
                    runtime.reset()

                step_idx = runtime.total_calls // passes_per_step
                stream_idx = runtime.total_calls % passes_per_step

                if stream_idx == 0:
                    runtime.step_start_time = time.time()
                    is_warmup = step_idx < warmup_steps
                    is_tail = step_idx >= (total_steps - tail_actual_steps)
                    has_history = len(runtime.history_steps.get(0, [])) >= 2
                    allowed_skips = int(round(
                        window_size + flex_window * max(runtime.actual_steps_done - warmup_steps, 0)
                    ))

                    runtime.step_should_forecast = (
                        not is_warmup
                        and not is_tail
                        and has_history
                        and runtime.consecutive_skips < allowed_skips
                    )

                    if runtime.step_should_forecast:
                        runtime.consecutive_skips += 1
                        runtime.forecast_steps_done += 1
                    else:
                        runtime.consecutive_skips = 0
                        runtime.actual_steps_done += 1

            # -------------------------------------------------------------
            # MODE B: AUTO-DETECTION (passes_per_step == 0)
            # Resolves step directly from ComfyUI's schedule table
            # -------------------------------------------------------------
            else:
                if sample_sigmas is not None and len(sample_sigmas) > 1:
                    sig_t = sample_sigmas[:-1] if sample_sigmas[-1] == 0.0 else sample_sigmas
                    diffs = torch.abs(sig_t.to(input_x.device) - t_val)
                    step_idx = torch.argmin(diffs).item()
                else:
                    step_idx = 0

                # Detect when sampler moves to next step
                if step_idx != runtime.current_outer_step:
                    if step_idx < runtime.current_outer_step or runtime.current_outer_step == -1 or (now - runtime.last_seen_time) > 120.0:
                        runtime.reset()

                    runtime.current_outer_step = step_idx
                    runtime.stream_counter = 0
                    runtime.step_start_time = time.time()

                    is_warmup = step_idx < warmup_steps
                    is_tail = step_idx >= (total_steps - tail_actual_steps)
                    has_history = len(runtime.history_steps.get(0, [])) >= 2
                    allowed_skips = int(round(
                        window_size + flex_window * max(runtime.actual_steps_done - warmup_steps, 0)
                    ))

                    runtime.step_should_forecast = (
                        not is_warmup
                        and not is_tail
                        and has_history
                        and runtime.consecutive_skips < allowed_skips
                    )

                    if runtime.step_should_forecast:
                        runtime.consecutive_skips += 1
                        runtime.forecast_steps_done += 1
                    else:
                        runtime.consecutive_skips = 0
                        runtime.actual_steps_done += 1

                stream_idx = runtime.stream_counter
                runtime.stream_counter += 1

            runtime.total_calls += 1
            runtime.last_seen_time = now
            runtime.last_sigma = t_val

            # Helper to reliably print the summary at 100% of generation
            def maybe_print_summary():
                is_last = (
                    (passes_per_step > 0 and step_idx >= total_steps - 1 and stream_idx == passes_per_step - 1)
                    or (passes_per_step == 0 and step_idx >= total_steps - 1 and stream_idx == 0)
                )
                if is_last:
                    total_ev = runtime.actual_steps_done
                    total_sk = runtime.forecast_steps_done
                    total_all = total_ev + total_sk
                    speedup = (total_all / total_ev) if total_ev > 0 else 1.0
                    print("\n" + "=" * 60)
                    print(" [Spectrum LTX 2.5 Acceleration Summary]")
                    print(f"  - Total outer steps     : {total_all}")
                    print(f"  - Real DiT evaluations  : {total_ev}")
                    print(f"  - Forecasted steps saved: {total_sk} ({total_sk/total_all*100:.1f}% skipped)")
                    print(f"  - Effective speedup     : ~{speedup:.2f}x faster DiT")
                    print("=" * 60 + "\n", flush=True)

            # ==========================================
            # CASE A: FORECAST PASS (Fast Skip)
            # ==========================================
            if runtime.step_should_forecast:
                try:
                    s_hist = runtime.history_steps[stream_idx]
                    hist_outputs = runtime.history_outputs[stream_idx]

                    def forecast_tensor(tensors, w_blend):
                        weights = solve_affine_chebyshev(
                            s_hist, step_idx, degree, ridge_lambda, w_blend, device=torch.device("cpu")
                        )
                        target_dev = input_x.device if isinstance(input_x, torch.Tensor) else tensors[-1].device
                        pred = torch.zeros_like(tensors[-1], dtype=torch.float32, device=target_dev)
                        for k, tens in enumerate(tensors):
                            pred.add_(tens.to(target_dev, dtype=torch.float32), alpha=weights[k].item())

                        # Direction polarity safeguard: Prevent velocity vector field reversal
                        last_t = tensors[-1].to(target_dev, dtype=torch.float32)
                        if torch.sum(pred * last_t) <= 0:
                            pred = last_t

                        pred = torch.nan_to_num(pred, nan=0.0)
                        return pred.to(dtype=tensors[-1].dtype)

                    first = hist_outputs[0]
                    if isinstance(first, (list, tuple)):
                        out = []
                        for m_i in range(len(first)):
                            w_val = blend_weight if m_i == 0 else audio_blend_weight
                            if isinstance(first[m_i], torch.Tensor):
                                m_hist = [entry[m_i] for entry in hist_outputs]
                                out.append(forecast_tensor(m_hist, w_val))
                            else:
                                out.append(first[m_i])
                        output = type(first)(out)
                    else:
                        output = forecast_tensor(hist_outputs, blend_weight)

                    # Log once per step
                    is_log_boundary = (
                        (passes_per_step > 0 and stream_idx == passes_per_step - 1)
                        or (passes_per_step == 0 and stream_idx == 0)
                    )
                    if is_log_boundary:
                        elapsed_ms = (time.time() - runtime.step_start_time) * 1000.0
                        pass_info = f"All {passes_per_step} passes" if passes_per_step > 0 else "Passes"
                        print(f"  [Spectrum LTX 2.5] Step {step_idx:02d}/{total_steps} -> [FORECAST SKIP] ({pass_info} skipped in {elapsed_ms:.1f}ms)", flush=True)

                    maybe_print_summary()
                    return output
                except Exception as e:
                    logger.warning(f"Spectrum forecast exception on stream {stream_idx}: {e}. Evaluating real pass.")

            # ==========================================
            # CASE B: REAL EVALUATION (Model Forward Pass)
            # ==========================================
            t_pass_start = time.time()
            output = apply_model(input_x, timestep, **c)
            runtime.store_anchor(stream_idx, step_idx, output)
            pass_elapsed = time.time() - t_pass_start

            # Concise console output
            cadence_str = f"Pass {stream_idx + 1}/{passes_per_step}" if passes_per_step > 0 else f"Pass {stream_idx + 1}"
            is_warm = step_idx < warmup_steps
            is_tl = step_idx >= (total_steps - tail_actual_steps)
            reason = "WARMUP" if is_warm else ("TAIL RECOVERY" if is_tl else "REFRESH")
            print(f"  [Spectrum LTX 2.5] Step {step_idx:02d}/{total_steps} -> [REAL DiT EVAL] ({cadence_str}, {reason}, {pass_elapsed:.1f}s)", flush=True)

            maybe_print_summary()
            return output

        patched_model.set_model_unet_function_wrapper(spectrum_wrapper)
        return (patched_model,)