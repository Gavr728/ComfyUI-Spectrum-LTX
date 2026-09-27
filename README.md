# ComfyUI-Spectrum-LTX

Fast **Spectrum acceleration** (Adaptive Spectral Feature Forecasting, arXiv:2603.01623) for the **LTX-2.5 22B Base** foundation model in ComfyUI.

> **Vibecoded Disclaimer:** This custom node is 100% vibecoded. The author does not know how to code. 

Skips 30–55% of transformer evaluations on a 20–30 step base generation by forecasting intermediate steps with a Chebyshev polynomial fit instead of running the full DiT. Speedup is **~1.5×–2.3× fewer model evaluations**, scaling with your step count and chosen preset — see the table below for exact numbers, not marketing rounding.

> **Honesty note:** this is an approximation, not a free lunch. Forecast steps are not identical to real ones. On the `quality` preset the difference is usually not visible; push toward `fast` and you will start to see it, especially on fast motion and on-screen text. Test before you commit to a preset for a final render.

---

## How it works (and what changed from earlier versions)

The node hooks three points in ComfyUI's execution graph via `WrappersMP`: `OUTER_SAMPLE`, `PREDICT_NOISE`, and `DIFFUSION_MODEL`. Nothing is guessed from call counts, timers, or manual pass multipliers anymore — the real sigma schedule and sampler object are read directly.

* **Automatic sampler detection.** During warmup (which always runs the real model), the node measures whether the sampler injects fresh noise between steps. Deterministic samplers (Euler, `res_multistep`, SEEDS-2 with `eta=0`, …) forecast in **feature space**: the final transformer block's hidden state is cached, forecast, and passed through LTX's own output head re-run at the *current* timestep — the method described in the paper. Noise-injecting samplers (ancestral, SDE, LCM, SEEDS-2 with `eta>0`, …) forecast in **denoised (x0) space** instead, rebuilding velocity from the *current* noisy latent so the sampler's injected noise is respected rather than skipped over.
* **Audio is always forecast in denoised space**, regardless of the video mode, because raw audio features don't extrapolate as smoothly as video features do.
* **Redundant zero-sigma tail steps are stripped automatically.** If your scheduler produces a sigma schedule ending `..., 0.0, 0.0` (e.g. `LTXVScheduler` with `stretch=true` and `terminal=0`), that final step is a division-by-zero no-op for Euler (and a wasted model call for everything else). The node removes it before sampling and logs how many it dropped. **This fix only applies while Spectrum is enabled** — if you hit this NaN with the node disabled, set `terminal` to a small positive value (e.g. `0.1`) on your scheduler node.
* There is **no manual override input anymore**. `passes_per_step` is gone. Detection reads `sample_sigmas`/`sigmas` and the sampler object directly, and correctly separates cond/uncond/STG streams by their actual identity rather than call order, so it doesn't desync on `MultimodalGuider` or multi-stage samplers the way call-counting did.

---

## Node inputs

| Input | Default | Notes |
| :--- | :---: | :--- |
| `preset` | `balanced` | See speedup table below. `quality` = always a real step after each forecast. `balanced` = up to 2 forecasts in a row in the later half of the run. `fast` = more aggressive, more drift. |
| `warmup_steps` | `5` | Initial steps that always run the real model. Also the window used to detect sampler stochasticity. Raise this (7–8) if you rely on crisp on-screen text or fine composition. |
| `tail_actual_steps` | `2` | Final steps that always run the real model. |
| `degree` | `3` | Chebyshev polynomial degree for the spectral fit. |
| `ridge_lambda` | `0.10` | Ridge regularization on the fit (intercept is left unpenalized, so weights always sum to exactly 1 — no amplitude drift). |
| `blend_weight` | `0.50` | Share of the forecast taken from the Chebyshev fit vs. a 2-point linear (Taylor) extrapolation. |
| `max_history` | `8` | Max anchors retained per stream. |
| `forecast_space` | `auto` | Leave on `auto` unless you're debugging — it already picks `features` or `denoised` correctly per sampler. |
| `history_storage` | `system_ram` | Use `vram` only if you have headroom; saves a host/device copy per anchor. |
| `validate` | `false` | Also runs the real model on forecast steps and logs relative error vs. the forecast (video/audio separately). No speedup while on — use it to sanity-check a new preset/prompt before trusting it. |
| `debug` | `false` | Logs a line per step (`ACTUAL`/`FORECAST`). |

---

## Speedup by preset

Measured as *model evaluations*, not wall-clock — text encoding, VAE decode, and any second-stage refinement pass are unaffected. On a 19–20 step first stage:

| preset | real evals | forecast evals | speedup |
| :--- | :---: | :---: | :---: |
| `quality` | 13 | 6–7 | **~1.54×** |
| `balanced` (default) | 12 | 7–8 | **~1.67×** |
| `fast` | 10 | 9–10 | **~2.0×** |

At 30 steps, `fast` reaches roughly **2.3×**. Longer schedules always benefit more, because warmup/tail are a fixed cost and the middle of the run — where skipping happens — grows.

---

### Practical Negative Prompting Rules
* **Target Concrete Objects:** Use negative prompts for physical items you want absent (`red hat, glasses`), not abstract quality descriptors.
* **Always Pair with DiffVAE (`video-vae-bf16`):** Prevents high guidance pressure (CFG/NAG) from blowing out highlights or creating neon/burned edges.

---

## Recommended Node Stack for LTX 2.5 Base

* **Stage 1: Base Generation (Half-Resolution, e.g. 960×544):**
  * **Model:** `ltx-2.5-22b-dev-transformer-bf16` → `SpectrumApplyLTX`
  * **Guider:** `LTXVDualCFGGuider` (Video CFG: `3.5`–`4.0`, Audio CFG: `1.0`–`7.0`). *Use `MultimodalGuider` only if you explicitly need STG.*
  * **Scheduler:** `LTXVScheduler` — set `terminal` to a small positive value (e.g. `0.1`), not `0`, to avoid a wasted/NaN-prone final step even without Spectrum.
  * **Sampler:** 15–25 steps
* **Bridge:**
  * `LTXVLatentUpsampler`
* **Stage 2: Refinement (Full Resolution):**
  * **Model:** Same Dev model + `ltx-2.5-22b-distilled-lora-450-bf16` at **`1.0` strength**
  * **Guider:** `CFGGuider` (`cfg: 1.0`)
  * **Sigmas on second pass:** `0.8025, 0.6332, 0.3425, 0.0` (3 steps)
  * **Spectrum should not be applied here** — 3 steps leaves nothing to safely forecast; the node will detect this and disable itself automatically (see below), but skip the node entirely on this stage to save the wrapper overhead.
* **Final Decode:**
  * `VAEDecodeTiled` with `ltx-2.5-video-vae-bf16.safetensors` (`DiffVAE`) to prevent motion smearing and color clipping.

---

## Using Spectrum on short / distilled schedules

**General recommendation: don't.** Spectrum needs a smooth multi-step trajectory to fit a polynomial to. An 8-step distilled run is mostly warmup and tail by construction — the node will print `inactive for this run: warmup (N) + tail (N) leave nothing to forecast` and simply run natively, which is the safe and expected outcome. Don't fight this by cranking `warmup_steps`/`tail_actual_steps` down to 1; the middle 2–3 steps you'd "save" aren't enough to fit `degree=3` reliably and you'll get stuttering motion for a marginal speed gain.

If you insist on trying it anyway, use `preset=quality`, `degree=1`, `warmup_steps=2`, `tail_actual_steps=1`, `blend_weight=0.3` — but validate first (`validate=true`) and expect visible artifacts.

*Spectrum is designed for the **Base model**, where 20–30 steps provide the continuous trajectory needed for reliable forecasting.*

---

## Troubleshooting

* **`inactive for this run: ...`** in the console — this is normal, not an error. It means the schedule was too short, or warmup+tail consumed the whole run. Nothing was skipped; output is identical to having the node disabled.
* **Euler produces NaN / audio encoder errors** — check for a `removed N redundant zero-sigma step(s)` line. If it's missing and you still get NaN, the problem is your scheduler's `terminal` setting, not Spectrum; set it above `0`.
* **Results drift more than expected** — try `preset=quality`, raise `warmup_steps`, and turn on `validate` to see per-step relative error (target: under ~0.05 on `x0`). If audio error is much higher than video, the audio VAE latents may be unusually non-smooth for your prompt; the video forecast is the one most closely following the paper's method.

---

## In-Depth Documentation

For complete mathematical derivations, bug post-mortems (`NestedTensor` handling, sampler noise detection, zero-sigma tail bug), and wiring diagrams, see the full guide:

👉 **[Base Practical Workflow Guide.md](./Base%20Practical%20Workflow%20Guide.md)**
👉 **[Official negative prompting Guide](https://ltx.io/blog/negative-prompts)**

--

## Acknowledgements & Attribution

- **Spectrum Algorithm**: Based on *“Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration”* (arXiv:2603.01623) by Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, and Stefano Ermon.
- **Upstream ComfyUI Integration**: Adapted and modified from [ComfyUI-Spectrum-MiniMax-H3](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3) by **xmarre**.

---

## License

This project is licensed under the **GNU General Public License v3.0 or later (GPL-3.0-or-later)**. See the [LICENSE](LICENSE.txt) file for the full license text.