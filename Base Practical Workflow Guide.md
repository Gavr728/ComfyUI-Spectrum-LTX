LTX-2.5 Base Practical Workflow Guide

This document summarizes findings, bug post-mortems, and practical optimizations
for LTX-2.5 within ComfyUI, specifically tailored for IC-LoRA Video-to-Video and
Image-to-Video pipelines.

1. Official Pipelines vs. Practical Reality

What Official Code Does

In Lightricks' official repository, there are two primary two-stage
implementations:

1.  ti2vid_two_stages.py (Standard): Stage 1 runs the uncompressed Dev model
    (full CFG, 20–25 steps, no distilled LoRA) with standard Euler. Stage 2
    upscales 2\times and refines the latent using Euler + Distilled LoRA at 0.8
    strength for 3 steps (STAGE_2_DISTILLED_SIGMAS =
    [0.909375, 0.725, 0.421875, 0.0]).
2.  ti2vid_two_stages_hq.py (HQ): Attaches Distilled LoRA (0.8) to both stages
    and uses the second-order res_2s sampler.

What Works Best in Practice (ComfyUI + IC-LoRA)

  - Stage 2 Distilled LoRA Strength: While the official script uses 0.8, in
    ComfyUI with SamplerCustomAdvanced, setting the Distilled LoRA on Stage 2 to
    1.0 yields noticeably crisper results. At 0.8, roughly 20% of the model
    still behaves like the 25-step base model, leaving the 3-step polish
    slightly under-converged.
  - Stage 2 Audio is Redundant: In both official pipelines and practical setups,
    Stage 2 refines video only. The audio output from Stage 2 is discarded; the
    final video's audio is either decoded from Stage 1 or directly muxed from
    the source file.

2. Node Setup, Connections & When to Use What

[Load Source Video] ──(Video)──► [Source Video Preprocess]
       │                                │
    (Audio)                     (Latent, 960x544)
       │                                │
       │                                ▼
       │                    [STAGE 1: Base Generation]
       │                    • LTX 2.5 Dev Model + Spectrum Node
       │                    • Guider: LTXV Dual CFG Guider (Video: 3.5, Audio: 1.0)
       │                    • Scheduler: LTXVScheduler 
       │                    • Sampler: Euler or res_2s (15–25 steps)
       │                                │
       │                                ▼
       │                    [LTXVLatentUpsampler (2x)]
       │                    • Doubles spatial latent 
       │                                │
       │                                ▼
       │                    [STAGE 2: Latent Refiner]
       │                    • Model + Distilled LoRA (Strength: 1.0)
       │                    • Guider: CFGGuider (CFG: 1.0)
       │                    • Sigmas: 0.8025, 0.6332, 0.3425, 0.0 (3 steps)
       │                    • Sampler: Euler
       │                                │
       │                                ▼
       │                    [Decode & Assembly]
       │                    • Video: VAEDecodeTiled (DiffVAE / video-vae-bf16)
       └──────(Raw Audio Track)────────►• Mux: CreateVideo

Node Guidelines:

LTXVScheduler

  - Rule: Never connect the latent wire into this node when doing V2V.
  - Why: If connected on clips longer than ~3 seconds (e.g., 121\text{ frames},
    960 \times 544 \implies 8,160\text{ tokens}), the node's unbounded
    token-shift formula calculates an extreme shift (e^{3.5} \approx 33). Early
    step differences collapse to \Delta \sigma \approx 0.0001, which drops below
    bfloat16 machine precision floor (2^{-7} \approx 0.0078). Consecutive steps
    round to identical numbers, the sampler computes dt = 0.0, and the model
    explodes into NaNs (solid black output).
  - Settings: Disconnect latent, set terminal: 0.1 (not 0.0 — see Section 7 for
    why a hard zero causes a second, unrelated NaN failure mode with Euler),
    max_shift: 2.05, base_shift: 0.95.

LTXVDualCFGGuider vs. MultimodalGuider

  - LTXVDualCFGGuider (Recommended):
      - Evaluates standard CFG across video and audio ([Positive, Negative])
        without extra passes.
      - Pass count: Exactly 2 passes with Euler (or 4 with res_2s).
      - Fast, stable, and completely avoids multimodal attention blowups.
  - MultimodalGuider (Heavy / STG):
      - Evaluates Video (Pos, Neg, Perturbed) + Audio (Pos, Neg, Perturbed).
      - Pass count: Explodes to 6 passes with Euler, or 10 to 12 passes with
        res_2s.
      - When to use: Only if you explicitly need STG to force dramatic camera
        sweeps or aggressive motion that the model otherwise refuses to animate.
      - Required STG Settings: Set skip_blocks: 28 and set rescale = 0.75 to
        0.90 (crucial to prevent overburning).

LTXVSeparateAVLatent vs. LTXVCropGuides (Crash Prevention)

  - The NestedTensor Bug: SamplerCustomAdvanced outputs a packed Audio-Video
    NestedTensor. If plugged directly into LTXVCropGuides, it crashes with:
    AttributeError: 'NestedTensor' object has no attribute 'clone'.
  - Fix: You must route the sampler output through LTXVSeparateAVLatent first to
    unpack the NestedTensor into a standard torch.Tensor video latent before
    sending it to LTXVCropGuides.
  - The Inverse Crash: If running a pure video workflow (audio latent completely
    disconnected upstream), do not use LTXVSeparateAVLatent. It will crash with
    IndexError: tuple index out of range because there is no second latent to
    unpack.

3. Alternative Sigmas & Refinement Schedules

When refining a 2\times upscaled latent in Stage 2, selecting the correct sigma
schedule determines whether the model adds texture or destroys the video.

Schedule Comparison

| Schedule                                          | Steps | Target Use Case                                        | Behavior & Trade-offs                                                                                                                                                                                         |
| :------------------------------------------------ | :---: | :----------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`0.8025, 0.6332, 0.3425, 0.0`** *(Recommended)* | **3** | **Optimal Texture & Sharpness**                        | Injects \~80% noise. Preserves \~20% structural signal, which is the sweet spot to erase $2\times$ upscaler interpolation blur while allowing the model to paint sharp pores, hair strands, and cloth weaves. |
| **`0.6800, 0.4600, 0.2200, 0.0`**                 | **3** | **Strict Identity / Hand Locking**                     | Injects \~68% noise, keeping \>32% signal. Excellent when Stage 1 is already very sharp and you want to prevent small faces, fingers, or text from shifting.                                                  |
| **`0.909375, 0.725, 0.421875, 0.0`**              | **3** | *Official Distilled Tail (Not recommended in ComfyUI)* | Injects \~91% noise, destroying \>90% of the upscaled video. In ComfyUI's noise mixer, 3 steps are insufficient to rebuild 91% noise, leading to ghosting and soft, washed-out detail.                        |

Note: this is a separate 3-step Stage 2 sigma schedule, unrelated to the
`terminal` parameter on `LTXVScheduler` discussed in Section 2/7 — do not
confuse the two.

4. Running With vs. Without Audio Latent

Why Disconnecting the Audio Latent Accelerates Generation

In LTX-2.5, video tokens and audio tokens share joint attention. However:

1.  Pass Elimination: Disconnecting the audio latent (or setting Audio CFG to
    1.0) eliminates unrolled audio guidance passes. If using MultimodalGuider,
    step time drops from ~100s down to ~35s.
3.  Audio is Preserved Regardless: The original audio track from the source
    video should be connected directly to CreateVideo via Link bypass. The final
    MP4 retains the full soundtrack without wasting DiT compute on audio
    latents.

5. Guidance Mechanics & Cross-Modal Interactions

Audio CFG vs. Video CFG

When source audio is extracted and frozen (LTXVSetAudioRefTokens), it acts as an
implicit temporal anchor:

  - High Audio CFG (5.0 – 7.0):
    Forces the model to obey the timing and rhythm of the source clip's audio
    waveform. Through cross-modal attention, the video latents are leashed to
    the original framing, camera movement, and pacing.
    Result: High resemblance to the source video, but less creative freedom to
    invent new micro-textures.
  - Low Audio CFG (1.0) + High Video CFG (4.0 – 4.5):
    Cuts the audio leash. The visual tokens are driven by the text prompt
    without gradient interference from the audio track.
    Result: richer micro-contrast, sharper mechanical/skin
    details, but slight drift from the source clip's exact framing.

The Reason Behind Oversaturation: DiffVAE vs. Conv VAE

  - Convolutional VAE (conv-bf16): Static 3D convolution filters that hard-clip
    at pixel boundaries (clamp(0, 1)). When latent variance runs high (from CFG
    or guidance), channels clip at maximum luminance, creating neon,
    oversaturated, "deep-fried" colors. It also creates temporal
    smearing/ghosting on high-speed motion.
  - Diffusion Decoder (video-vae-bf16 / DiffVAE): Uses Neighborhood Attention
    generative reconstruction. It intelligently acts as a learned dynamic range
    tone-mapper. Even with high CFG (4.0+) or bright lighting, it maps energy
    into natural specular highlights and preserves midtones without color
    burning. It also tracks temporal trajectories cleanly across fast motion.
  - Rule: Use DiffVAE for action/motion scenes and high CFG headroom. Use Conv
    VAE only on static/low-motion scenes where you want instant ~2-second
    decoding.

7. Spectrum Acceleration: Quick Reference & Cadence Rules

Spectrum accelerates Stage 1 by forecasting a fraction of transformer
evaluations instead of running them, using a Chebyshev polynomial fit blended
with local linear (Taylor) extrapolation. There is no manual pass-counting
input anymore — the node reads the actual sigma schedule and sampler object
and detects everything automatically.

What auto-detection does (no configuration required)

  - Sampler stochasticity is measured during warmup. Deterministic samplers
    (Euler, res_multistep, SEEDS-2 with eta=0, …) forecast in feature space —
    the paper's method: the last transformer block's hidden state is cached,
    forecast, and re-passed through LTX's own output head at the current
    timestep. Noise-injecting samplers (ancestral, SDE, LCM, SEEDS-2 with
    eta>0, …) forecast in denoised (x0) space instead, rebuilding velocity from
    the current noisy latent so the sampler's injected noise is respected
    rather than skipped over. You do not need to know which category your
    sampler falls into; the node logs its finding on every run, e.g.:
    sampler 'sample_euler' is deterministic (measured noise ratio 0.010) ->
    forecasting in features space
  - Audio is always forecast in denoised space regardless of the video mode,
    because raw audio features do not extrapolate as smoothly as video
    features.
  - Streams (cond/uncond/STG branches) are identified by their actual identity
    (uuid, cond_or_uncond flag, STG/skip-block settings, tensor shape), not by
    call order — this is what makes it safe to use with MultimodalGuider and
    two-stage samplers (res_2s, Heun) without any manual multiplier.
  - Redundant trailing zero-sigma steps are stripped automatically. If your
    scheduler's sigma list ends ..., 0.0, 0.0 (e.g. LTXVScheduler with
    stretch=true and terminal=0), that final step is a wasted model call and,
    with Euler, a 0/0 NaN. The node removes it and logs how many steps it
    dropped. This is a Spectrum-node-side fix and only applies while Spectrum
    is enabled on that model — if you see this NaN without Spectrum, set
    terminal to a small positive value (e.g. 0.1) on LTXVScheduler directly.
    This is unrelated to the latent-connected token-shift underflow bug
    described in Section 2 — that one comes from an extreme shift value on
    long clips, this one comes from a literal repeated zero at the tail.

Node inputs

  - preset: quality / balanced (default) / fast — controls how many forecasts
    are allowed in a row. quality always runs a real step after each forecast
    (closest to unaccelerated output). balanced allows up to 2 forecasts in a
    row in the later half of the run. fast allows more (2 early / 3 late) —
    noticeably more drift, use only after validating.
  - warmup_steps: 5 (default) — initial steps that always run the real model.
    This window is also what the sampler-noise detector uses, so don't set it
    below ~3.
  - tail_actual_steps: 2 (default) — final steps that always run the real
    model, to restore micro-detail before decode.
  - degree: 3 (default) — Chebyshev polynomial degree. Lower is more stable but
    less accurate on longer skips; higher can overshoot with sparse history.
  - ridge_lambda: 0.10 — ridge regularization; the intercept term is left
    unpenalized so forecast weights always sum to exactly 1 (no amplitude
    drift toward zero, which earlier versions suffered from).
  - blend_weight: 0.50 — share of the forecast taken from the Chebyshev fit vs.
    a 2-point linear extrapolation.
  - max_history: 8 — anchors retained per stream.
  - forecast_space: auto (default) — leave on auto; it already selects
    features or denoised correctly per sampler. Only override for debugging.
  - history_storage: system_ram (default) — saves VRAM.
  - validate: false — when true, also runs the real model on forecast steps
    and logs relative error vs. the forecast, separately for video and audio.
    No speedup while enabled; use it once when trying a new preset or a
    demanding prompt (fast motion, on-screen text) before trusting the result.
  - debug: false — logs ACTUAL/FORECAST per step.

Measured speedup (model evaluations, not wall-clock — text encode, VAE decode,
and Stage 2 are unaffected)

| Preset     | Real evals (20 steps) | Forecast evals | Speedup   |
| :--------- | :--------------------: | :-------------: | :-------: |
| quality    | 13                      | 6–7              | ~1.54×    |
| balanced   | 12                      | 7–8              | ~1.67×    |
| fast       | 10                      | 9–10             | ~2.0×     |

At 30 steps, fast reaches roughly 2.3×. Longer schedules benefit more because
warmup/tail are a fixed cost.

Honesty note: forecast steps are not identical to real ones. On quality the
difference is usually not visible; pushing toward fast will eventually show up
as artifacts, most often on fast motion and on-screen text. Always validate a
new preset/prompt combination once before trusting it for a final render.

Using Spectrum on short / distilled (8-step) schedules

General recommendation: don't. An 8-step distilled run is mostly warmup+tail by
construction. The node will print inactive for this run: warmup (N) + tail (N)
leave nothing to forecast in 8 steps and simply run natively — this is the
correct, safe outcome, not a bug. Don't force it by dropping warmup/tail to 1;
2–3 middle steps isn't enough history to fit degree=3 reliably and you'll get
stuttering motion for a marginal gain. If you insist, use preset=quality,
degree=1, warmup_steps=2, tail_actual_steps=1, blend_weight=0.3, and validate
first.

8. Summary Troubleshooting Table

| Error / Symptom                                                      | Root Cause                                                                                                                                                                   | Immediate Fix                                                                                                                 |
| :------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------- |
| **`NaN` / Pitch Black output mid-sampling**                          | `LTXVScheduler` has `latent` connected on a clip with $> 4,096$ tokens, causing $\Delta \sigma < 0.0001$ (`bfloat16` underflow $\implies dt=0 \implies \text{div by zero}$). | Disconnect `latent` input on `LTXVScheduler`. Set `terminal = 0.1` (a hard `0.0` triggers a *different* NaN bug — see below). |
| **`NaN` on Euler specifically, AAC/audio mux error at save step**    | `LTXVScheduler` has `terminal = 0.0` **and** `stretch = true`, producing a sigma schedule ending `..., 0.0, 0.0`. Euler computes `d = (x - denoised)/sigma`, i.e. `0/0` on that final step. | Set `terminal` to a small positive value (e.g. `0.1`). If Spectrum is enabled on that model, it also auto-strips the duplicate zero step and logs `removed N redundant zero-sigma step(s)` — but fix the scheduler regardless, since this also affects non-Spectrum runs. |
| **`AttributeError: 'NestedTensor' object has no attribute 'clone'`** | Output of `SamplerCustomAdvanced` (a `NestedTensor`) is connected directly to `LTXVCropGuides`.                                                                              | Route sampler output through `LTXVSeparateAVLatent` first. Connect unpacked `video_latent` to crop node.                      |
| **`IndexError: tuple index out of range` on `latents[1]`**           | A pure video latent (audio disconnected) is fed into `LTXVSeparateAVLatent`.                                                                                                 | Bypass/remove `LTXVSeparateAVLatent`. Connect sampler output directly to next node.                                           |
| **`ValueError: not enough values to unpack (expected 2, got 1)`**    | A pure video latent is fed into `MultimodalGuider` (which hard-requires `[vx, ax]`).                                                                                         | Keep `LTXVConcatAVLatent` connected before the sampler, or replace `MultimodalGuider` with standard `CFGGuider`.              |
| **Console prints `inactive for this run: ...` and nothing is skipped** | Normal, not an error. Warmup + tail consumed the whole schedule (typical on short/distilled runs), or the schedule has fewer than 3 steps.                                   | Expected behavior — output equals the node being disabled. Increase step count, or lower `warmup_steps`/`tail_actual_steps` if you have headroom to spare. |
| **Blown highlights, neon edges, color clipping**                     | Using `conv-bf16` VAE with high CFG, causing the static convolution filters to hard-clamp outside RGB $[0, 1]$.                                                              | Switch `video_vae` to `ltx-2.5-video-vae-bf16.safetensors` (`DiffVAE`), which uses generative tone-mapping.                   |