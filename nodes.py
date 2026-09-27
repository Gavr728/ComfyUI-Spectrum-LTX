"""
ComfyUI-Spectrum-LTX
Spectrum (arXiv 2603.01623) for native ComfyUI LTXV / LTXAV (LTX-2.x) models.
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


Video forecast space is chosen automatically per run:
  * deterministic sampler -> feature space (final transformer block output, native
    output head re-run with the CURRENT timestep)                       [paper/H3]
  * stochastic sampler    -> denoised (x0) space, velocity rebuilt from the
    CURRENT noisy input so the sampler's injected noise is respected.
Audio (LTXAV) is always forecast in denoised space.

Step tracking is driven by the sampler's own per-step callback (authoritative);
sigma is only used to place sub-stages of multi-stage samplers inside a step.

Also removes redundant trailing zero-sigma steps (sigma 0 -> 0), which are a
no-op for correct samplers, waste one full model evaluation, and make Euler
produce NaN (0/0).
"""
import math
import logging
import statistics

import torch
import comfy.patcher_extension

log = logging.getLogger("SpectrumLTX")
KEY = "spectrum_ltx"
TAG = "[Spectrum LTX]"

_REQUIRED = ("_process_input", "_prepare_timestep", "_process_output", "transformer_blocks")
_STREAM_OPTS = ("stg_self_attn_blocks", "stg_skip_self_attn", "ptb_index", "stg_indexes",
                "skip_layers", "run_vx", "run_ax", "a2v_cross_attn", "v2a_cross_attn")
_ALWAYS_NOISY = ("lcm", "ddpm", "restart")
_ETA_NOISY = ("ancestral", "sde", "seeds", "sa_solver")
NOISE_THRESHOLD = 0.05

# max consecutive forecasts: (first half of forecastable range, second half)
PRESETS = {"quality": (1, 1), "balanced": (1, 2), "fast": (2, 3)}


def _say(msg):
    print(f"{TAG} {msg}", flush=True)


def _to_float(v, default):
    try:
        return float(v)
    except Exception:
        return default


def _strip_zero_tail(sigmas):
    """Drop trailing duplicate zeros: [..., 0, 0] -> [..., 0]."""
    if not torch.is_tensor(sigmas) or sigmas.ndim != 1:
        return sigmas, 0
    k = len(sigmas)
    while k > 2 and float(sigmas[k - 1]) == 0.0 and float(sigmas[k - 2]) == 0.0:
        k -= 1
    removed = len(sigmas) - k
    return (sigmas[:k] if removed else sigmas), removed


# ============================================================ forecaster
def _cheb(t, M):
    cols = [torch.ones_like(t)]
    if M >= 1:
        cols.append(t)
    for _ in range(2, M + 1):
        cols.append(2.0 * t * cols[-1] - cols[-2])
    return torch.stack(cols, dim=-1)


def _linear_weights(taus, tau_star):
    """2-point extrapolation from an anchor pair spaced at least as far apart as
    the extrapolation gap (avoids blow-up across 2-stage sub-steps)."""
    K = len(taus)
    w = torch.zeros(K, dtype=torch.float64)
    gap = tau_star - taus[-1]
    if K < 2 or gap <= 1e-12:
        w[-1] = 1.0
        return w
    j = 0
    for i in range(K - 2, -1, -1):
        if taus[-1] - taus[i] >= gap - 1e-9:
            j = i
            break
    d = taus[-1] - taus[j]
    if d <= 1e-9:
        w[-1] = 1.0
        return w
    r = min(gap / d, 1.0)
    w[-1] += 1.0 + r
    w[j] -= r
    return w


def spectrum_weights(taus, tau_star, degree, lam, blend):
    """w with H_hat(tau*) = sum_k w_k H_k. Chebyshev ridge regression with an
    unpenalised intercept, so sum(w) == 1 exactly (no shrinkage toward zero)."""
    K = len(taus)
    t = torch.tensor(taus, dtype=torch.float64)
    M = max(0, min(int(degree), K - 1))
    Phi = _cheb(t, M)
    phi = _cheb(torch.tensor([tau_star], dtype=torch.float64), M)[0]
    R = lam * torch.eye(M + 1, dtype=torch.float64)
    R[0, 0] = 0.0
    w_spec = Phi @ torch.linalg.solve(Phi.T @ Phi + R, phi)
    w = blend * w_spec + (1.0 - blend) * _linear_weights(taus, tau_star)
    return w.tolist()


# ============================================================ helpers
def _parts(obj):
    if torch.is_tensor(obj):
        return [obj], False
    if isinstance(obj, (list, tuple)):
        return list(obj), True
    t = getattr(obj, "tensors", None)
    if isinstance(t, (list, tuple)):
        return list(t), True
    return None, False


def _finite(parts):
    return all(bool(torch.isfinite(p).all()) for p in parts if torch.is_tensor(p))


def _sig_view(sig, ref):
    s = sig.detach().to(ref.device, torch.float32).flatten()
    B = ref.shape[0]
    if s.numel() != B:
        s = s.repeat(B // s.numel()) if B % s.numel() == 0 else s[:1].expand(B)
    return s.view(B, *([1] * (ref.ndim - 1)))


def _sampler_prior(sampler):
    fn = getattr(sampler, "sampler_function", None)
    name = (getattr(fn, "__name__", None)
            or getattr(getattr(fn, "func", None), "__name__", None)
            or type(sampler).__name__ or "unknown")
    name = str(name).lower()
    opts = {}
    kw = getattr(fn, "keywords", None)
    if isinstance(kw, dict):
        opts.update(kw)
    opts.update(getattr(sampler, "extra_options", None) or {})
    eta = _to_float(opts.get("eta", 1.0), 1.0)
    s_noise = _to_float(opts.get("s_noise", 1.0), 1.0)
    if any(k in name for k in _ALWAYS_NOISY):
        noisy = True
    elif any(k in name for k in _ETA_NOISY):
        noisy = eta > 0.0 and s_noise > 0.0
    else:
        noisy = False
    return name, noisy


# ============================================================ noise detector
class NoiseDetector:
    """Fraction of the sampler-input update not explained by the span of recent
    sampler inputs and denoised outputs: ~0 for deterministic samplers, large
    when fresh noise is injected."""
    DEPTH = 4

    def __init__(self):
        self.xs, self.ds, self.scores = [], [], []

    @staticmethod
    def _flat(obj):
        parts, _ = _parts(obj)
        if not parts:
            return None
        return torch.cat([p.detach().reshape(-1).float() for p in parts])

    def observe(self, x, d):
        with torch.no_grad():
            y, dd = self._flat(x), self._flat(d)
            if y is None or dd is None or y.numel() != dd.numel():
                return
            if self.xs and self.xs[-1].numel() != y.numel():
                self.release()
            if self.xs:
                step = torch.linalg.vector_norm(y - self.xs[-1])
                if step > 1e-8 * torch.linalg.vector_norm(y):
                    Q = []
                    for v in self.xs[-self.DEPTH:] + self.ds[-self.DEPTH:]:
                        w = v.clone()
                        for _ in range(2):
                            for q in Q:
                                w -= torch.dot(q, w) * q
                        n = torch.linalg.vector_norm(w)
                        if n > 1e-6 * torch.linalg.vector_norm(v):
                            Q.append(w / n)
                    r = y.clone()
                    for _ in range(2):
                        for q in Q:
                            r -= torch.dot(q, r) * q
                    self.scores.append(float(torch.linalg.vector_norm(r) / step))
            self.xs = (self.xs + [y.clone()])[-self.DEPTH:]
            self.ds = (self.ds + [dd.clone()])[-self.DEPTH:]

    def verdict(self):
        if len(self.scores) < 2:
            return None, None
        m = statistics.median(self.scores)
        return m > NOISE_THRESHOLD, m

    def release(self):
        self.xs.clear()
        self.ds.clear()


# ============================================================ state
class SpectrumConfig:
    def __init__(self, **kw):
        self.__dict__.update(kw)
        self.run = None


class RunState:
    def __init__(self, cfg, sigmas, sampler):
        self.cfg = cfg
        self.sig = sigmas.detach().float().cpu().flatten() if sigmas is not None else torch.zeros(1)
        self.n = len(self.sig) - 1
        s = self.sig
        self.sched_ok = bool(torch.isfinite(s).all()) and (self.n < 1 or bool((s[1:] <= s[:-1]).all()))
        self.sampler_name, self.prior_noisy = _sampler_prior(sampler)
        self.mode = None if cfg.forecast_space == "auto" else cfg.forecast_space
        self.detector = NoiseDetector() if self.mode is None else None
        self.min_anchors = max(2, cfg.degree + 1)
        self.caps = PRESETS.get(cfg.preset, PRESETS["balanced"])
        self.feat = {}          # stream -> [(tau, [hidden parts])]
        self.den = {}           # stream -> [(tau, [x0 parts])]
        self.out_meta = None    # (is_list, dtypes) of native diffusion output
        self.decisions = {}
        self.consec = 0
        self.cb_step = 0        # completed outer steps, from the sampler callback
        self.cb_seen = False
        self.warned_sigma = False
        self.n_actual = self.n_forecast = self.n_fallback = 0

        self.disabled_reason = None
        if self.n < 3:
            self.disabled_reason = f"only {self.n} steps"
        elif cfg.warmup_steps >= self.n - cfg.tail_actual_steps:
            self.disabled_reason = (f"warmup ({cfg.warmup_steps}) + tail ({cfg.tail_actual_steps}) "
                                    f"leave nothing to forecast in {self.n} steps")
        self.enabled = self.disabled_reason is None

    def schedule_summary(self):
        s = [float(v) for v in self.sig]
        head = ", ".join(f"{v:.4f}" for v in s[:3])
        tail = ", ".join(f"{v:.4f}" for v in s[-3:])
        return f"[{head}, ..., {tail}] ({len(s)} values)"

    def locate(self, sigma):
        """Return (outer_step, tau). Step index is authoritative from the sampler
        callback; sigma only refines the position of sub-stages within a step."""
        s, n = self.sig, self.n
        sig_step = None
        if self.sched_ok and math.isfinite(sigma):
            if sigma >= float(s[0]):
                sig_step = 0
            else:
                for i in range(n):
                    if float(s[i + 1]) <= sigma <= float(s[i]):
                        sig_step = i
                        break
        step = min(max(self.cb_step, sig_step if sig_step is not None else 0), n - 1)

        frac = 0.0
        hi, lo = float(s[step]), float(s[step + 1])
        if math.isfinite(sigma) and lo <= sigma <= hi and hi > lo:
            frac = min((hi - sigma) / (hi - lo), 0.999)
        elif not self.warned_sigma and (not math.isfinite(sigma) or sigma > hi * 1.01 + 1e-4 or sigma < lo - 1e-4):
            self.warned_sigma = True
            _say(f"note: model sigma {sigma:.6g} is outside the scheduled bracket of step {step} "
                 f"[{lo:.6g}, {hi:.6g}]; using sampler step counter for timing. "
                 f"schedule={self.schedule_summary()}")
        return step, 2.0 * (step + frac) / n - 1.0

    def _lock_mode(self):
        noisy, score = self.detector.verdict() if self.detector else (None, None)
        if noisy is None:
            noisy, src = self.prior_noisy, "sampler settings"
        else:
            src = f"measured noise ratio {score:.3f}"
        self.mode = "denoised" if noisy else "features"
        _say(f"sampler '{self.sampler_name}' is {'stochastic' if noisy else 'deterministic'} ({src}) "
             f"-> video in {self.mode} space, audio in denoised space")
        if self.mode == "denoised":
            self.feat.clear()
        if self.detector:
            self.detector.release()

    def _ready(self):
        hist = self.feat if self.mode == "features" else self.den
        return bool(hist) and max(len(h) for h in hist.values()) >= self.min_anchors

    def decide(self, step, sigma):
        if step in self.decisions:
            return self.decisions[step]
        c = self.cfg
        lo, hi = c.warmup_steps, self.n - c.tail_actual_steps
        f = False
        if lo <= step < hi:
            if self.mode is None:
                self._lock_mode()
            if self._ready():
                progress = (step - lo) / max(hi - lo, 1)
                cap = self.caps[0] if progress < 0.5 else self.caps[1]
                if self.consec < cap:
                    self.consec += 1
                    f = True
                else:
                    self.consec = 0
            else:
                self.consec = 0
        else:
            self.consec = 0
        self.decisions[step] = f
        if f:
            self.n_forecast += 1
        else:
            self.n_actual += 1
        if c.debug:
            _say(f"step {step:02d}/{self.n} sigma={sigma:.5f} -> {'FORECAST' if f else 'ACTUAL'}"
                 + (f" [{self.mode}]" if self.mode else ""))
        return f

    def stream_keys(self, to, xparts):
        B = xparts[0].shape[0]
        cou = list(to.get("cond_or_uncond") or [])
        uu = [str(u) for u in (to.get("uuids") or [])]
        extra = tuple(repr(to.get(k)) for k in _STREAM_OPTS)
        shapes = tuple(tuple(p.shape[1:]) for p in xparts)
        if len(cou) > 1 and B % len(cou) == 0:
            bs = B // len(cou)
            keys = [(uu[j] if j < len(uu) else None, cou[j], extra, shapes) for j in range(len(cou))]
        else:
            bs = B
            keys = [(tuple(uu), tuple(cou), extra, shapes)]
        return keys, bs

    def push(self, hist, keys, bs, tau, parts, as_float):
        for j, k in enumerate(keys):
            snap = []
            for p in parts:
                t = p[j * bs:(j + 1) * bs].detach()
                if as_float:
                    t = t.float()
                t = t.to("cpu", copy=True) if self.cfg.history_storage == "system_ram" else t.clone()
                snap.append(t)
            H = hist.setdefault(k, [])
            if H and abs(H[-1][0] - tau) < 1e-9:
                H[-1] = (tau, snap)
            else:
                H.append((tau, snap))
            while len(H) > self.cfg.max_history:
                H.pop(0)

    def forecast(self, hist, keys, tau, device):
        c = self.cfg
        chunks = []
        for k in keys:
            H = hist.get(k)
            if not H or len(H) < self.min_anchors:
                return None
            w = spectrum_weights([h[0] for h in H], tau, c.degree, c.ridge_lambda, c.blend_weight)
            out = []
            for i in range(len(H[-1][1])):
                acc = torch.zeros(H[-1][1][i].shape, dtype=torch.float32, device=device)
                for wk, (_, snap) in zip(w, H):
                    acc.add_(snap[i].to(device, non_blocking=True).float(), alpha=float(wk))
                if not torch.isfinite(acc).all():
                    raise RuntimeError("non-finite forecast")
                out.append(acc)
            chunks.append(out)
        if any(len(ch) != len(chunks[0]) for ch in chunks):
            raise RuntimeError("inconsistent stream structure")
        if len(chunks) == 1:
            return chunks[0]
        return [torch.cat([ch[i] for ch in chunks], dim=0) for i in range(len(chunks[0]))]

    def release(self):
        self.feat.clear()
        self.den.clear()
        if self.detector:
            self.detector.release()


# ============================================================ model execution
def _run_native(executor, dm, args, kwargs, capture):
    cap = {}
    handle = None
    if capture:
        handle = dm.transformer_blocks[-1].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o))
    try:
        out = executor(*args, **kwargs)
    finally:
        if handle is not None:
            handle.remove()
    return out, cap.get("h")


def _x0_parts(xparts, out, sig):
    oparts, _ = _parts(out)
    if oparts is None or len(oparts) > len(xparts):
        return None
    res = []
    for xp, op in zip(xparts, oparts):
        if xp.shape != op.shape:
            return None
        res.append(xp.float() - _sig_view(sig, xp) * op.float())
    return res


def _denoised_velocity(run, keys, tau, xparts, sig):
    if run.out_meta is None:
        return None
    _, dtypes = run.out_meta
    x0 = run.forecast(run.den, keys, tau, xparts[0].device)
    if x0 is None or len(x0) != len(dtypes):
        return None
    v = []
    for i, x0i in enumerate(x0):
        xp = xparts[i]
        s = _sig_view(sig, xp).clamp_min(1e-6)
        v.append(((xp.float() - x0i) / s).to(dtypes[i]))
    return v


def _feature_output(dm, run, keys, tau, args, kwargs):
    x, timestep, to, kf = args[0], args[1], args[5], args[6]
    kw = dict(kwargs)
    denoise_mask = kw.pop("denoise_mask", None)
    x_first = x[0] if isinstance(x, (list, tuple)) else x
    merged = {**to, **kw}
    xh, _coords, add = dm._process_input(x, kf, denoise_mask, **merged)
    merged.update(add)
    _ts, emb_ts, prompt_ts = dm._prepare_timestep(timestep, x_first.shape[0], x_first.dtype, **merged)
    merged["prompt_timestep"] = prompt_ts

    ref, is_list = _parts(xh)
    pred = run.forecast(run.feat, keys, tau, ref[0].device)
    if pred is None:
        return None
    if len(pred) != len(ref) or any(p.shape != r.shape for p, r in zip(pred, ref)):
        raise RuntimeError("hidden-state shape changed vs history")
    hidden = [p.to(r.dtype) for p, r in zip(pred, ref)]
    return dm._process_output(hidden if is_list else hidden[0], emb_ts, kf, **merged)


def _forecast_output(dm, run, keys, tau, args, kwargs, xparts, sig):
    if run.mode == "features":
        fout = _feature_output(dm, run, keys, tau, args, kwargs)
        if fout is None:
            return None
        oparts, is_list = _parts(fout)
        if len(oparts) > 1:                                   # LTXAV: audio from denoised space
            dv = _denoised_velocity(run, keys, tau, xparts, sig)
            if dv is None or len(dv) != len(oparts):
                return None
            oparts = [oparts[0]] + dv[1:]
        return oparts if is_list else oparts[0]
    dv = _denoised_velocity(run, keys, tau, xparts, sig)
    if dv is None:
        return None
    return dv if run.out_meta[0] else dv[0]


# ============================================================ wrappers
def spectrum_diffusion_wrapper(executor, *args, **kwargs):
    # LTXBaseModel.forward -> execute(x, timestep, context, attention_mask,
    #   frame_rate, transformer_options, keyframe_idxs, denoise_mask=..., **kw)
    to = args[5] if len(args) > 5 else None
    cfg = to.get(KEY) if isinstance(to, dict) else None
    run = getattr(cfg, "run", None)
    if run is None or not run.enabled or len(args) < 7:
        return executor(*args, **kwargs)
    sig = to.get("sigmas")
    xparts, _ = _parts(args[0])
    if sig is None or xparts is None:
        return executor(*args, **kwargs)

    dm = executor.class_obj
    sigma = float(sig.detach().flatten().max())
    step, tau = run.locate(sigma)
    keys, bs = run.stream_keys(to, xparts)

    if run.decide(step, sigma):
        try:
            out = _forecast_output(dm, run, keys, tau, args, kwargs, xparts, sig)
            if out is not None and _finite(_parts(out)[0]):
                if cfg.validate:
                    real_out, _ = _run_native(executor, dm, args, kwargs, False)
                    a = _x0_parts(xparts, out, sig)
                    b = _x0_parts(xparts, real_out, sig)
                    if a and b:
                        errs = [float((u - v).norm() / v.norm().clamp_min(1e-8)) for u, v in zip(a, b)]
                        names = ["video", "audio"] + [f"part{i}" for i in range(2, len(errs))]
                        _say(f"validate step {step:02d} [{run.mode}] x0 rel.err " +
                             ", ".join(f"{n}={e:.4f}" for n, e in zip(names, errs)))
                return out
            if out is not None:
                log.warning(f"{TAG} non-finite forecast at step {step}; running model.")
        except Exception as e:
            log.warning(f"{TAG} forecast failed at step {step} ({e}); running model.")
        run.n_fallback += 1

    want_feat = run.mode in (None, "features")
    out, hidden = _run_native(executor, dm, args, kwargs, want_feat)
    if want_feat and hidden is not None:
        run.push(run.feat, keys, bs, tau, _parts(hidden)[0], as_float=False)
    x0 = _x0_parts(xparts, out, sig)
    if x0 is not None:
        oparts, is_list = _parts(out)
        run.out_meta = (is_list, [o.dtype for o in oparts])
        run.push(run.den, keys, bs, tau, x0, as_float=True)
    return out


def spectrum_predict_noise_wrapper(executor, *args, **kwargs):
    out = executor(*args, **kwargs)
    try:
        cfg = executor.class_obj.model_options.get("transformer_options", {}).get(KEY)
        run = getattr(cfg, "run", None)
        if run is not None and run.enabled and run.mode is None and run.detector is not None:
            run.detector.observe(args[0] if args else kwargs.get("x"), out)
    except Exception as e:
        log.debug(f"{TAG} noise detector skipped: {e}")
    return out


def spectrum_sampler_sample_wrapper(executor, *args, **kwargs):
    # KSAMPLER.sample(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
    extra_args = args[2] if len(args) > 2 else kwargs.get("extra_args", {})
    cfg = (extra_args or {}).get("model_options", {}).get("transformer_options", {}).get(KEY)
    run = getattr(cfg, "run", None)
    if run is None or not run.enabled:
        return executor(*args, **kwargs)

    orig_cb = args[3] if len(args) > 3 else kwargs.get("callback")

    def step_callback(*cb_args, **cb_kwargs):
        run.cb_seen = True
        idx = None
        if cb_args and isinstance(cb_args[0], int):
            idx = cb_args[0]
        run.cb_step = max(run.cb_step, (idx + 1) if idx is not None else run.cb_step + 1)
        if orig_cb is not None:
            return orig_cb(*cb_args, **cb_kwargs)

    if len(args) > 3:
        args = args[:3] + (step_callback,) + args[4:]
    else:
        kwargs["callback"] = step_callback
    return executor(*args, **kwargs)


def spectrum_outer_sample_wrapper(executor, *args, **kwargs):
    # CFGGuider.outer_sample(noise, latent_image, sampler, sigmas, denoise_mask, ...)
    guider = executor.class_obj
    cfg = guider.model_options.get("transformer_options", {}).get(KEY)
    if not isinstance(cfg, SpectrumConfig):
        return executor(*args, **kwargs)

    sampler = args[2] if len(args) > 2 else kwargs.get("sampler")
    if len(args) > 3:
        sigmas, removed = _strip_zero_tail(args[3])
        if removed:
            args = args[:3] + (sigmas,) + args[4:]
    else:
        sigmas, removed = _strip_zero_tail(kwargs.get("sigmas"))
        if removed:
            kwargs["sigmas"] = sigmas
    if removed:
        _say(f"removed {removed} redundant zero-sigma step(s) (scheduler terminal=0): "
             f"saves {removed} model evaluation(s) and prevents Euler NaN")

    run = RunState(cfg, sigmas, sampler)
    if not run.enabled:
        _say(f"inactive for this run: {run.disabled_reason}")
        return executor(*args, **kwargs)

    _say(f"active: sampler={run.sampler_name}, steps={run.n}, preset={cfg.preset}, "
         f"warmup={cfg.warmup_steps}, tail={cfg.tail_actual_steps}, space={cfg.forecast_space}")
    if cfg.debug or not run.sched_ok:
        _say(f"schedule {run.schedule_summary()}" + ("" if run.sched_ok else
             "  WARNING: non-finite or non-monotonic sigmas; timing falls back to sampler step counter"))
    prev, cfg.run = cfg.run, run
    try:
        return executor(*args, **kwargs)
    finally:
        cfg.run = prev
        tot = run.n_actual + run.n_forecast
        base = tot + removed
        _say(f"done [{run.mode or 'no forecast'}]: actual={run.n_actual} forecast={run.n_forecast} "
             f"fallback_calls={run.n_fallback} (~{base / max(run.n_actual, 1):.2f}x fewer model steps "
             f"vs {base} unaccelerated)")
        if tot < max(run.n // 2, 2):
            _say(f"WARNING: step tracking saw only {tot} of {run.n} steps "
                 f"(sampler callback seen: {run.cb_seen}). Please rerun with debug=true and report the log.")
        run.release()


# ============================================================ node
class SpectrumApplyLTX:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "preset": (list(PRESETS.keys()), {"default": "balanced",
                "tooltip": "quality: real step after every forecast. balanced: up to 2 forecasts in a row "
                           "in the later half. fast: 2 early / 3 late (more drift)."}),
            "warmup_steps": ("INT", {"default": 5, "min": 1, "max": 100,
                "tooltip": "Initial steps always run on the model (composition, text layout, sampler detection)."}),
            "tail_actual_steps": ("INT", {"default": 2, "min": 0, "max": 20,
                "tooltip": "Final steps always run on the model."}),
            "degree": ("INT", {"default": 3, "min": 1, "max": 8}),
            "ridge_lambda": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 10.0, "step": 0.01}),
            "blend_weight": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "Chebyshev share; remainder is 2-point linear extrapolation."}),
            "max_history": ("INT", {"default": 8, "min": 2, "max": 32}),
            "forecast_space": (["auto", "features", "denoised"], {"default": "auto",
                "tooltip": "Video forecast space. auto: features for deterministic samplers, "
                           "denoised for noise-injecting ones. Audio always uses denoised."}),
            "history_storage": (["system_ram", "vram"], {"default": "system_ram"}),
            "validate": ("BOOLEAN", {"default": False,
                "tooltip": "Also run the model on forecast steps and print forecast error (no speed-up)."}),
            "debug": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "sampling/spectrum"

    def apply(self, model, enabled, preset, warmup_steps, tail_actual_steps, degree, ridge_lambda,
              blend_weight, max_history, forecast_space, history_storage, validate, debug):
        if not enabled:
            return (model,)
        dm = model.get_model_object("diffusion_model")
        missing = [a for a in _REQUIRED if not hasattr(dm, a)]
        if missing:
            raise ValueError(f"Spectrum LTX: not a native LTXV/LTXAV model (missing {missing}).")

        cfg = SpectrumConfig(
            preset=preset, warmup_steps=int(warmup_steps), tail_actual_steps=int(tail_actual_steps),
            degree=int(degree), ridge_lambda=float(ridge_lambda), blend_weight=float(blend_weight),
            max_history=max(int(max_history), int(degree) + 1),
            forecast_space=forecast_space, history_storage=history_storage,
            validate=bool(validate), debug=bool(debug),
        )
        m = model.clone()
        m.model_options["transformer_options"][KEY] = cfg
        W = comfy.patcher_extension.WrappersMP
        m.add_wrapper_with_key(W.OUTER_SAMPLE, KEY, spectrum_outer_sample_wrapper)
        m.add_wrapper_with_key(W.SAMPLER_SAMPLE, KEY, spectrum_sampler_sample_wrapper)
        m.add_wrapper_with_key(W.PREDICT_NOISE, KEY, spectrum_predict_noise_wrapper)
        m.add_wrapper_with_key(W.DIFFUSION_MODEL, KEY, spectrum_diffusion_wrapper)
        return (m,)


NODE_CLASS_MAPPINGS = {"SpectrumApplyLTX": SpectrumApplyLTX}
NODE_DISPLAY_NAME_MAPPINGS = {"SpectrumApplyLTX": "Spectrum Apply LTX"}