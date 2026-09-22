"""
Pure-NumPy Box Embeddings
==========================

A from-scratch reimplementation of the box-embedding model used for the
specificity-as-containment result, WITHOUT a deep learning framework. The
model is simple enough (per-rule center + half-width, hinge losses) that
manual gradients are a handful of lines and this avoids any torch/CUDA
dependency entirely -- useful if your environment has GPU/CUDA library
issues, and more portable to run anywhere.

Box parameterization: center c in R^d, raw width parameter rho in R^d,
actual half-width w = softplus(rho) + floor. Bounds = [c - w, c + w].
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BoxBankNP:
    def __init__(self, rule_ids: List[str], dim: int, width_floor: float = 0.05,
                seed: int = 0):
        self.rule_ids = list(rule_ids)
        self.index = {rid: i for i, rid in enumerate(rule_ids)}
        n = len(rule_ids)
        self.dim = dim
        self.width_floor = width_floor
        rng = np.random.default_rng(seed)
        self.center = rng.normal(0, 1.0, size=(n, dim))
        self.rho = np.full((n, dim), 0.4)
        # Adam state
        self._m = {k: np.zeros_like(getattr(self, k)) for k in ("center", "rho")}
        self._v = {k: np.zeros_like(getattr(self, k)) for k in ("center", "rho")}
        self._t = 0

    def bounds(self, rid: str) -> Tuple[np.ndarray, np.ndarray]:
        i = self.index[rid]
        w = softplus(self.rho[i]) + self.width_floor
        c = self.center[i]
        return c - w, c + w

    def log_volume(self, rid: str) -> float:
        lo, hi = self.bounds(rid)
        return float(np.sum(np.log(hi - lo + 1e-9)))

    def all_log_volumes(self) -> Dict[str, float]:
        return {rid: self.log_volume(rid) for rid in self.rule_ids}

    def contains_point(self, rid: str, point: np.ndarray) -> bool:
        lo, hi = self.bounds(rid)
        return bool(np.all(point >= lo) and np.all(point <= hi))

    def intersection_log_volume(self, id_a: str, id_b: str) -> float:
        lo_a, hi_a = self.bounds(id_a)
        lo_b, hi_b = self.bounds(id_b)
        lo, hi = np.maximum(lo_a, lo_b), np.minimum(hi_a, hi_b)
        w = hi - lo
        if np.any(w <= 0):
            return float("-inf")
        return float(np.sum(np.log(w + 1e-9)))

    # -- training: manual gradients, Adam update ------------------------
    def _adam_step(self, grads: Dict[str, np.ndarray], lr: float = 0.05,
                   b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self._t += 1
        for k, g in grads.items():
            self._m[k] = b1 * self._m[k] + (1 - b1) * g
            self._v[k] = b2 * self._v[k] + (1 - b2) * (g ** 2)
            mhat = self._m[k] / (1 - b1 ** self._t)
            vhat = self._v[k] / (1 - b2 ** self._t)
            setattr(self, k, getattr(self, k) - lr * mhat / (np.sqrt(vhat) + eps))

    def train_step(self, fires: List[Tuple[str, np.ndarray]],
                  override_pairs: List[Tuple[str, str]],
                  lr: float = 0.05, containment_weight: float = 5.0,
                  volume_floor_log: float = -8.0) -> float:
        """One gradient step over: point-containment hinge (per fire),
        containment-margin hinge (per override/general pair), and a
        volume-floor hinge to discourage collapse. Returns total loss."""
        grad_c = np.zeros_like(self.center)
        grad_rho = np.zeros_like(self.rho)
        total_loss = 0.0
        n_fires = max(1, len(fires))

        for rid, point in fires:
            i = self.index[rid]
            w = softplus(self.rho[i]) + self.width_floor
            c = self.center[i]
            lo, hi = c - w, c + w
            below = np.maximum(lo - point, 0.0)   # point below lower bound
            above = np.maximum(point - hi, 0.0)   # point above upper bound
            loss = np.sum(below + above) / n_fires
            total_loss += loss
            # d(below)/dc = -1 where below>0 else 0 ; d(below)/dw = +1 where below>0
            # d(above)/dc = +1 where above>0 else 0 ; d(above)/dw = +1 where above>0
            dbelow = (below > 0).astype(float)
            dabove = (above > 0).astype(float)
            dc = (-dbelow + dabove) / n_fires
            dw = (dbelow + dabove) / n_fires
            grad_c[i] += dc
            grad_rho[i] += dw * softplus_grad(self.rho[i])

        for spec_id, gen_id in override_pairs:
            si, gi = self.index[spec_id], self.index[gen_id]
            w_s = softplus(self.rho[si]) + self.width_floor
            w_g = softplus(self.rho[gi]) + self.width_floor
            c_s, c_g = self.center[si], self.center[gi]
            lo_s, hi_s = c_s - w_s, c_s + w_s
            lo_g, hi_g = c_g - w_g, c_g + w_g
            # want lo_s >= lo_g  and  hi_s <= hi_g  (spec inside gen)
            lo_violation = np.maximum(lo_g - lo_s, 0.0)   # >0 means violated
            hi_violation = np.maximum(hi_s - hi_g, 0.0)
            loss = containment_weight * np.sum(lo_violation + hi_violation)
            total_loss += loss
            dlo_v = (lo_violation > 0).astype(float)
            dhi_v = (hi_violation > 0).astype(float)
            # d(lo_violation)/d(lo_s) = -1 ; d(lo_violation)/d(lo_g) = +1
            # d(hi_violation)/d(hi_s) = +1 ; d(hi_violation)/d(hi_g) = -1
            grad_c[si] += containment_weight * (-dlo_v + dhi_v)
            grad_rho[si] += containment_weight * (dlo_v + dhi_v) * softplus_grad(self.rho[si])
            grad_c[gi] += containment_weight * (dlo_v - dhi_v)
            grad_rho[gi] += containment_weight * -(dlo_v + dhi_v) * softplus_grad(self.rho[gi])

        # volume floor: discourage any box's log-volume from falling below a floor
        for i in range(len(self.rule_ids)):
            w = softplus(self.rho[i]) + self.width_floor
            logvol = np.sum(np.log(2 * w + 1e-9))
            if logvol < volume_floor_log:
                total_loss += (volume_floor_log - logvol) * 0.1
                grad_rho[i] += -0.1 * (2.0 / (2 * w + 1e-9)) * softplus_grad(self.rho[i])

        self._adam_step({"center": grad_c, "rho": grad_rho}, lr=lr)
        return total_loss
