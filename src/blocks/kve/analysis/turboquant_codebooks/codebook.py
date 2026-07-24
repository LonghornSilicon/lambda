"""Lloyd-Max optimal scalar quantizer for the rotated-unit-vector coordinate
distribution used by TurboQuant (Algorithm 1).

After a random orthogonal rotation of a unit vector in R^d, each coordinate
`u_i` has density

    f(t) = C_d * (1 - t^2) ** ((d - 3) / 2),   t in [-1, 1]

where `C_d = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))` is the normalizer.
This is the symmetric projection density of a uniform unit vector; equivalently
`u_i^2 ~ Beta(1/2, (d-1)/2)`.

We run classical Lloyd-Max on that density to produce, for a given bit-width
`b`, a codebook of `2^b` centroids and `2^b + 1` decision boundaries that
minimises expected squared error. Codebooks are cached as JSON under
``codebooks/codebook_d{d}_b{b}.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

_CODEBOOK_DIR = Path(__file__).parent / "codebooks"

_SHIPPED_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (64, 1), (64, 2), (64, 3), (64, 4),
    (128, 1), (128, 2), (128, 3), (128, 4),
    (576, 3),
)


@dataclass
class Codebook:
    d: int
    bits: int
    n_levels: int
    bounds: np.ndarray     # shape (n_levels + 1,), first = -1, last = +1
    centroids: np.ndarray  # shape (n_levels,)
    mse: float             # expected squared error under the Beta density

    def to_json(self) -> Dict:
        return {
            "d": int(self.d),
            "bits": int(self.bits),
            "n_levels": int(self.n_levels),
            "bounds": self.bounds.tolist(),
            "centroids": self.centroids.tolist(),
            "mse": float(self.mse),
        }

    @classmethod
    def from_json(cls, data: Dict) -> "Codebook":
        return cls(
            d=int(data["d"]),
            bits=int(data["bits"]),
            n_levels=int(data["n_levels"]),
            bounds=np.asarray(data["bounds"], dtype=np.float64),
            centroids=np.asarray(data["centroids"], dtype=np.float64),
            mse=float(data["mse"]),
        )


def _beta_grid(d: int, n_grid: int = 200001, eps: float = 1e-9):
    """Build a fine grid on [-1, 1] with the (normalised) Beta density and its
    cumulative integrals ``F = int_{-1}^t f`` and ``Ft = int_{-1}^t s f(s) ds``.
    """
    if d < 2:
        raise ValueError(f"Beta coordinate density requires d >= 2, got {d}")
    t = np.linspace(-1.0 + eps, 1.0 - eps, n_grid)
    if d == 2:
        log_f = np.zeros_like(t)
    else:
        log_f = 0.5 * (d - 3) * np.log1p(-t * t)
    log_f -= log_f.max()
    f = np.exp(log_f)
    dt = np.diff(t)
    trap = 0.5 * (f[1:] + f[:-1]) * dt
    area = trap.sum()
    f /= area
    trap = 0.5 * (f[1:] + f[:-1]) * dt
    F = np.concatenate([[0.0], np.cumsum(trap)])
    trap_t = 0.5 * (t[1:] * f[1:] + t[:-1] * f[:-1]) * dt
    Ft = np.concatenate([[0.0], np.cumsum(trap_t)])
    trap_t2 = 0.5 * (t[1:] ** 2 * f[1:] + t[:-1] ** 2 * f[:-1]) * dt
    Ft2 = np.concatenate([[0.0], np.cumsum(trap_t2)])
    return t, f, F, Ft, Ft2


def _interp(grid: np.ndarray, cumulative: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.interp(x, grid, cumulative)


def lloyd_max(
    d: int,
    bits: int,
    n_grid: int = 200001,
    max_iter: int = 500,
    tol: float = 1e-12,
    init_bounds: np.ndarray | None = None,
) -> Codebook:
    """Run Lloyd-Max on the Beta(1/2, (d-1)/2) coordinate density.

    Returns a ``Codebook`` with symmetric boundaries in [-1, 1] and centroids
    chosen to minimise MSE under the density.
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")
    n_levels = 1 << bits
    grid, _, F, Ft, Ft2 = _beta_grid(d, n_grid=n_grid)

    if init_bounds is None:
        bounds = np.linspace(-1.0, 1.0, n_levels + 1)
    else:
        bounds = np.asarray(init_bounds, dtype=np.float64).copy()

    centroids = 0.5 * (bounds[1:] + bounds[:-1])

    for _ in range(max_iter):
        mass = _interp(grid, F, bounds[1:]) - _interp(grid, F, bounds[:-1])
        moment = _interp(grid, Ft, bounds[1:]) - _interp(grid, Ft, bounds[:-1])
        safe = mass > 1e-18
        new_centroids = np.where(safe, moment / np.where(safe, mass, 1.0),
                                 0.5 * (bounds[1:] + bounds[:-1]))
        new_bounds = bounds.copy()
        new_bounds[1:-1] = 0.5 * (new_centroids[:-1] + new_centroids[1:])
        new_bounds[0] = -1.0
        new_bounds[-1] = 1.0
        delta = np.max(np.abs(new_bounds - bounds))
        bounds = new_bounds
        centroids = new_centroids
        if delta < tol:
            break

    mass = _interp(grid, F, bounds[1:]) - _interp(grid, F, bounds[:-1])
    moment = _interp(grid, Ft, bounds[1:]) - _interp(grid, Ft, bounds[:-1])
    second = _interp(grid, Ft2, bounds[1:]) - _interp(grid, Ft2, bounds[:-1])
    mse = float(np.sum(second - 2.0 * centroids * moment + centroids * centroids * mass))

    return Codebook(
        d=d,
        bits=bits,
        n_levels=n_levels,
        bounds=bounds,
        centroids=centroids,
        mse=mse,
    )


def load_or_build_codebook(
    d: int,
    bits: int,
    cache_dir: Path | str | None = None,
    rebuild: bool = False,
    **lloyd_kwargs,
) -> Codebook:
    """Return a cached codebook if present, otherwise run Lloyd-Max and save."""
    cache_dir = Path(cache_dir) if cache_dir is not None else _CODEBOOK_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"codebook_d{d}_b{bits}.json"
    if path.exists() and not rebuild:
        with open(path, "r") as fp:
            return Codebook.from_json(json.load(fp))

    init = None
    if bits > 1:
        prev_path = cache_dir / f"codebook_d{d}_b{bits - 1}.json"
        if prev_path.exists():
            with open(prev_path, "r") as fp:
                prev = Codebook.from_json(json.load(fp))
            # Warm start: split each prior interval at its centroid.
            new_bounds = [prev.bounds[0]]
            for k in range(prev.n_levels):
                new_bounds.append(prev.centroids[k])
                new_bounds.append(prev.bounds[k + 1])
            init = np.array(new_bounds)

    cb = lloyd_max(d, bits, init_bounds=init, **lloyd_kwargs)
    with open(path, "w") as fp:
        json.dump(cb.to_json(), fp, indent=2)
    return cb


def quantize_to_codebook(y: np.ndarray, cb: Codebook) -> np.ndarray:
    """Map continuous values in [-1, 1] to codebook indices in [0, n_levels).

    Uses the interior decision boundaries (``cb.bounds[1:-1]``) via
    ``np.searchsorted``.
    """
    interior = cb.bounds[1:-1]
    idx = np.searchsorted(interior, y, side="right")
    dtype = np.uint8 if cb.n_levels <= 256 else np.int32
    return idx.astype(dtype)


def dequantize_from_codebook(idx: np.ndarray, cb: Codebook) -> np.ndarray:
    return cb.centroids[idx.astype(np.int64)]


def build_shipped_codebooks(
    configs: Iterable[Tuple[int, int]] = _SHIPPED_CONFIGS,
    rebuild: bool = False,
    verbose: bool = True,
) -> List[Codebook]:
    """Pre-build the codebooks shipped with the package."""
    out: List[Codebook] = []
    for (d, b) in configs:
        cb = load_or_build_codebook(d, b, rebuild=rebuild)
        out.append(cb)
        if verbose:
            print(f"  d={d:4d} b={b} n_levels={cb.n_levels:4d} mse={cb.mse:.6e}")
    return out


def _main():
    import argparse

    parser = argparse.ArgumentParser(description="Build TurboQuant Lloyd-Max codebooks")
    parser.add_argument("--rebuild", action="store_true",
                        help="recompute even if cached")
    parser.add_argument("--d", type=int, nargs="+", default=None,
                        help="dimensions to build (default: shipped set)")
    parser.add_argument("--bits", type=int, nargs="+", default=None,
                        help="bit widths to build")
    args = parser.parse_args()

    if args.d is None and args.bits is None:
        configs = _SHIPPED_CONFIGS
    else:
        ds = args.d if args.d else [64, 128]
        bs = args.bits if args.bits else [1, 2, 3, 4]
        configs = [(d, b) for d in ds for b in bs]

    print(f"Building {len(configs)} codebooks -> {_CODEBOOK_DIR}")
    build_shipped_codebooks(configs, rebuild=args.rebuild)


if __name__ == "__main__":
    _main()
