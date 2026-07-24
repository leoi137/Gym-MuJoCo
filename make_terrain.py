"""Generate the desert heightfield + ground texture for Spyder-Desert-v0.

Writes two files consumed by assets/spyder12_desert.xml:

    assets/terrain/desert_hfield.bin   -- MuJoCo custom binary heightfield
                                          (int32 nrow, int32 ncol, float32 data),
                                          full [0,1] span so MuJoCo's automatic
                                          min-max normalization is the identity
    assets/terrain/desert_texture.png  -- ground albedo baked from the SAME
                                          height data (slope->rock, height->sand
                                          tint, grain), stretched once across
                                          the heightfield by the material

Why procedural noise and not a "real" DEM: at spider scale (0.5 m legs, 80 x
80 m world) what makes ground read as natural is its statistics, not any
specific mountain. Measured landscapes have an approximately power-law height
spectrum, P(k) ~ k^-beta with beta ~= 2 -- roughness at every scale, more of
it at large scales. We synthesize exactly that in Fourier space (random
phases, k^-beta/2 amplitudes, inverse FFT), which -- unlike lattice/Perlin
noise -- has no grid direction baked in. Two transforms turn the smooth
fractal into desert landforms:

    ridging       h -> 1 - |h| creases the field along the wandering zero
                  contours: sharp crestlines and V-gullies, the signature of
                  wind-scoured dunes and eroded rock
    domain warp   sampling the field at noise-displaced coordinates bends
                  ridges into the curved, flow-like shapes of real relief

The map is composed in METERS relative to the spawn surface (h = 0 at the
origin pad), from four layers:

    dunes      band-passed (wavelength ~14-30 m) anisotropic field, ridged:
               wind-aligned crestlines, ~0.8 m crest-to-trough
    mountains  warped ridged fractal, gated by distance from spawn -- open
               basin out to ~10 m, rocky foothills, up to ~+3.3 m peaks near
               the map edge. Distance gating makes difficulty grow smoothly
               along the +x reward direction: the policy meets slopes it can
               survive before slopes it can't (a built-in curriculum).
    detail     high-passed fractal, ~ +-6 cm: gravel and dirt clods underfoot.
               This is the layer the feet actually feel on every step.
    spawn pad  cosine-blended flat disk (r = 2.5 m flat, 6 m blend) so
               reset_model's authored stance always starts on level ground.

The torso spawn height does NOT depend on constants here: SpyderEnv reads the
compiled heightfield and re-bases init_qpos z off the measured ground height
under the origin, so regenerating with a different seed never breaks resets.

Usage:
    python make_terrain.py            # deterministic (seed 7)
    python make_terrain.py --seed 12  # a different desert
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
from PIL import Image

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "terrain"

# Must match the <hfield> element in spyder12_desert.xml (see main()'s check).
HALF_EXTENT = 40.0   # meters; size="40 40 ..." -> 80 x 80 m world
GRID = 1024          # cells per side -> ~7.8 cm horizontal resolution
CELL = 2 * HALF_EXTENT / GRID


def _spectral_field(rng: np.random.Generator, n: int, beta: float,
                    stretch: tuple[float, float] = (1.0, 1.0),
                    band: tuple[float, float] | None = None) -> np.ndarray:
    """A random field with power spectrum P(k) ~ k^-beta, zero mean, unit std.

    stretch scales frequency per axis: stretch=(1, 3) makes features 3x longer
    along y than x. band=(lo_m, hi_m) keeps only wavelengths in that range
    (meters) -- a band-passed field oscillates at a characteristic wavelength,
    which is what gives dune fields their quasi-regular crest spacing.
    """
    kx = np.fft.fftfreq(n, d=CELL)[None, :] * stretch[0]
    ky = np.fft.fftfreq(n, d=CELL)[:, None] * stretch[1]
    k = np.hypot(kx, ky)
    k[0, 0] = np.inf                      # kill the DC component
    amp = k ** (-beta / 2.0)              # amplitude^2 = power ~ k^-beta
    if band is not None:
        lo_m, hi_m = band                 # wavelengths -> frequency window
        amp *= np.exp(-0.5 * ((np.log(np.maximum(k, 1e-9) *
                                      np.sqrt(lo_m * hi_m)) /
                               (0.5 * np.log(hi_m / lo_m))) ** 2))
    phase = rng.uniform(0.0, 2.0 * np.pi, (n, n))
    field = np.fft.ifft2(amp * np.exp(1j * phase)).real
    field -= field.mean()
    return field / field.std()


def _warp(field: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Resample `field` at coordinates displaced by (dx, dy) in meters --
    domain warping. Bilinear; the field is treated as periodic (it is: FFT
    synthesis is periodic by construction), so no edge seams."""
    n = field.shape[0]
    rows = (np.arange(n)[:, None] + dy / CELL) % n
    cols = (np.arange(n)[None, :] + dx / CELL) % n
    r0 = np.floor(rows).astype(int) % n
    c0 = np.floor(cols).astype(int) % n
    r1, c1 = (r0 + 1) % n, (c0 + 1) % n
    fr, fc = rows - np.floor(rows), cols - np.floor(cols)
    return ((field[r0, c0] * (1 - fc) + field[r0, c1] * fc) * (1 - fr)
            + (field[r1, c0] * (1 - fc) + field[r1, c1] * fc) * fr)


def build_height_m(seed: int) -> np.ndarray:
    """The composed terrain in meters, spawn surface = 0. Returns (GRID, GRID),
    row axis = y (south->north), col axis = x (west->east)."""
    rng = np.random.default_rng(seed)
    n = GRID

    coords = np.linspace(-HALF_EXTENT, HALF_EXTENT, n)
    x, y = coords[None, :], coords[:, None]
    dist = np.sqrt(x * x + y * y)

    # -- Dune field ----------------------------------------------------------
    # Band-passed anisotropic field; its zero contours are evenly spaced
    # wandering lines. 1-|f| turns them into crests; ^1.5 sharpens the crest
    # and rounds the interdune flat, the classic transverse-dune profile.
    f_dune = _spectral_field(rng, n, beta=2.0, stretch=(1.0, 0.35),
                             band=(14.0, 30.0))
    crest = np.clip(1.0 - np.abs(f_dune), 0.0, None) ** 1.5
    dunes = 0.8 * (crest - crest.mean())

    # -- Mountains -----------------------------------------------------------
    # Ridged fractal, domain-warped so ridgelines curve like eroded rock, and
    # a second, finer ridged field multiplied ON the ridges (roughness where
    # rock is exposed, not in the basins) -- a cheap multifractal. Every field
    # is band-limited: a raw k^-beta field in 2D has divergent gradient energy
    # at high k (grid-scale spikes -> 80 degree slopes), so each layer only
    # keeps the wavelengths it is meant to model.
    wx = 5.0 * _spectral_field(rng, n, beta=2.5, band=(20.0, 80.0))
    wy = 5.0 * _spectral_field(rng, n, beta=2.5, band=(20.0, 80.0))
    r1 = 1.0 - np.abs(_warp(_spectral_field(rng, n, beta=2.2, band=(14.0, 80.0)),
                            wx, wy))
    r2 = 1.0 - np.abs(_spectral_field(rng, n, beta=1.8, band=(3.0, 10.0)))
    ridged = np.clip(r1, 0.0, None) ** 1.8
    ridged *= 1.0 + 0.25 * r2
    # smoothstep gate: 0 inside r=10 m, 1 beyond r=32 m
    t = np.clip((dist - 10.0) / 22.0, 0.0, 1.0)
    gate = t * t * (3.0 - 2.0 * t)
    mountains = 3.3 * ridged * gate

    # -- Fine dirt / gravel roughness everywhere -----------------------------
    detail = 0.04 * _spectral_field(rng, n, beta=1.5, band=(0.5, 3.0))

    h = dunes + mountains + detail

    # -- Spawn pad: flatten a disk at the origin, cosine blend outward -------
    r_flat, r_blend = 2.5, 6.0
    w = np.clip((dist - r_flat) / (r_blend - r_flat), 0.0, 1.0)
    h *= 0.5 - 0.5 * np.cos(np.pi * w)   # 0 at center -> 1 outside

    return h


def save_hfield_bin(h_m: np.ndarray, path: Path) -> tuple[float, float]:
    """Normalize to an exact [0,1] span (so MuJoCo's own min-max normalization
    changes nothing) and write the custom binary format."""
    h_min, h_max = float(h_m.min()), float(h_m.max())
    data = ((h_m - h_min) / (h_max - h_min)).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", *data.shape))
        f.write(data.tobytes())
    return h_min, h_max


def save_texture(h_m: np.ndarray, path: Path, seed: int) -> None:
    """Bake ground albedo from the height data: sand base, slope-driven rock,
    height-driven tint, grain, and baked-in sun shading (MuJoCo's per-vertex
    lighting on a 1024^2 hfield is too coarse to show 8 cm gravel; baking the
    shading into the albedo is what makes the relief visible)."""
    rng = np.random.default_rng(seed + 1)
    n = h_m.shape[0]

    gy, gx = np.gradient(h_m, CELL)
    slope = np.hypot(gx, gy)                              # rise/run
    rockiness = np.clip((slope - 0.25) / 0.45, 0.0, 1.0)  # rock past ~14 deg

    sand = np.array([0.85, 0.67, 0.43])
    sand_low = np.array([0.74, 0.58, 0.38])   # damper, darker basins
    rock = np.array([0.53, 0.46, 0.40])

    height01 = (h_m - h_m.min()) / (h_m.max() - h_m.min())
    base = sand_low[None, None] + (sand - sand_low)[None, None] * height01[..., None]
    color = base + (rock[None, None] - base) * rockiness[..., None]

    # Lambert shading from a low afternoon sun (matches the XML light dir).
    sun = np.array([-0.45, -0.25, 0.86])
    sun /= np.linalg.norm(sun)
    norm = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    norm /= np.linalg.norm(norm, axis=-1, keepdims=True)
    # Soft floor (0.55): the live scene light shades the same slopes again,
    # so a hard baked shadow would double-darken every south-west face.
    lambert = np.clip(norm @ sun, 0.0, 1.0)
    color *= (0.55 + 0.45 * lambert)[..., None]

    # Multiplicative grain: fine speckle + broad patches of windblown crust.
    grain = 0.92 + 0.08 * rng.random((n, n))
    patches = 0.94 + 0.12 * _spectral_field(rng, n, beta=2.0, band=(2.0, 8.0)) * 0.5
    color *= (grain * patches)[..., None]

    img = (np.clip(color, 0.0, 1.0) * 255).astype(np.uint8)
    # flipud: OpenGL puts texture t=0 at the image's LAST row, so without the
    # flip the albedo is mirrored north-south against the geometry (verified
    # with a marker render; the painted shading visibly fights the relief).
    Image.fromarray(np.flipud(img)).save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    h = build_height_m(args.seed)
    h_min, h_max = save_hfield_bin(h, ASSET_DIR / "desert_hfield.bin")
    save_texture(h, ASSET_DIR / "desert_texture.png", args.seed)

    span = h_max - h_min
    print(f"grid {GRID}x{GRID} over {2*HALF_EXTENT:.0f}x{2*HALF_EXTENT:.0f} m "
          f"({CELL*100:.1f} cm/cell)")
    print(f"elevation span {span:.2f} m (min {h_min:+.2f}, max {h_max:+.2f} "
          f"relative to spawn surface)")
    # MuJoCo rescales the normalized [0,1] data by the hfield z-size, so the
    # XML must carry THESE numbers for the world to be metrically correct:
    print(f"XML check: <hfield ... size=\"{HALF_EXTENT:.0f} {HALF_EXTENT:.0f} "
          f"{span:.2f} 1.0\"/> and hfield geom pos = \"0 0 {h_min:.2f}\"")


if __name__ == "__main__":
    main()
