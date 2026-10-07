"""
=====================================================================
Q2  |  LAB 2 — Point / photometric transforms
     "Night-time CCTV Exposure Fixer"
=====================================================================
SCENARIO
  A night-time CCTV frame is badly underexposed (all pixels squeezed into
  the dark end of the range). Recover detail using point transforms only
  (every output pixel depends ONLY on the same input pixel: s = T(r)).

TASKS
  (a) Implement, as functions returning uint8:
        negative(g)                    s = 255 - r
        log_transform(g)               s = c*log(1+r),  c = 255/log(1+r_max)
        gamma_transform(g, gamma)      s = 255*(r/255)^gamma   (via a 256-entry LUT)
        contrast_stretch(g,r1,s1,r2,s2) piecewise linear through
                                        (0,0),(r1,s1),(r2,s2),(255,255)
        minmax_stretch(g)              s = (r-rmin)*255/(rmax-rmin)
        level_slice(g, A, B, keep)     [A,B] -> 255, others -> 0 or unchanged
        bit_plane(g, k)                ((r >> k) & 1) * 255
  (b) AUTO-GAMMA: find the gamma that brings the mean intensity closest to
      128 by (i) a grid search over 0.10..3.00 and (ii) the closed form
      gamma = log(0.5) / log(mean/255). Compare the two.
  (c) Contrast stretch with (r1,s1)=(rmin,0) and (r2,s2)=(rmax,255) and
      show it equals the min-max stretch.
  (d) Plot ALL transfer functions T(r) on one labelled graph.
  (e) Split the stretched image into its 8 bit-planes; reconstruct it
      from only the top 4 planes and report the MSE.
  (f) Print a Pandas table of mean / std / min / max for every output.

CONCEPTS TESTED
  log expands dark values, gamma < 1 brightens / > 1 darkens, LUT,
  piecewise rules (r1=s1 & r2=s2 -> identity; r1=r2, s1=0, s2=255 ->
  threshold), slicing, bit-plane significance, float before log/pow.

HOW TO RUN
  Set IMAGE_PATH. Missing file -> synthetic dark scene is generated.
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_PATH = "night.jpg"


def show(imgs, titles, cols=4, cmap="gray"):
    rows = (len(imgs) + cols - 1) // cols
    plt.figure(figsize=(4 * cols, 3.6 * rows))
    for i, (im, t) in enumerate(zip(imgs, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB),
                   cmap=cmap, vmin=0, vmax=255)
        plt.title(t, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def synthetic_dark():
    rng = np.random.default_rng(0)
    x = np.tile(np.linspace(0, 1, 512), (384, 1))
    scene = 120 * x
    cv2.rectangle(scene, (60, 80), (200, 300), 200, -1)    # building
    cv2.circle(scene, (380, 120), 50, 255, -1)              # street lamp
    for i in range(70, 300, 40):
        cv2.rectangle(scene, (80, i), (110, i + 20), 90, -1)  # windows
    scene += rng.normal(0, 6, scene.shape)
    scene = np.clip(scene, 0, 255) * (70 / 255)             # squeeze into 0..70
    return scene.astype(np.uint8)


def load_gray(path):
    if path and os.path.exists(path):
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is not None:
            return g
    print(f"[info] '{path}' not found -> synthetic underexposed image")
    return synthetic_dark()


# ---------------- (a) transforms ----------------
def negative(g):
    return 255 - g


def log_transform(g):
    f = g.astype(np.float64)
    c = 255 / np.log(1 + f.max())
    return np.uint8(np.clip(c * np.log1p(f), 0, 255))


def gamma_lut(gamma):
    return np.clip(255 * (np.arange(256) / 255.0) ** gamma, 0, 255).astype(np.uint8)


def gamma_transform(g, gamma):
    return cv2.LUT(g, gamma_lut(gamma))


def stretch_lut(r1, s1, r2, s2):
    return np.interp(np.arange(256), [0, r1, r2, 255], [0, s1, s2, 255]).astype(np.uint8)


def contrast_stretch(g, r1, s1, r2, s2):
    assert r1 <= r2 and s1 <= s2, "need r1<=r2 and s1<=s2 for a monotonic function"
    return cv2.LUT(g, stretch_lut(r1, s1, r2, s2))


def minmax_stretch(g):
    f = g.astype(np.float64)
    return np.uint8(np.round((f - f.min()) * 255 / (f.max() - f.min())))


def level_slice(g, A, B, keep_background=True):
    out = g.copy() if keep_background else np.zeros_like(g)
    out[(g >= A) & (g <= B)] = 255
    return out


def bit_plane(g, k):
    return ((g >> k) & 1).astype(np.uint8) * 255


gray = load_gray(IMAGE_PATH)
mean0 = gray.mean()
print(f"original mean = {mean0:.1f}, range = [{gray.min()}, {gray.max()}]")

# ---------------- (b) auto gamma ----------------
gammas = np.round(np.arange(0.10, 3.001, 0.05), 2)
errors = [abs(gamma_transform(gray, gm).mean() - 128) for gm in gammas]
g_grid = gammas[int(np.argmin(errors))]
g_closed = np.log(0.5) / np.log(mean0 / 255)
print(f"(b) grid-search gamma = {g_grid}  -> mean {gamma_transform(gray, g_grid).mean():.1f}")
print(f"    closed-form gamma = {g_closed:.3f} -> mean {gamma_transform(gray, g_closed).mean():.1f}")
print("    (closed form maps the MEAN pixel to 128; grid search makes the mean OF THE OUTPUT 128 —"
      " they differ because mean(T(r)) != T(mean(r)) for non-linear T)")

# ---------------- (c) stretch == min-max ----------------
rmin, rmax = int(gray.min()), int(gray.max())
cs = contrast_stretch(gray, rmin, 0, rmax, 255)
mm = minmax_stretch(gray)
print(f"(c) max |contrast_stretch - minmax| = {np.abs(cs.astype(int) - mm).max()} (<=1 due to rounding)")

outputs = {
    "Original": gray,
    "Negative": negative(gray),
    "Log": log_transform(gray),
    f"Gamma {g_grid}": gamma_transform(gray, g_grid),
    "Gamma 2.0": gamma_transform(gray, 2.0),
    "Contrast stretch": cs,
    "Slice [40,70] keep": level_slice(gray, 40, 70, True),
    "Slice [40,70] binary": level_slice(gray, 40, 70, False),
}
show(list(outputs.values()), list(outputs.keys()))

# ---------------- (d) transfer curves ----------------
r = np.arange(256)
plt.figure(figsize=(7, 6))
plt.plot(r, 255 - r, label="negative")
plt.plot(r, (255 / np.log(1 + rmax)) * np.log1p(r), label=f"log (c from r_max={rmax})")
for gm in [0.3, g_grid, 1.0, 2.0]:
    plt.plot(r, gamma_lut(gm), label=f"gamma {gm}")
plt.plot(r, stretch_lut(rmin, 0, rmax, 255), "--", label=f"stretch ({rmin},0)-({rmax},255)")
plt.plot(r, stretch_lut(100, 0, 100, 255), ":", label="r1=r2 -> threshold")
plt.xlim(0, 255); plt.ylim(0, 260)
plt.xlabel("input r"); plt.ylabel("output s"); plt.title("Transfer functions")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()

# ---------------- (e) bit planes ----------------
planes = [bit_plane(cs, k) for k in range(8)]
show(planes[::-1], [f"bit {k}" + (" (MSB)" if k == 7 else " (LSB)" if k == 0 else "")
                     for k in range(7, -1, -1)])
top4 = sum(((cs >> k) & 1).astype(np.uint16) << k for k in range(4, 8)).astype(np.uint8)
mse = np.mean((cs.astype(float) - top4) ** 2)
print(f"(e) reconstruction from bits 4-7: MSE = {mse:.2f} "
      f"(max possible error per pixel = 15)")
show([cs, top4], ["Stretched (8 bits)", f"Top-4 planes, MSE={mse:.1f}"], cols=2)

# ---------------- (f) stats table ----------------
stats = pd.DataFrame({k: [v.mean(), v.std(), v.min(), v.max()] for k, v in outputs.items()},
                     index=["mean", "std", "min", "max"]).T.round(1)
print("\n(f)\n", stats)
