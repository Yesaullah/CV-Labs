"""
=====================================================================
Q3  |  LAB 2 — Histograms, equalization (manual), CLAHE, colour HE,
                histogram matching
     "X-ray Contrast Enhancement Pipeline"
=====================================================================
SCENARIO
  A hospital's old scanner produces low-contrast X-rays (histogram packed
  in a narrow band). Build and justify an enhancement pipeline.

TASKS
  (a) Compute the 256-bin histogram MANUALLY (np.bincount) and verify it is
      identical to cv2.calcHist. Compute the normalised histogram
      p(r_k) = n_k / MN and the CDF.
  (b) Implement histogram equalization MANUALLY with
          s_k = round( (L-1) * CDF(r_k) )                (textbook form)
      and the "cdf_min" form used by OpenCV
          s_k = round( (CDF(r_k)-CDF_min) / (1-CDF_min) * (L-1) )
      Report the max absolute difference of each against cv2.equalizeHist.
  (c) Apply CLAHE with clipLimit in {1, 2, 4, 8} (tile 8x8). Report the
      standard deviation (global contrast) of each and comment.
  (d) COLOUR image: equalize (i) each B,G,R channel separately and
      (ii) only the Y channel of YCrCb. Measure the mean absolute HUE shift
      w.r.t. the original for both and explain which is correct.
  (e) HISTOGRAM MATCHING (specification): make the X-ray's histogram follow
      that of a reference image, implemented with CDFs + np.interp.
  (f) Plot every image with its histogram, and overlay the CDF before and
      after equalization (equalized CDF should be ~ a straight line).

CONCEPTS TESTED
  calcHist argument lists, normalised histogram, CDF, HE formula,
  why HE output is only approximately flat, CLAHE clipLimit/tiles,
  equalizing luminance not RGB, histogram specification.

HOW TO RUN
  Set XRAY_PATH / COLOR_PATH. Missing files -> synthetic images.
=====================================================================
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

XRAY_PATH = "xray.jpg"
COLOR_PATH = "color.jpg"


def synthetic_xray():
    rng = np.random.default_rng(1)
    img = np.full((400, 400), 110.0)
    cv2.ellipse(img, (200, 210), (130, 170), 0, 0, 360, 135, -1)   # chest
    for y in range(90, 330, 30):                                   # ribs
        cv2.ellipse(img, (200, y), (120, 25), 0, 200, 340, 160, 6)
    cv2.rectangle(img, (190, 60), (210, 350), 165, -1)             # spine
    img = cv2.GaussianBlur(img, (7, 7), 0) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def synthetic_color():
    img = np.zeros((300, 400, 3), np.uint8)
    img[:] = (60, 90, 120)
    cv2.circle(img, (120, 150), 70, (40, 160, 70), -1)
    cv2.rectangle(img, (230, 80), (360, 220), (150, 70, 60), -1)
    return cv2.convertScaleAbs(img, alpha=0.6, beta=30)            # low contrast


def load(path, gray, fallback):
    if path and os.path.exists(path):
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
        if im is not None:
            return im
    print(f"[info] '{path}' not found -> synthetic image")
    return fallback()


xray = load(XRAY_PATH, True, synthetic_xray)
color = load(COLOR_PATH, False, synthetic_color)
MN, L = xray.size, 256

# ---------------- (a) histogram, pdf, cdf ----------------
hist_manual = np.bincount(xray.ravel(), minlength=256)
hist_cv = cv2.calcHist([xray], [0], None, [256], [0, 256]).ravel().astype(int)
print("(a) manual histogram == calcHist:", np.array_equal(hist_manual, hist_cv))
pdf = hist_manual / MN
cdf = np.cumsum(pdf)
print(f"    sum(p) = {pdf.sum():.4f}, occupied levels = {np.count_nonzero(hist_manual)}, "
      f"range = [{xray.min()}, {xray.max()}]")

# ---------------- (b) manual equalization ----------------
lut_text = np.round((L - 1) * cdf).astype(np.uint8)
cdf_min = cdf[cdf > 0].min()
lut_cv = np.round((cdf - cdf_min) / (1 - cdf_min) * (L - 1)).clip(0, 255).astype(np.uint8)
eq_text, eq_cvform, eq_cv = lut_text[xray], lut_cv[xray], cv2.equalizeHist(xray)
print("(b) max |textbook - equalizeHist| =", np.abs(eq_text.astype(int) - eq_cv).max())
print("    max |cdf_min form - equalizeHist| =", np.abs(eq_cvform.astype(int) - eq_cv).max())
print("    output levels used:", np.count_nonzero(np.bincount(eq_cv.ravel(), minlength=256)),
      "-> HE cannot create new levels, so the histogram is only approximately flat")

# ---------------- (c) CLAHE ----------------
print(f"(c) std original = {xray.std():.1f}, std global HE = {eq_cv.std():.1f}")
clahe_imgs = {}
for cl in [1, 2, 4, 8]:
    out = cv2.createCLAHE(clipLimit=cl, tileGridSize=(8, 8)).apply(xray)
    clahe_imgs[f"CLAHE clip={cl}"] = out
    print(f"    CLAHE clip={cl}: std = {out.std():.1f}")
print("    Higher clipLimit -> more local contrast but more amplified noise; "
      "CLAHE avoids HE's noise blow-up in flat areas.")

# ---------------- (d) colour equalization ----------------
per_channel = cv2.merge([cv2.equalizeHist(ch) for ch in cv2.split(color)])
ycc = cv2.cvtColor(color, cv2.COLOR_BGR2YCrCb)
ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
y_only = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def hue_shift(a, b):
    ha = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)[:, :, 0].astype(int)
    hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[:, :, 0].astype(int)
    d = np.abs(ha - hb)
    return np.minimum(d, 180 - d).mean()          # hue is circular (0..179)


print(f"(d) mean hue shift  per-channel HE = {hue_shift(color, per_channel):.2f}, "
      f"Y-only HE = {hue_shift(color, y_only):.2f}")
print("    Equalizing B,G,R independently changes their ratios -> colours change. "
      "Equalize luminance only.")

# ---------------- (e) histogram matching ----------------
ref = np.clip(np.random.default_rng(2).normal(90, 45, xray.shape), 0, 255).astype(np.uint8)
ref_cdf = np.cumsum(np.bincount(ref.ravel(), minlength=256)) / ref.size
match_lut = np.interp(cdf, ref_cdf, np.arange(256)).astype(np.uint8)   # src cdf -> ref level
matched = match_lut[xray]
print(f"(e) matched mean/std = {matched.mean():.1f}/{matched.std():.1f}  "
      f"(reference {ref.mean():.1f}/{ref.std():.1f})")

# ---------------- (f) plots ----------------
items = [("Original", xray), ("Manual HE", eq_text), ("cv2.equalizeHist", eq_cv),
         ("CLAHE clip=2", clahe_imgs["CLAHE clip=2"]), ("Matched to ref", matched)]
plt.figure(figsize=(18, 7))
for i, (t, im) in enumerate(items):
    plt.subplot(2, len(items), i + 1); plt.imshow(im, cmap="gray", vmin=0, vmax=255)
    plt.title(t); plt.axis("off")
    plt.subplot(2, len(items), i + 1 + len(items))
    plt.hist(im.ravel(), bins=256, range=(0, 256), color="gray")
    c = np.cumsum(np.bincount(im.ravel(), minlength=256)) / im.size
    plt.twinx().plot(c, "r"); plt.xlim(0, 255)
plt.suptitle("Images, histograms (gray) and CDFs (red)"); plt.tight_layout(); plt.show()

plt.figure(figsize=(14, 4))
for i, (t, im) in enumerate([("Colour original", color), ("Per-channel HE (wrong)", per_channel),
                             ("Y-channel HE (correct)", y_only)]):
    plt.subplot(1, 3, i + 1); plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
