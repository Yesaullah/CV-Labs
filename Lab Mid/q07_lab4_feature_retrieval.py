"""
=====================================================================
Q7  |  LAB 4 — Feature descriptors: HOG, LBP, colour histogram,
                edge-direction histogram, texture energy; retrieval
     "Fabric Pattern Search Engine"
=====================================================================
SCENARIO
  A textile company wants to search its catalogue: given a photo of a
  fabric (possibly ROTATED and noisy), return the most similar catalogue
  items. Compare which descriptors survive rotation.

TASKS
  (a) Build (or load) a catalogue of 6 fabric images; create a QUERY that
      is catalogue item #0 rotated 90 deg + Gaussian noise.
  (b) For every image (resized to 128x128) extract:
        - HOG  (9 orientations, 8x8 cells, 2x2 blocks) -> print vector length
        - LBP  (P=8, R=1, 'uniform')  -> normalised (P+2)-bin histogram
        - HSV colour histogram, 8x8 bins over H,S  -> 64-D, normalised
        - HED: histogram of gradient angles (0..180, 18 bins) at Canny
          edge pixels only
        - texture energy (local mean of I^2, 7x7) and texture contrast
          (local std, 7x7) -> report their global means
  (c) Rank the catalogue against the query for each descriptor using
      cv2.compareHist (HISTCMP_CORREL for histograms; cosine similarity
      for HOG). Print a Pandas table of scores and the top-1 per descriptor.
  (d) Combine descriptors (average of min-max-normalised scores) and give
      the final ranking.
  (e) Explain from the results which descriptors are rotation-invariant
      and why (print the explanation).
  (f) Visualise: catalogue, query, HOG image of query, LBP image of query,
      bar charts of LBP and HED histograms for the query vs item #0.

CONCEPTS TESTED
  global vs local features, HOG pipeline (gradients -> cells -> block
  norm), LBP uniform bins = P+2, HSV hist ranges [0,180,0,256],
  compareHist metrics, float before squaring, rotation invariance.

HOW TO RUN
  Set CATALOG_PATHS (list of 6 images) to use your own; otherwise
  synthetic textures are generated.
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from skimage import exposure

CATALOG_PATHS = []   # e.g. ["f1.jpg", ..., "f6.jpg"]
SIZE = 128


def synthetic_catalog():
    s = 256
    yy, xx = np.mgrid[0:s, 0:s]
    imgs = []
    def colorize(g, bgr):
        g = g.astype(np.float32)[..., None] / 255
        return (g * np.array(bgr, np.float32) + (1 - g) * 30).astype(np.uint8)
    imgs.append(colorize(((xx // 16) % 2) * 255, (40, 40, 220)))             # 0 red vertical stripes
    imgs.append(colorize(((yy // 16) % 2) * 255, (40, 40, 220)))             # 1 red horizontal stripes
    imgs.append(colorize((((xx // 32) + (yy // 32)) % 2) * 255, (220, 60, 40)))  # 2 blue checker
    imgs.append(colorize(((xx + yy) // 20 % 2) * 255, (40, 200, 40)))        # 3 green diagonal
    rng = np.random.default_rng(5)
    noise = cv2.GaussianBlur(rng.integers(0, 256, (s, s)).astype(np.uint8), (9, 9), 0)
    imgs.append(colorize(cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX), (40, 200, 220)))  # 4 yellow blotch
    dots = np.zeros((s, s), np.uint8)
    for y in range(16, s, 32):
        for x in range(16, s, 32):
            cv2.circle(dots, (x, y), 9, 255, -1)
    imgs.append(colorize(dots, (200, 40, 200)))                               # 5 magenta polka
    return imgs


catalog = [cv2.imread(p) for p in CATALOG_PATHS] if CATALOG_PATHS else []
if len(catalog) < 2 or any(c is None for c in catalog):
    print("[info] using 6 synthetic fabric textures")
    catalog = synthetic_catalog()
catalog = [cv2.resize(c, (SIZE, SIZE)) for c in catalog]
names = [f"item{i}" for i in range(len(catalog))]

# ---------------- (a) query ----------------
rng = np.random.default_rng(7)
query = cv2.rotate(catalog[0], cv2.ROTATE_90_CLOCKWISE)
query = np.clip(query + rng.normal(0, 8, query.shape), 0, 255).astype(np.uint8)


# ---------------- (b) descriptors ----------------
def features(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)        # LBP/HOG are noise-sensitive: smooth first
    f = {}
    f["HOG"], hog_img = hog(g, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)
    P, R = 8, 1
    lbp = local_binary_pattern(g, P, R, method="uniform")
    h, _ = np.histogram(lbp, bins=np.arange(P + 3))
    f["LBP"] = (h / h.sum()).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ch = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    f["Colour"] = cv2.normalize(ch, None).flatten()
    gb = g
    gx = cv2.Sobel(gb, cv2.CV_64F, 1, 0); gy = cv2.Sobel(gb, cv2.CV_64F, 0, 1)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180                  # unsigned orientation
    edges = cv2.Canny(gb, 50, 150)
    hed, _ = np.histogram(ang[edges > 0], bins=18, range=(0, 180))
    f["HED"] = (hed / max(hed.sum(), 1)).astype(np.float32)
    fl = g.astype(np.float32)                                    # float! uint8**2 overflows
    energy = cv2.blur(fl ** 2, (7, 7))
    contrast = np.sqrt(np.maximum(energy - cv2.blur(fl, (7, 7)) ** 2, 0))
    f["energy"], f["contrast"] = energy.mean(), contrast.mean()
    return f, hog_img, lbp


db = [features(c)[0] for c in catalog]
qf, q_hog, q_lbp = features(query)
print(f"(b) HOG length = {qf['HOG'].shape[0]}  "
      f"(= 15x15 blocks x 2x2 cells x 9 bins for 128x128)")
print(f"    LBP bins = {qf['LBP'].shape[0]}, colour bins = {qf['Colour'].shape[0]}, HED bins = {qf['HED'].shape[0]}")
print(pd.DataFrame({"energy": [d["energy"] for d in db] + [qf["energy"]],
                    "contrast": [d["contrast"] for d in db] + [qf["contrast"]]},
                   index=names + ["query"]).round(1).T)


# ---------------- (c) ranking ----------------
def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


scores = {"HOG": [cosine(qf["HOG"], d["HOG"]) for d in db]}
for key in ["LBP", "Colour", "HED"]:
    scores[key] = [cv2.compareHist(qf[key], d[key], cv2.HISTCMP_CORREL) for d in db]
table = pd.DataFrame(scores, index=names).round(3)
print("\n(c) similarity to query (higher = more similar)\n", table)
for k in table:
    print(f"    top-1 by {k:6s}: {table[k].idxmax()}")

# ---------------- (d) combined ----------------
norm = (table - table.min()) / (table.max() - table.min() + 1e-12)
table["combined"] = norm.mean(axis=1).round(3)
print("\n(d) final ranking (all 4):", list(table["combined"].sort_values(ascending=False).index))
inv = norm[["LBP", "Colour"]].mean(axis=1).round(3).sort_values(ascending=False)
print("    rotation-invariant only (LBP + Colour):", inv.to_dict())

# ---------------- (e) explanation ----------------
print("""
(e) Rotation invariance:
    - Colour histogram: counts pixel colours, ignores positions -> invariant.
    - LBP 'uniform': histogram of local patterns; uniform codes group rotated
      versions of a pattern -> (approximately) invariant.
    - HOG: stores orientation PER CELL in a fixed spatial layout; a 90 deg
      rotation moves gradients to other bins/cells -> NOT invariant
      (with the 90-deg rotation HOG picks the HORIZONTAL-stripe item).
    - LBP and Colour give item0 and item1 IDENTICAL scores: they cannot tell
      a texture from its rotated copy — that IS rotation invariance.
    - Mixing invariant and non-invariant descriptors lets the non-invariant
      ones drag the ranking; for rotated queries use invariant ones only.
    - LBP is very noise-sensitive (the noisy query resembles the blotchy
      item), which is why every image is Gaussian-blurred first.
    - HED: rotation shifts the orientation histogram circularly -> NOT
      invariant unless histograms are aligned to their dominant peak.""")

# ---------------- (f) visualise ----------------
plt.figure(figsize=(16, 3))
for i, im in enumerate(catalog + [query]):
    plt.subplot(1, len(catalog) + 1, i + 1)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("QUERY" if i == len(catalog) else names[i]); plt.axis("off")
plt.tight_layout(); plt.show()

plt.figure(figsize=(16, 4))
plt.subplot(141); plt.imshow(exposure.rescale_intensity(q_hog, in_range=(0, q_hog.max() / 2)), cmap="gray")
plt.title("HOG of query"); plt.axis("off")
plt.subplot(142); plt.imshow(q_lbp, cmap="gray"); plt.title("LBP of query"); plt.axis("off")
x = np.arange(len(qf["LBP"]))
plt.subplot(143); plt.bar(x - 0.2, qf["LBP"], 0.4, label="query"); plt.bar(x + 0.2, db[0]["LBP"], 0.4, label="item0")
plt.title("LBP histograms"); plt.legend()
x = np.arange(18) * 10
plt.subplot(144); plt.bar(x - 2, qf["HED"], 4, label="query"); plt.bar(x + 2, db[0]["HED"], 4, label="item0")
plt.title("Edge-direction histograms (deg)"); plt.legend()
plt.tight_layout(); plt.show()
