"""
=====================================================================
Q8  |  LAB 5 — Global / Otsu (manual) / adaptive thresholding,
                region growing, K-means, quantitative evaluation (IoU)
     "Which Segmentation Method Should the Factory Use?"
=====================================================================
SCENARIO
  Dark defects (holes, stains) must be segmented on a metal sheet photographed
  under UNEVEN lighting (bright on the left, dark on the right). A ground-
  truth mask is available, so methods can be compared with numbers.

TASKS
  (a) Global threshold with T in {60, 100, 140} (dark objects ->
      THRESH_BINARY_INV).
  (b) Implement Otsu from scratch (maximise between-class variance
      w0*w1*(mu0-mu1)^2), compare T with OpenCV's Otsu (with and without a
      5x5 Gaussian pre-blur). Plot the histogram with T marked.
  (c) Adaptive thresholding: MEAN and GAUSSIAN with blockSize in
      {11, 51, 101} and C in {2, 10}.
  (d) Region growing from a seed (8-connectivity, |I - I_seed| <= T) for
      T in {10, 25, 50}; then an ADAPTIVE version that compares to the
      running region mean instead of the seed.
  (e) K-means (K=2) on [intensity] and on [intensity, x, y] features; pick
      the darker cluster as foreground.
  (f) Compute IoU and Dice of every method against the ground truth,
      print a sorted Pandas leaderboard, and show the best 8 masks.
  (g) Bonus: remove lighting first (divide by a heavily blurred background
      estimate), then re-run Otsu and report the new IoU.

CONCEPTS TESTED
  assumptions behind each method (bimodal histogram, uniform lighting,
  connectivity), blockSize odd, C effect, Otsu ret value, region growing
  leakage, K-means float32 input, evaluation with IoU/Dice.
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_PATH = "sheet.jpg"     # ground truth only exists for the synthetic image


def synthetic_sheet():
    rng = np.random.default_rng(11)
    h, w = 300, 450
    gt = np.zeros((h, w), np.uint8)
    for (x, y, r) in [(70, 80, 25), (200, 200, 35), (330, 90, 20), (390, 230, 28), (120, 230, 18)]:
        cv2.circle(gt, (x, y), r, 255, -1)
    cv2.ellipse(gt, (260, 70), (40, 15), 30, 0, 360, 255, -1)
    light = np.tile(np.linspace(1.0, 0.35, w), (h, 1))          # uneven illumination
    base = np.where(gt > 0, 70, 190).astype(np.float64)
    img = base * light + rng.normal(0, 8, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8), gt


if os.path.exists(IMAGE_PATH):
    gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE); gt = None
    print("[warn] real image: no ground truth, IoU will be skipped")
else:
    print("[info] using synthetic sheet with ground truth")
    gray, gt = synthetic_sheet()
results = {}


def iou_dice(pred, gt):
    p, g = pred > 0, gt > 0
    inter, union = np.logical_and(p, g).sum(), np.logical_or(p, g).sum()
    return inter / union, 2 * inter / (p.sum() + g.sum())


# ---------------- (a) global ----------------
for T in [60, 100, 140]:
    results[f"global T={T}"] = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY_INV)[1]


# ---------------- (b) Otsu ----------------
def otsu_manual(g):
    p = np.bincount(g.ravel(), minlength=256) / g.size
    levels = np.arange(256)
    best_v, best_t = -1, 0
    for t in range(1, 256):
        w0, w1 = p[:t].sum(), p[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (levels[:t] * p[:t]).sum() / w0
        mu1 = (levels[t:] * p[t:]).sum() / w1
        v = w0 * w1 * (mu0 - mu1) ** 2
        if v > best_v:
            best_v, best_t = v, t
    return best_t


t_manual = otsu_manual(gray)
t_cv, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
t_cv_b, otsu_b = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
print(f"(b) Otsu manual T = {t_manual} | OpenCV T = {t_cv:.0f} | OpenCV after blur T = {t_cv_b:.0f}")
print("    (manual uses 'pixel < t' as class 0; OpenCV uses '<= t' -> can differ by 1)")
results["Otsu"], results["Otsu + blur"] = otsu, otsu_b
plt.figure(figsize=(7, 3)); plt.hist(gray.ravel(), 256, range=(0, 256))
plt.axvline(t_cv, color="r", label=f"Otsu T={t_cv:.0f}"); plt.legend()
plt.title("Histogram — lighting gradient smears the two modes"); plt.show()

# ---------------- (c) adaptive ----------------
for mname, m in [("mean", cv2.ADAPTIVE_THRESH_MEAN_C), ("gauss", cv2.ADAPTIVE_THRESH_GAUSSIAN_C)]:
    for bs in [11, 51, 101]:
        for C in [2, 10]:
            results[f"adapt {mname} b={bs} C={C}"] = cv2.adaptiveThreshold(
                gray, 255, m, cv2.THRESH_BINARY_INV, bs, C)


# ---------------- (d) region growing ----------------
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def region_grow(g, seed, T, adaptive=False):
    h, w = g.shape
    mask = np.zeros((h, w), np.uint8)
    x0, y0 = seed
    ref = float(g[y0, x0]); total, count = ref, 1
    stack = [(x0, y0)]; mask[y0, x0] = 255
    while stack:
        x, y = stack.pop()
        for dx, dy in N8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] == 0:
                if abs(float(g[ny, nx]) - ref) <= T:        # float: no uint8 wrap-around
                    mask[ny, nx] = 255
                    stack.append((nx, ny))
                    if adaptive:
                        total += float(g[ny, nx]); count += 1; ref = total / count
    return mask


seeds = [(70, 80), (200, 200), (330, 90), (390, 230), (120, 230), (260, 70)]   # one per defect
for T in [10, 25, 50]:
    m = np.zeros_like(gray)
    for s in seeds:
        m |= region_grow(gray, s, T)
    results[f"region grow T={T}"] = m
m = np.zeros_like(gray)
for s in seeds:
    m |= region_grow(gray, s, 20, adaptive=True)
results["region grow adaptive T=20"] = m

# ---------------- (e) K-means ----------------
h, w = gray.shape
yy, xx = np.mgrid[0:h, 0:w]
crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
for name, Z in [("kmeans [I]", gray.reshape(-1, 1)),
                ("kmeans [I,x,y]", np.stack([gray.ravel(), xx.ravel() * 0.3, yy.ravel() * 0.3], 1))]:
    _, lab, cen = cv2.kmeans(Z.astype(np.float32), 2, None, crit, 10, cv2.KMEANS_PP_CENTERS)
    dark = int(np.argmin(cen[:, 0]))
    results[name] = (lab.reshape(h, w) == dark).astype(np.uint8) * 255

# ---------------- (g) illumination correction ----------------
bg = cv2.GaussianBlur(cv2.medianBlur(gray, 31), (0, 0), 25).astype(np.float32)
flat = cv2.normalize(gray.astype(np.float32) / (bg + 1), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
results["flat-field + Otsu"] = cv2.threshold(flat, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# ---------------- (f) leaderboard ----------------
if gt is not None:
    board = pd.DataFrame({k: iou_dice(v, gt) for k, v in results.items()},
                         index=["IoU", "Dice"]).T.sort_values("IoU", ascending=False).round(3)
    print("\n(f) leaderboard\n", board)
    best = list(board.index[:7])
    imgs = [gray, gt] + [results[k] for k in best]
    titles = ["Input", "Ground truth"] + [f"{k}\nIoU={board.loc[k, 'IoU']}" for k in best]
else:
    best = list(results)[:7]
    imgs, titles = [gray] + [results[k] for k in best], ["Input"] + best
plt.figure(figsize=(16, 7))
for i, (im, t) in enumerate(zip(imgs, titles)):
    plt.subplot(2, 5, i + 1); plt.imshow(im, cmap="gray"); plt.title(t, fontsize=8); plt.axis("off")
plt.tight_layout(); plt.show()
