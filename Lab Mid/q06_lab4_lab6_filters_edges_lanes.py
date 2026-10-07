"""
=====================================================================
Q6  |  LAB 4 + LAB 6 — Convolution from scratch, smoothing vs noise,
                       gradients, Canny / LoG, Hough lane detection
     "Dash-cam Lane Assistant"
=====================================================================
SCENARIO
  A noisy dash-cam frame must be cleaned and analysed to find the two lane
  markings in front of the car.

TASKS
  (a) Implement conv2d(img, kernel, stride, pad) from scratch (true
      convolution = kernel flipped 180 deg). Check the output size against
      O = floor((N - K + 2P)/S) + 1 for valid/same/stride-2, and verify the
      'same' result equals cv2.filter2D with the FLIPPED kernel and
      BORDER_CONSTANT.
  (b) Build a 5x5 Gaussian kernel BY FORMULA (sigma = 1), normalise it and
      compare with cv2.getGaussianKernel outer product.
  (c) Add Gaussian noise (sigma=20) and 5 % salt-and-pepper noise.
      Denoise each with box 5x5, Gaussian 5x5, median 5, bilateral.
      Implement PSNR yourself, verify with cv2.PSNR, and print a table.
      State which filter wins for which noise.
  (d) Sobel gx, gy (CV_64F), magnitude, direction. Visualise direction as
      an HSV image (hue = angle, value = magnitude).
  (e) Edge maps: Canny with (low, high) in {(30,90),(50,150),(100,250)},
      Laplacian of raw image vs LoG (Gaussian then Laplacian).
  (f) LANES: Canny -> trapezium ROI mask -> HoughLinesP. Split segments by
      slope sign into left/right, average each side into ONE line
      (np.polyfit on endpoints) and draw both lanes from the bottom of the
      image up to the ROI top.

CONCEPTS TESTED
  correlation vs convolution, padding/stride sizes, filter2D has no
  stride, Gaussian weights, median for salt & pepper, CV_64F to keep
  negative gradients, Canny hysteresis thresholds, LoG noise robustness,
  Hough rho-theta voting, None-check on Hough output.
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_PATH = "road.jpg"


def show(imgs, titles, cols=4, cmap="gray"):
    rows = (len(imgs) + cols - 1) // cols
    plt.figure(figsize=(4.2 * cols, 3.4 * rows))
    for i, (im, t) in enumerate(zip(imgs, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap=cmap)
        plt.title(t, fontsize=9); plt.axis("off")
    plt.tight_layout(); plt.show()


def synthetic_road():
    h, w = 360, 640
    img = np.zeros((h, w, 3), np.uint8)
    img[:h // 2] = (200, 170, 120)                                    # sky
    road = np.array([[0, h], [w // 2 - 40, h // 2], [w // 2 + 40, h // 2], [w, h]], np.int32)
    img[h // 2:] = (60, 110, 60)                                      # grass
    cv2.fillPoly(img, [road], (80, 80, 80))
    cv2.line(img, (90, h), (w // 2 - 15, h // 2 + 10), (255, 255, 255), 8)   # left lane
    for y in range(h, h // 2 + 10, -50):                              # dashed right lane
        t0, t1 = (h - y) / (h / 2 - 10), (h - y + 28) / (h / 2 - 10)
        x0, x1 = int(560 + (w // 2 + 15 - 560) * t0), int(560 + (w // 2 + 15 - 560) * t1)
        cv2.line(img, (x0, y), (x1, y - 28), (0, 220, 255), 8)
    return img


img = cv2.imread(IMAGE_PATH) if os.path.exists(IMAGE_PATH) else None
if img is None:
    print("[info] using synthetic road")
    img = synthetic_road()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
H, W = gray.shape


# ---------------- (a) conv from scratch ----------------
def conv2d(im, k, stride=1, pad=0):
    k = np.flipud(np.fliplr(k)).astype(np.float32)          # TRUE convolution
    im = np.pad(im.astype(np.float32), pad)                  # zero padding
    kh, kw = k.shape
    oh = (im.shape[0] - kh) // stride + 1
    ow = (im.shape[1] - kw) // stride + 1
    out = np.zeros((oh, ow), np.float32)
    for i in range(kh):                                      # loop over kernel, not pixels (fast)
        for j in range(kw):
            out += k[i, j] * im[i:i + stride * oh:stride, j:j + stride * ow:stride]
    return out


k = np.array([[1, 2, 0], [0, 1, -1], [-2, 0, 1]], np.float32)  # asymmetric on purpose
size = lambda N, K, P, S: (N - K + 2 * P) // S + 1
for name, P, S in [("valid", 0, 1), ("same", 1, 1), ("stride 2", 1, 2)]:
    out = conv2d(gray, k, S, P)
    print(f"(a) {name:8s}: got {out.shape}, formula ({size(H, 3, P, S)}, {size(W, 3, P, S)})")
same = conv2d(gray, k, 1, 1)
ref = cv2.filter2D(gray.astype(np.float32), -1, cv2.flip(k, -1), borderType=cv2.BORDER_CONSTANT)
corr = cv2.filter2D(gray.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
print("    conv2d == filter2D(flipped):", np.allclose(same, ref, atol=1e-3),
      "| conv2d == filter2D(unflipped):", np.allclose(same, corr, atol=1e-3))

# ---------------- (b) Gaussian by formula ----------------
ax = np.arange(-2, 3)
xx, yy = np.meshgrid(ax, ax)
G = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.0 ** 2))
G /= G.sum()
g1 = cv2.getGaussianKernel(5, 1.0)
print("(b) formula kernel == getGaussianKernel outer product:", np.allclose(G, g1 @ g1.T))
print(np.round(G, 4))

# ---------------- (c) noise & denoising ----------------
rng = np.random.default_rng(0)
gauss_noisy = np.clip(gray + rng.normal(0, 20, gray.shape), 0, 255).astype(np.uint8)
sp_noisy = gray.copy()
r = rng.random(gray.shape)
sp_noisy[r < 0.025] = 0
sp_noisy[r > 0.975] = 255


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)


filters = {
    "box 5x5": lambda x: cv2.blur(x, (5, 5)),
    "gaussian 5x5": lambda x: cv2.GaussianBlur(x, (5, 5), 0),
    "median 5": lambda x: cv2.medianBlur(x, 5),
    "bilateral": lambda x: cv2.bilateralFilter(x, 9, 75, 75),
}
rows = {"noisy (no filter)": [psnr(gray, gauss_noisy), psnr(gray, sp_noisy)]}
den = {}
for n, f in filters.items():
    a, b = f(gauss_noisy), f(sp_noisy)
    den[n] = (a, b)
    rows[n] = [psnr(gray, a), psnr(gray, b)]
assert abs(psnr(gray, gauss_noisy) - cv2.PSNR(gray, gauss_noisy)) < 1e-6
table = pd.DataFrame(rows, index=["Gaussian noise", "Salt & pepper"]).T.round(2)
print("\n(c) PSNR (dB, higher = better). Manual PSNR verified against cv2.PSNR\n", table)
print("    best for Gaussian noise:", table["Gaussian noise"].iloc[1:].idxmax(),
      "| best for salt & pepper:", table["Salt & pepper"].iloc[1:].idxmax())
show([gauss_noisy, den["gaussian 5x5"][0], sp_noisy, den["median 5"][1]],
     ["Gaussian noise", "-> Gaussian blur", "Salt & pepper", "-> median"])

# ---------------- (d) gradients ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 1.4)
gx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
gy = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
hsv = np.zeros((H, W, 3), np.uint8)
hsv[..., 0] = (ang / 2).astype(np.uint8)                      # 0..360 -> 0..180
hsv[..., 1] = 255
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
dir_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
lost = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)               # trap: negatives clipped
show([cv2.convertScaleAbs(gx), lost, cv2.convertScaleAbs(gy),
      cv2.convertScaleAbs(mag), dir_vis],
     ["|gx| (CV_64F)", "gx in CV_8U (half the edges lost)", "|gy|", "magnitude", "direction (hue)"],
     cols=5)

# ---------------- (e) Canny, Laplacian, LoG ----------------
cannys = [cv2.Canny(blur, lo, hi) for lo, hi in [(30, 90), (50, 150), (100, 250)]]
lap_raw = cv2.convertScaleAbs(cv2.Laplacian(gauss_noisy, cv2.CV_64F, ksize=3))
lap_log = cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(gauss_noisy, (7, 7), 1.5), cv2.CV_64F, ksize=3))
for (lo, hi), c in zip([(30, 90), (50, 150), (100, 250)], cannys):
    print(f"(e) Canny({lo},{hi}): {np.count_nonzero(c)} edge pixels")
show(cannys + [lap_raw, lap_log],
     ["Canny 30/90", "Canny 50/150", "Canny 100/250", "Laplacian (noisy)", "LoG (noisy)"], cols=5)

# ---------------- (f) lanes ----------------
edges = cv2.Canny(blur, 50, 150)
roi_top = int(H * 0.55)
trap = np.array([[(0, H), (int(W * 0.45), roi_top), (int(W * 0.55), roi_top), (W, H)]], np.int32)
mask = np.zeros_like(edges)
cv2.fillPoly(mask, trap, 255)
masked = cv2.bitwise_and(edges, mask)
segs = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=60)

out = img.copy()
sides = {"left": [], "right": []}
if segs is not None:
    for x1, y1, x2, y2 in segs[:, 0]:
        slope = (y2 - y1) / (x2 - x1 + 1e-6)
        if abs(slope) < 0.4:
            continue                                        # ignore near-horizontal
        side = "left" if slope < 0 else "right"            # y grows downward!
        sides[side] += [(x1, y1), (x2, y2)]
        cv2.line(out, (x1, y1), (x2, y2), (255, 0, 255), 1)
for side, pts in sides.items():
    if len(pts) < 2:
        print(f"(f) no {side} lane found"); continue
    pts = np.array(pts)
    m, b = np.polyfit(pts[:, 1], pts[:, 0], 1)             # x = m*y + b (robust for steep lines)
    xb, xt = int(m * H + b), int(m * roi_top + b)
    cv2.line(out, (xb, H), (xt, roi_top), (0, 0, 255), 6)
    print(f"(f) {side} lane: x = {m:.3f}*y + {b:.1f}  ({len(pts) // 2} segments)")
show([edges, mask, masked, out], ["Canny", "ROI mask", "Masked edges", "Detected lanes"])
