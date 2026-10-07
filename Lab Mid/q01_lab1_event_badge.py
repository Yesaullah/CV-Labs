"""
=====================================================================
Q1  |  LAB 1 — Image I/O, ROI, drawing, masking, blending, statistics
     "Event Badge Generator"
=====================================================================
SCENARIO
  A conference wants an automatic badge maker. Given any portrait photo,
  produce a printable 400 x 600 (w x h) badge.

TASKS
  (a) Load the image, raise an error if it fails. Print its shape, dtype,
      resolution (W x H), number of channels and raw memory size in KB.
  (b) Take the largest CENTRED SQUARE crop of the photo and resize it to
      300 x 300.
  (c) Cut the square into a CIRCULAR photo (radius 140) using a mask:
      inside = photo, outside = pure white.
  (d) Build the badge on a blank canvas:
        - vertical colour gradient background (dark blue -> light blue),
          built with NumPy only (no loops)
        - 8-px border rectangle
        - paste the circular photo at the top centre using the mask
          (do NOT overwrite the background outside the circle)
        - the attendee name CENTRED horizontally (use cv2.getTextSize)
        - a semi-transparent (alpha = 0.5) red band at the bottom 15 %
          with the word "SPEAKER" written on it
  (e) Using Pandas, print descriptive statistics of the B, G, R channels of
      the square crop and report which channel is dominant.
  (f) Brighten the crop by +100 with NumPy addition and with cv2.add.
      Count how many values WRAPPED AROUND in the NumPy version and explain.
  (g) Display everything in one figure with titles; save the badge as PNG.

CONCEPTS TESTED
  imread / BGR vs RGB, img.shape = (h, w, c), slicing img[y, x],
  resize takes (w, h), np.zeros canvas, circle/rectangle/putText,
  bitwise_and/or/not with masks, addWeighted, pandas describe,
  uint8 saturation (cv2) vs wrap-around (NumPy).

HOW TO RUN
  Set IMAGE_PATH below. If the file is missing a synthetic portrait is used,
  so the script always runs.
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_PATH = "portrait.jpg"
NAME = "Yesaullah"


def show(imgs, titles, cols=3, cmap="gray"):
    rows = (len(imgs) + cols - 1) // cols
    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, (im, t) in enumerate(zip(imgs, titles)):
        plt.subplot(rows, cols, i + 1)
        if im.ndim == 3:
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(im, cmap=cmap)
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def synthetic_portrait():
    img = np.full((480, 640, 3), (190, 200, 210), np.uint8)
    cv2.rectangle(img, (220, 330), (420, 480), (90, 60, 40), -1)      # shoulders
    cv2.circle(img, (320, 220), 105, (120, 160, 215), -1)             # face
    cv2.circle(img, (285, 200), 12, (40, 40, 40), -1)                 # eyes
    cv2.circle(img, (355, 200), 12, (40, 40, 40), -1)
    cv2.ellipse(img, (320, 265), (40, 18), 0, 0, 180, (60, 60, 160), 4)  # smile
    return img


def load(path):
    if path and os.path.exists(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not decode {path}")
        return img
    print(f"[info] '{path}' not found -> using synthetic portrait")
    return synthetic_portrait()


# ---------------- (a) load & inspect ----------------
img = load(IMAGE_PATH)
h, w = img.shape[:2]
c = 1 if img.ndim == 2 else img.shape[2]
print("(a) shape:", img.shape, "| dtype:", img.dtype)
print(f"    resolution: {w} x {h} | channels: {c} | memory: {img.nbytes / 1024:.1f} KB")

# ---------------- (b) centred square crop ----------------
side = min(h, w)
y0, x0 = (h - side) // 2, (w - side) // 2
square = img[y0:y0 + side, x0:x0 + side]          # slicing is [rows=y, cols=x]
square = cv2.resize(square, (300, 300))           # (WIDTH, HEIGHT)

# ---------------- (c) circular photo ----------------
mask = np.zeros((300, 300), np.uint8)
cv2.circle(mask, (150, 150), 140, 255, -1)
inside = cv2.bitwise_and(square, square, mask=mask)
white = np.full_like(square, 255)
outside = cv2.bitwise_and(white, white, mask=cv2.bitwise_not(mask))
circular = cv2.bitwise_or(inside, outside)

# ---------------- (d) badge ----------------
BW, BH = 400, 600
top, bottom = np.array([90, 30, 10], np.float32), np.array([250, 200, 150], np.float32)  # BGR
t = np.linspace(0, 1, BH, dtype=np.float32)[:, None]                 # (BH, 1)
grad = (1 - t) * top + t * bottom                                    # (BH, 3) broadcast
badge = np.repeat(grad[:, None, :], BW, axis=1).astype(np.uint8)     # (BH, BW, 3)

cv2.rectangle(badge, (4, 4), (BW - 5, BH - 5), (255, 255, 255), 8)

px, py = (BW - 300) // 2, 40                                         # top-left of photo
roi = badge[py:py + 300, px:px + 300]
roi[mask == 255] = square[mask == 255]                               # paste only inside circle
cv2.circle(badge, (px + 150, py + 150), 140, (255, 255, 255), 3)     # photo ring

font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3
(tw, th_), base = cv2.getTextSize(NAME, font, scale, thick)
cv2.putText(badge, NAME, ((BW - tw) // 2, 400), font, scale, (255, 255, 255), thick)

overlay = badge.copy()
band_top = int(BH * 0.85)
cv2.rectangle(overlay, (0, band_top), (BW, BH), (0, 0, 255), -1)
badge = cv2.addWeighted(overlay, 0.5, badge, 0.5, 0)                 # 0.5*overlay + 0.5*badge
(tw, th_), _ = cv2.getTextSize("SPEAKER", font, 1.2, 3)
cv2.putText(badge, "SPEAKER", ((BW - tw) // 2, band_top + (BH - band_top + th_) // 2),
            font, 1.2, (255, 255, 255), 3)

# ---------------- (e) channel statistics ----------------
B, G, R = cv2.split(square)
df = pd.DataFrame({"Blue": B.ravel(), "Green": G.ravel(), "Red": R.ravel()})
print("\n(e) channel statistics of square crop\n", df.describe().round(2))
print("    dominant channel (highest mean):", df.mean().idxmax())

# ---------------- (f) uint8 arithmetic ----------------
np_add = square + np.uint8(100)              # wraps: 200 + 100 = 44
cv_add = cv2.add(square, np.full_like(square, 100))   # saturates: 200 + 100 = 255
wrapped = int(np.count_nonzero(square.astype(np.int16) + 100 > 255))
print(f"\n(f) values that wrapped in NumPy addition: {wrapped} "
      f"({100 * wrapped / square.size:.1f}% of all values)")
print("    NumPy uint8 arithmetic is modulo 256; cv2.add clips to 255.")

# ---------------- (g) display & save ----------------
cv2.imwrite("badge.png", badge)
show([img, square, mask, circular, np_add, cv_add, badge],
     ["Original", "Square crop 300x300", "Mask", "Circular photo",
      "NumPy +100 (wraps)", "cv2.add +100 (saturates)", "Final badge"], cols=4)
