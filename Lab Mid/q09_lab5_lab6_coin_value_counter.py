"""
=====================================================================
Q9  |  LAB 5 + LAB 6 — Watershed, distance transform, contours,
                       Hough circles, HSV colour classification
     "Coin Value Counter for a Vending Machine"
=====================================================================
SCENARIO
  A tray contains gold, silver and copper coins, several of them TOUCHING.
  Count the coins, classify each by colour and compute the total value.
  Values: gold = Rs 10, silver = Rs 5, copper = Rs 1.

TASKS
  (a) Segment coins from the background (gray -> blur -> Otsu, then
      opening). Count coins NAIVELY with external contours and show why
      touching coins are merged.
  (b) Separate touching coins with the full WATERSHED pipeline:
      sure background (dilate), distance transform, sure foreground
      (fraction of max), unknown region, connectedComponents markers,
      cv2.watershed. Draw boundaries in red. Report the count for
      distance fractions {0.3, 0.5, 0.7} in a table.
  (c) Count coins again with cv2.HoughCircles for param2 in {15, 25, 35}
      and explain the trend.
  (d) For every watershed region compute mean H, S, V (HSV) inside its mask
      and classify: low saturation -> silver; else hue < 15 -> copper;
      else gold. Print a per-coin table (label, area, H, S, V, class).
  (e) Annotate the image with each coin's class and print/draw the
      TOTAL VALUE. Compare the watershed count vs the Hough count.

CONCEPTS TESTED
  Otsu, morphology (open/dilate), distanceTransform, markers + 1 and
  unknown = 0, boundaries = -1, watershed needs 3-channel image + int32
  markers, Hough param2 = accumulator threshold, HSV ranges in OpenCV
  (H 0-179), per-region statistics with masks.
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_PATH = "coins.jpg"
VALUES = {"gold": 10, "silver": 5, "copper": 1}


def show(imgs, titles, cols=4):
    rows = (len(imgs) + cols - 1) // cols
    plt.figure(figsize=(4.5 * cols, 3.8 * rows))
    for i, (im, t) in enumerate(zip(imgs, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2RGB), cmap="gray")
        plt.title(t, fontsize=9); plt.axis("off")
    plt.tight_layout(); plt.show()


def synthetic_coins():
    rng = np.random.default_rng(21)
    img = np.full((420, 600, 3), (40, 70, 35), np.uint8)            # dark green felt
    coins = [((110, 110), 48, "gold"), ((203, 118), 46, "silver"),  # touching pair
             ((300, 300), 40, "copper"), ((378, 300), 40, "copper"),  # touching pair
             ((480, 120), 52, "silver"), ((140, 300), 44, "gold"),
             ((470, 330), 36, "copper"), ((330, 110), 42, "gold")]
    colours = {"gold": (40, 180, 225), "silver": (195, 195, 200), "copper": (50, 95, 190)}
    for (c, r, k) in coins:
        cv2.circle(img, c, r, colours[k], -1)
        cv2.circle(img, (c[0] - r // 3, c[1] - r // 3), r // 4,
                   tuple(min(255, v + 40) for v in colours[k]), -1)  # highlight
        cv2.circle(img, c, r - 6, tuple(max(0, v - 35) for v in colours[k]), 2)  # rim
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    return img, coins


img = cv2.imread(IMAGE_PATH) if os.path.exists(IMAGE_PATH) else None
truth = None
if img is None:
    print("[info] using synthetic coin tray")
    img, truth = synthetic_coins()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------- (a) segmentation + naive count ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)
T, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if th[0, 0] == 255:                                   # coins must be WHITE
    th = cv2.bitwise_not(th)
k3 = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, k3, iterations=2)
opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, k3, iterations=2)   # fill highlight holes
cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = [c for c in cnts if cv2.contourArea(c) > 300]
naive = img.copy(); cv2.drawContours(naive, cnts, -1, (255, 0, 255), 2)
print(f"(a) Otsu T = {T:.0f}; naive contour count = {len(cnts)} "
      f"(touching coins form one blob)" + (f"; true count = {len(truth)}" if truth else ""))


# ---------------- (b) watershed ----------------
def watershed_count(frac):
    sure_bg = cv2.dilate(opening, k3, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, frac * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    n, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1                     # background -> 1 (0 is reserved for unknown)
    markers[unknown == 255] = 0
    markers = cv2.watershed(img.copy(), markers)   # 3-channel image, int32 markers
    return n - 1, markers, dist, sure_bg, sure_fg, unknown


rows = []
for f in [0.3, 0.5, 0.7]:
    n, mk, *_ = watershed_count(f)
    rows.append({"fraction": f, "markers": n, "regions": len(np.unique(mk)) - 2})
print("(b) distance-fraction study\n", pd.DataFrame(rows).to_string(index=False))
FRAC = 0.5
n_ws, markers, dist, sure_bg, sure_fg, unknown = watershed_count(FRAC)
ws_vis = img.copy(); ws_vis[markers == -1] = (0, 0, 255)

# ---------------- (c) Hough circles ----------------
mb = cv2.medianBlur(gray, 5)
hough_vis, hough_counts = img.copy(), {}
for p2 in [15, 25, 35]:
    circles = cv2.HoughCircles(mb, cv2.HOUGH_GRADIENT, dp=1.2, minDist=55,
                               param1=100, param2=p2, minRadius=25, maxRadius=65)
    hough_counts[p2] = 0 if circles is None else circles.shape[1]
    if p2 == 25 and circles is not None:
        for x, y, r in np.uint16(np.around(circles[0])):
            cv2.circle(hough_vis, (x, y), r, (0, 255, 0), 2)
            cv2.circle(hough_vis, (x, y), 2, (0, 0, 255), 3)
print("(c) Hough counts by param2:", hough_counts,
      "-> lower param2 = fewer votes needed = more (incl. false) circles")

# ---------------- (d) colour classification ----------------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
records = []
for lab in np.unique(markers):
    if lab in (-1, 1):                         # boundary / background
        continue
    m = (markers == lab).astype(np.uint8)
    area = int(m.sum())
    if area < 300:
        continue
    m = cv2.erode(m, k3, iterations=2)        # avoid boundary pixels
    H, S, V = [float(hsv[..., i][m > 0].mean()) for i in range(3)]
    cls = "silver" if S < 60 else ("copper" if H < 15 else "gold")
    ys, xs = np.nonzero(m)
    records.append({"label": int(lab), "area": area, "H": round(H, 1), "S": round(S, 1),
                    "V": round(V, 1), "class": cls, "cx": int(xs.mean()), "cy": int(ys.mean())})
df = pd.DataFrame(records)
print("(d) per-coin table\n", df.drop(columns=["cx", "cy"]).to_string(index=False))

# ---------------- (e) total value ----------------
final = img.copy(); final[markers == -1] = (0, 0, 255)
for _, r in df.iterrows():
    cv2.putText(final, f"{r['class']}", (r.cx - 30, r.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(final, f"{r['class']}", (r.cx - 30, r.cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
counts = df["class"].value_counts().to_dict()
total = sum(VALUES[c] * n for c, n in counts.items())
cv2.putText(final, f"Coins: {len(df)}  Total: Rs {total}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
print(f"(e) counts = {counts} -> TOTAL = Rs {total}")
print(f"    watershed count = {len(df)}, Hough (param2=25) = {hough_counts[25]}")
if truth:
    true_total = sum(VALUES[k] for *_, k in truth)
    print(f"    ground truth: {len(truth)} coins, Rs {true_total}")

dist_vis = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
show([img, opening, naive, dist_vis, sure_fg, unknown, ws_vis, hough_vis, final],
     ["Input", "Otsu + morphology", f"Naive contours: {len(cnts)}", "Distance transform",
      f"Sure FG ({FRAC}*max)", "Unknown", "Watershed boundaries", "Hough param2=25",
      f"Classified, Rs {total}"], cols=3)
