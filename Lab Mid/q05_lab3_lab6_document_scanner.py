"""
=====================================================================
Q5  |  LAB 3 + LAB 6 — Perspective (homography), affine from 3 points,
                       DLT by hand, contours, SIFT + RANSAC
     "Mobile Document Scanner"
=====================================================================
SCENARIO
  A user photographs an exam sheet at an angle. Produce a flat, top-down
  scan — first with manually given corners, then fully automatically.

TASKS
  (a) Write order_points(pts) that sorts 4 unordered corners into
      TL, TR, BR, BL (hint: x+y is min at TL / max at BR; y-x is min at TR /
      max at BL). Compute the output width/height from the edge lengths
      and rectify the page with getPerspectiveTransform + warpPerspective.
  (b) AUTO CORNERS: threshold the photo, find the largest external contour,
      approximate it with approxPolyDP to 4 vertices, rectify again.
  (c) Solve the homography BY HAND (8 unknowns, 8 equations from 4 point
      pairs, h33 = 1) with np.linalg.solve; compare to OpenCV's matrix.
  (d) Use only 3 corners and getAffineTransform. Measure where the 4th
      corner lands (pixel error) and explain why an affine transform
      CANNOT undo perspective.
  (e) SIFT + BFMatcher + Lowe ratio test (0.75) + findHomography(RANSAC)
      between the clean template and the photo. Report matches, inliers,
      and the mean corner error of the projected template outline.
  (f) Display: photo with detected corners, all rectified results,
      SIFT matches, and the projected outline.

CONCEPTS TESTED
  4 points -> homography (8 DOF), 3 points -> affine (6 DOF), float32
  point arrays, point order consistency, dsize=(W,H), findContours /
  approxPolyDP, SIFT 128-D descriptors, ratio test, RANSAC inliers,
  perspectiveTransform for points.

HOW TO RUN
  Self-contained: a synthetic template + its perspective "photo" are
  generated (ground-truth corners known). To use real photos, set
  PHOTO_PATH and TEMPLATE_PATH.
=====================================================================
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

PHOTO_PATH = "photo.jpg"
TEMPLATE_PATH = "template.jpg"


def rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im


def make_template():
    rng = np.random.default_rng(3)
    doc = np.full((560, 400, 3), 245, np.uint8)
    cv2.putText(doc, "CV LAB EXAM", (60, 60), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
    for i in range(12):
        y = 110 + i * 32
        cv2.putText(doc, f"Q{i + 1}.", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.line(doc, (80, y), (int(rng.integers(220, 370)), y), (60, 60, 60), 2)
    for _ in range(25):                                   # texture -> more SIFT keypoints
        x, y = rng.integers(30, 370), rng.integers(500, 540)
        cv2.circle(doc, (int(x), int(y)), int(rng.integers(3, 9)), (0, 0, 0), -1)
    cv2.rectangle(doc, (280, 20), (380, 80), (0, 0, 180), 3)
    return doc


def make_photo(doc):
    h, w = doc.shape[:2]
    gt = np.float32([[160, 90], [520, 140], [560, 590], [110, 540]])  # TL TR BR BL
    H = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), gt)
    bg = np.full((680, 700, 3), (60, 80, 100), np.uint8)
    bg = cv2.add(bg, np.random.default_rng(4).integers(0, 25, bg.shape, dtype=np.uint8))
    warped = cv2.warpPerspective(doc, H, (700, 680))
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), H, (700, 680))
    bg[m > 0] = warped[m > 0]
    return bg, gt


template = cv2.imread(TEMPLATE_PATH) if os.path.exists(TEMPLATE_PATH) else None
photo = cv2.imread(PHOTO_PATH) if os.path.exists(PHOTO_PATH) else None
gt_corners = None
if template is None or photo is None:
    print("[info] using synthetic template + perspective photo")
    template = make_template()
    photo, gt_corners = make_photo(template)


# ---------------- (a) order + rectify ----------------
def order_points(pts):
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def rectify(img, pts):
    tl, tr, br, bl = order_points(pts)
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    P = cv2.getPerspectiveTransform(order_points(pts), dst)
    return cv2.warpPerspective(img, P, (W, H)), P, dst


manual_pts = gt_corners if gt_corners is not None else np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
shuffled = manual_pts[[2, 0, 3, 1]]                       # deliberately unordered
print("(a) ordered corners:\n", order_points(shuffled))
scan_manual, P_cv, dst = rectify(photo, shuffled)

# ---------------- (b) automatic corners ----------------
g = cv2.GaussianBlur(cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), (5, 5), 0)
_, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))   # fill the text
cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
page = max(cnts, key=cv2.contourArea)
approx = cv2.approxPolyDP(page, 0.02 * cv2.arcLength(page, True), True)
if len(approx) != 4:
    raise RuntimeError(f"expected 4 corners, got {len(approx)} — tune epsilon/threshold")
auto_pts = order_points(approx)
scan_auto, _, _ = rectify(photo, auto_pts)
print("(b) auto corners:\n", auto_pts)
if gt_corners is not None:
    print(f"    mean error vs ground truth = {np.linalg.norm(auto_pts - gt_corners, axis=1).mean():.2f} px")


# ---------------- (c) DLT by hand ----------------
def homography_dlt(src, dst):
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    h8 = np.linalg.solve(np.array(A, float), np.array(b, float))
    return np.append(h8, 1).reshape(3, 3)


H_manual = homography_dlt(order_points(shuffled), dst)
print("(c) manual H == getPerspectiveTransform:", np.allclose(H_manual, P_cv, atol=1e-6))

# ---------------- (d) affine from 3 points ----------------
src3 = order_points(shuffled)
A = cv2.getAffineTransform(src3[[0, 1, 3]], dst[[0, 1, 3]])          # TL, TR, BL
scan_affine = cv2.warpAffine(photo, A, (scan_manual.shape[1], scan_manual.shape[0]))
br_pred = A @ np.append(src3[2], 1)
print(f"(d) affine sends BR corner to {br_pred.round(1)} instead of {dst[2]} "
      f"-> error {np.linalg.norm(br_pred - dst[2]):.1f} px")
print("    Affine keeps parallel lines parallel; a perspective photo has converging "
      "edges, so 6 DOF cannot fit 4 corners (needs 8 DOF).")

# ---------------- (e) SIFT + RANSAC ----------------
sift = cv2.SIFT_create()
gt_ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
gp_ = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
k1, d1 = sift.detectAndCompute(gt_, None)
k2, d2 = sift.detectAndCompute(gp_, None)
knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
print(f"(e) keypoints template/photo = {len(k1)}/{len(k2)}, good matches = {len(good)}")
outline = photo.copy()
match_img = None
if len(good) >= 4:
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstp = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    Hs, inl = cv2.findHomography(src, dstp, cv2.RANSAC, 5.0)
    th_, tw_ = gt_.shape
    corners = np.float32([[0, 0], [tw_, 0], [tw_, th_], [0, th_]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, Hs).reshape(4, 2)
    cv2.polylines(outline, [np.int32(proj)], True, (0, 255, 0), 4)
    print(f"    RANSAC inliers = {int(inl.sum())}/{len(good)}")
    if gt_corners is not None:
        print(f"    mean corner error = {np.linalg.norm(proj - gt_corners, axis=1).mean():.2f} px")
    match_img = cv2.drawMatches(template, k1, photo, k2, [m for m, f in zip(good, inl.ravel()) if f][:60],
                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
else:
    print("    not enough matches for a homography")

# ---------------- (f) display ----------------
vis = photo.copy()
for i, (x, y) in enumerate(auto_pts):
    cv2.circle(vis, (int(x), int(y)), 10, (0, 0, 255), -1)
    cv2.putText(vis, ["TL", "TR", "BR", "BL"][i], (int(x) + 12, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
panels = [(vis, "Photo + auto corners"), (th, "Otsu + close"), (scan_manual, "(a) 4-point scan"),
          (scan_auto, "(b) auto scan"), (scan_affine, "(d) affine (3 pts) — skewed"),
          (outline, "(e) SIFT+RANSAC outline")]
plt.figure(figsize=(18, 10))
for i, (im, t) in enumerate(panels):
    plt.subplot(2, 3, i + 1); plt.imshow(rgb(im), cmap="gray"); plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
if match_img is not None:
    plt.figure(figsize=(14, 7)); plt.imshow(rgb(match_img)); plt.title("Inlier SIFT matches")
    plt.axis("off"); plt.show()
