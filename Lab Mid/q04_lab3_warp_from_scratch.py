"""
=====================================================================
Q4  |  LAB 3 — Homogeneous matrices, composition, inverse mapping,
                no-crop canvas, transform classification
     "Build Your Own warpAffine"
=====================================================================
SCENARIO
  Your embedded device has no OpenCV warp functions. Implement geometric
  transforms yourself with NumPy and verify them against OpenCV.

TASKS
  (a) Write 3x3 homogeneous builders T(tx,ty), S(sx,sy), R(deg), Sh(shx,shy)
      and about_center(M, cx, cy).
  (b) Build ONE composite matrix that: scales by 0.8 about the centre,
      THEN rotates 30 deg (counter-clockwise on screen) about the centre,
      THEN translates by (+40, +20). Print it and its determinant.
  (c) Implement warp_manual(img, M, (W,H)) using INVERSE MAPPING with
      nearest-neighbour interpolation (vectorised, no Python pixel loops).
      Compare with cv2.warpAffine(..., INTER_NEAREST): % identical pixels.
  (d) Implement fit_canvas(M, w, h): transform the 4 image corners, compute
      the bounding box and return an adjusted matrix + canvas size so NO
      corner is cut. Apply to (shear 0.4 then rotate 45 deg).
  (e) Prove numerically that cv2.getRotationMatrix2D(center, +theta, 1)
      equals the textbook R(-theta) applied about the centre.
  (f) Write classify(M) returning one of: identity, translation, rigid,
      similarity, affine, projective (+ "with reflection" if det<0).
      Test it on six matrices.
  (g) Map a set of points (a square) through each transform and plot them
      to visualise what each class preserves.

CONCEPTS TESTED
  homogeneous coordinates, right-most matrix applied first, rotation
  sign convention (y axis points down), forward vs inverse mapping (holes),
  dsize = (W, H), determinant = area scale, DOF hierarchy.
=====================================================================
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "image.jpg"


def synthetic():
    img = np.full((300, 400, 3), 230, np.uint8)
    for x in range(0, 400, 40):
        cv2.line(img, (x, 0), (x, 299), (180, 180, 180), 1)
    for y in range(0, 300, 40):
        cv2.line(img, (0, y), (399, y), (180, 180, 180), 1)
    cv2.rectangle(img, (40, 40), (180, 140), (0, 0, 200), -1)
    cv2.circle(img, (290, 190), 60, (200, 120, 0), -1)
    cv2.putText(img, "TOP", (160, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
    return img


img = cv2.imread(IMAGE_PATH) if os.path.exists(IMAGE_PATH) else None
if img is None:
    print("[info] using synthetic grid image")
    img = synthetic()
h, w = img.shape[:2]
cx, cy = w / 2, h / 2


# ---------------- (a) builders ----------------
def T(tx, ty):        return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)
def S(sx, sy):        return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)
def Sh(shx=0, shy=0): return np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]], float)


def R(deg):  # textbook matrix; with y pointing DOWN it turns CLOCKWISE on screen
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def about_center(M, cx, cy):
    return T(cx, cy) @ M @ T(-cx, -cy)


# ---------------- (b) composite ----------------
# order of application: scale -> rotate -> translate  => write right-to-left
M = T(40, 20) @ about_center(R(-30), cx, cy) @ about_center(S(0.8, 0.8), cx, cy)
np.set_printoptions(precision=4, suppress=True)
print("(b) composite matrix:\n", M)
print("    det of 2x2 part =", round(np.linalg.det(M[:2, :2]), 4), "(= 0.8^2 = area scale)")


# ---------------- (c) inverse-mapping warp ----------------
def warp_manual(src, M, size):
    W, H = size
    Minv = np.linalg.inv(M)
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))                 # destination grid
    dst_pts = np.stack([xs.ravel(), ys.ravel(), np.ones(W * H)])     # 3 x N
    sx, sy, sw = Minv @ dst_pts
    sx, sy = sx / sw, sy / sw                                        # works for projective too
    sxi, syi = np.floor(sx + 0.5).astype(int), np.floor(sy + 0.5).astype(int)   # nearest neighbour
    valid = (sxi >= 0) & (sxi < src.shape[1]) & (syi >= 0) & (syi < src.shape[0])
    out = np.zeros((H, W) + src.shape[2:], src.dtype)
    out.reshape(H * W, -1)[valid] = src[syi[valid], sxi[valid]].reshape(valid.sum(), -1)
    return out


mine = warp_manual(img, M, (w, h))
ref = cv2.warpAffine(img, M[:2].astype(np.float32), (w, h), flags=cv2.INTER_NEAREST)
same = np.all(mine == ref, axis=2).mean() * 100
print(f"(c) manual vs cv2.warpAffine (nearest): {same:.2f}% identical pixels "
      "(tiny differences = rounding at .5 boundaries)")

# why inverse mapping? forward mapping (src -> dst) leaves HOLES when enlarging:
ys, xs = np.mgrid[0:h, 0:w]
Sbig = about_center(S(1.7, 1.7), cx, cy)
fwd_big = np.zeros_like(img)
p = Sbig @ np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)])
u, v = np.round(p[0]).astype(int), np.round(p[1]).astype(int)
ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
fwd_big[v[ok], u[ok]] = img[ys.ravel()[ok], xs.ravel()[ok]]


# ---------------- (d) no-crop canvas ----------------
def fit_canvas(M, w, h):
    corners = np.array([[0, w - 1, w - 1, 0], [0, 0, h - 1, h - 1], [1, 1, 1, 1]], float)
    c = M @ corners
    c = c[:2] / c[2]
    xmin, ymin = c.min(axis=1)
    xmax, ymax = c.max(axis=1)
    return T(-xmin, -ymin) @ M, (int(np.ceil(xmax - xmin)) + 1, int(np.ceil(ymax - ymin)) + 1)


M2 = R(-45) @ Sh(0.4, 0)                       # shear first, then rotate
M2_fit, size2 = fit_canvas(M2, w, h)
cropped = warp_manual(img, M2, (w, h))
full = warp_manual(img, M2_fit, size2)
print(f"(d) shear+rotate canvas: {size2} instead of {(w, h)}")

# ---------------- (e) OpenCV rotation sign ----------------
theta = 30
cvR = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
mine_R = about_center(R(-theta), cx, cy)[:2]
print(f"(e) getRotationMatrix2D(+{theta}) == about_center(R(-{theta})):",
      np.allclose(cvR, mine_R))


# ---------------- (f) classification ----------------
def classify(M, tol=1e-6):
    M = np.asarray(M, float)
    M = M / M[2, 2]
    if not np.allclose(M[2, :2], 0, atol=tol):
        return "projective (homography)"
    A, t = M[:2, :2], M[:2, 2]
    det = np.linalg.det(A)
    refl = " with reflection" if det < 0 else ""
    if np.allclose(A, np.eye(2), atol=tol):
        return "identity" if np.allclose(t, 0, atol=tol) else "translation"
    AtA = A.T @ A
    if np.allclose(AtA, np.eye(2), atol=1e-4):
        return "rigid (Euclidean)" + refl
    if np.allclose(AtA, AtA[0, 0] * np.eye(2), atol=1e-4):
        return f"similarity (scale {np.sqrt(AtA[0, 0]):.3f})" + refl
    return "affine" + refl


tests = {
    "T(10,5)": T(10, 5),
    "R(20)+T": T(5, 5) @ R(20),
    "composite (b)": M,
    "shear+rotate (d)": M2,
    "reflect-x": np.diag([1.0, -1, 1]),
    "perspective": np.array([[1, 0.1, 0], [0, 1, 0], [0.001, 0.0005, 1]]),
}
print("(f) classification")
for name, mat in tests.items():
    print(f"    {name:18s} -> {classify(mat)}")

# ---------------- (g) plots ----------------
sq = np.array([[0, 100, 100, 0, 0], [0, 0, 100, 100, 0], [1, 1, 1, 1, 1]], float)
plt.figure(figsize=(6, 6))
plt.plot(sq[0], sq[1], "k-", lw=3, label="original")
for name, mat in tests.items():
    q = mat @ sq
    plt.plot(q[0] / q[2], q[1] / q[2], label=name)
plt.gca().invert_yaxis(); plt.axis("equal"); plt.legend(fontsize=8)
plt.title("Unit square through each transform (y axis down)"); plt.show()

fig = plt.figure(figsize=(16, 7))
for i, (im, t) in enumerate([(img, "Original"), (mine, "(b) manual warp"),
                             (ref, "(b) cv2.warpAffine"), (fwd_big, "Forward map x1.7 -> HOLES"),
                             (cropped, "(d) shear+rot cropped"), (full, "(d) fit_canvas")]):
    plt.subplot(2, 3, i + 1); plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title(t); plt.axis("off")
plt.tight_layout(); plt.show()
