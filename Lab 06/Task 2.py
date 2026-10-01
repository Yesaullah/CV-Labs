"""
Lab 06 - Task 2: Asset Tracking in a Computer Lab Using SIFT
Pipeline: template library -> SIFT -> BF kNN + Lowe ratio -> RANSAC homography
-> iterative multi-instance detection -> robustness tests -> inventory CSV.
SIFT matches specific instances (same model/appearance), not categories.
"""
import os, csv
import cv2
import numpy as np

DATA_DIR = "data"
OUT_DIR = "outputs/task2"
os.makedirs(OUT_DIR, exist_ok=True)

# asset name -> (source image, crop box x1,y1,x2,y2)
TEMPLATES = {
    "Acer Keyboard":    ("Image_2.jpg", (485, 440, 830, 700)),
    "Acer CPU":         ("Image_2.jpg", (870, 430, 1270, 900)),
    "HP Monitor":       ("Image_5.jpg", (608, 508, 848, 672)),
    "Dell CRT Monitor": ("Image_1.jpg", (645, 525, 990, 820)),
    "Dell Desktop CPU": ("Image_1.jpg", (620, 780, 1020, 950)),
}
SCENES = [f"Image_{i}.jpg" for i in range(1, 8)]

RATIO, MIN_MATCHES, MIN_INLIERS, MAX_INSTANCES = 0.75, 12, 10, 10

sift = cv2.SIFT_create()
matcher = cv2.BFMatcher(cv2.NORM_L2)   # exact + deterministic (FLANN is randomised)


def build_library():
    lib = {}
    for name, (fname, (x1, y1, x2, y2)) in TEMPLATES.items():
        img = cv2.imread(os.path.join(DATA_DIR, fname))
        crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
        kp, des = sift.detectAndCompute(crop, None)
        lib[name] = dict(kp=kp, des=des, shape=crop.shape)
        cv2.imwrite(os.path.join(OUT_DIR, f"template_{name.replace(' ', '_')}.jpg"),
                    img[y1:y2, x1:x2])
        print(f"[template] {name:<17} from {fname}: {len(kp)} keypoints")
    return lib


def valid_quad(quad, scene_shape, tmpl_area):
    """Reject degenerate homographies (twisted, tiny, huge, off-image)."""
    q = quad.reshape(4, 2)
    if not cv2.isContourConvex(q.astype(np.int32)):
        return False
    area = cv2.contourArea(q.astype(np.float32))
    h, w = scene_shape[:2]
    if area < 0.002 * h * w or area > 0.9 * h * w:
        return False
    if not 0.01 <= area / tmpl_area <= 10:
        return False
    sides = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
    if min(sides) < 0.15 * max(sides):
        return False
    inside = np.sum((q[:, 0] > -0.1 * w) & (q[:, 0] < 1.1 * w) &
                    (q[:, 1] > -0.1 * h) & (q[:, 1] < 1.1 * h))
    return inside == 4


def detect_instances(tmpl, kp_s, des_s, scene_shape):
    """Find all instances of one template (iterative RANSAC, removing used keypoints)."""
    dets = []
    if tmpl["des"] is None or des_s is None or len(kp_s) < 2:
        return dets
    th, tw = tmpl["shape"]
    corners = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]]).reshape(-1, 1, 2)
    active = np.ones(len(kp_s), dtype=bool)
    for _ in range(MAX_INSTANCES):
        idx = np.where(active)[0]
        if len(idx) < MIN_MATCHES:
            break
        knn = matcher.knnMatch(tmpl["des"], des_s[idx], k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < RATIO * n.distance]
        if len(good) < MIN_MATCHES:
            break
        src = np.float32([tmpl["kp"][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_s[idx[m.trainIdx]].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            break
        inliers = int(mask.sum())
        quad = cv2.perspectiveTransform(corners, H)
        poly = quad.reshape(-1, 2).astype(np.float32)
        for j in idx:                                   # free this region for next search
            if cv2.pointPolygonTest(poly, kp_s[j].pt, False) >= 0:
                active[j] = False
        active[[idx[good[i].trainIdx] for i in range(len(good)) if mask[i]]] = False
        if inliers < MIN_INLIERS:
            break
        if valid_quad(quad, scene_shape, tw * th):
            dets.append(dict(quad=quad, inliers=inliers))
    return dets


COLORS = {"Acer Keyboard": (0, 255, 0), "Acer CPU": (255, 0, 255),
          "HP Monitor": (0, 165, 255), "Dell CRT Monitor": (255, 255, 0),
          "Dell Desktop CPU": (0, 0, 255)}


def process_scene(name, img, lib, inventory, counter):
    kp_s, des_s = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
    vis, found = img.copy(), {}
    for asset, tmpl in lib.items():
        dets = sorted(detect_instances(tmpl, kp_s, des_s, img.shape),
                      key=lambda d: d["quad"][:, 0, 0].mean())
        for d in dets:
            counter[asset] = counter.get(asset, 0) + 1
            aid = f"{''.join(w[0] for w in asset.split())}-{counter[asset]:02d}"
            q = np.int32(d["quad"])
            cv2.polylines(vis, [q], True, COLORS[asset], 3)
            x, y = q.reshape(-1, 2).min(axis=0)
            cv2.putText(vis, f"{aid} ({d['inliers']})", (int(x), max(int(y) - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[asset], 2)
            cx, cy = d["quad"].reshape(-1, 2).mean(axis=0)
            inventory.append([aid, asset, name, int(cx), int(cy), d["inliers"]])
        found[asset] = len(dets)
    cv2.imwrite(os.path.join(OUT_DIR, f"detected_{name}"), vis)
    return found


def rotate(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, M, (nw, nh))


def occlude(img, frac=0.3, seed=0):
    rng, out = np.random.default_rng(seed), img.copy()
    h, w = img.shape[:2]
    for _ in range(4):
        bw, bh = int(w * frac / 2), int(h * frac / 2)
        x, y = rng.integers(0, w - bw), rng.integers(0, h - bh)
        cv2.rectangle(out, (int(x), int(y)), (int(x + bw), int(y + bh)), (40, 40, 40), -1)
    return out


TRANSFORMS = {"rot30": lambda im: rotate(im,
