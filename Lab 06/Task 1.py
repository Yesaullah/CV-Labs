"""
Lab 06 - Task 1: Computer Screen Detection in a Computer Lab (Hough Lines)
Pipeline
  1. Orientation-aware Hough Line Transform: Canny edges are split by gradient
     direction into vertical / horizontal edge maps and HoughLinesP runs on each
     (on the full cluttered edge map, HoughLinesP misses most vertical edges).
  2. Candidate rectangles: (a) rectangular Canny contours, (b) dark blobs
     (bezels / switched-off screens), (c) bright blobs (switched-on screens).
  3. Hough verification: keep a candidate only if >= 3 of its 4 sides are
     supported by Hough segments; snap each side to the nearest Hough line.
  4. ON / OFF status from interior brightness + texture.
  5. Missing screens: group screens into rows, flag unusually large gaps.
     Controlled test: erase one detected screen (inpainting) and re-run.
"""
import os
import cv2
import numpy as np

DATA_DIR = "data"
OUT_DIR = "outputs/task1"
os.makedirs(OUT_DIR, exist_ok=True)
IMAGES = [f"Image_{i}.jpg" for i in range(1, 8)]

ASPECT_RANGE = (1.0, 2.5)        # monitors are landscape
AREA_RANGE = (0.0015, 0.20)      # fraction of image area
SIDE_TOL = 5                     # px tolerance when matching a side to a line
SIDE_MIN_COVER = 0.45            # a side counts as supported above this
MIN_SUPPORTED_SIDES = 3
ON_BRIGHTNESS = 85               # mean V (HSV) of interior
ON_TEXTURE = 28                  # std of gray in interior
GAP_FACTOR = 1.8                 # gap > 1.8 x median spacing => missing screen


# 1. Orientation-aware Hough lines
def hough_lines(gray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 40, 120)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    v_edges = np.where(np.abs(gx) > 2 * np.abs(gy), edges, 0).astype(np.uint8)
    h_edges = np.where(np.abs(gy) > 2 * np.abs(gx), edges, 0).astype(np.uint8)
    min_len = max(15, int(0.012 * max(h, w)))
    lines_v = cv2.HoughLinesP(v_edges, 1, np.pi / 180, threshold=18,
                              minLineLength=min_len, maxLineGap=5)
    lines_h = cv2.HoughLinesP(h_edges, 1, np.pi / 180, threshold=18,
                              minLineLength=min_len, maxLineGap=5)
    lines_v = [] if lines_v is None else lines_v[:, 0]
    lines_h = [] if lines_h is None else lines_h[:, 0]
    return lines_h, lines_v, edges


def line_masks(shape, lines_h, lines_v):
    """Raster masks of H / V Hough lines, thickened by SIDE_TOL across the line."""
    h, w = shape
    hm = np.zeros((h, w), np.uint8)
    vm = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in lines_h:
        cv2.line(hm, (x1, y1), (x2, y2), 255, 1)
    for x1, y1, x2, y2 in lines_v:
        cv2.line(vm, (x1, y1), (x2, y2), 255, 1)
    k = 2 * SIDE_TOL + 1
    return (cv2.dilate(hm, np.ones((k, 1), np.uint8)),
            cv2.dilate(vm, np.ones((1, k), np.uint8)), hm, vm)


# 2. Candidate proposals
def shape_ok(x, y, bw, bh, h, w):
    if bh == 0:
        return False
    a = bw * bh / (h * w)
    return (AREA_RANGE[0] <= a <= AREA_RANGE[1] and
            ASPECT_RANGE[0] <= bw / bh <= ASPECT_RANGE[1])


def contour_candidates(gray):
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    out = []
    for lo, hi in ((30, 90), (50, 150)):
        e = cv2.dilate(cv2.Canny(blur, lo, hi), np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(e, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            hull = cv2.convexHull(c)
            approx = cv2.approxPolyDP(hull, 0.04 * cv2.arcLength(hull, True), True)
            if len(approx) != 4:
                continue
            x, y, bw, bh = cv2.boundingRect(approx)
            if cv2.contourArea(approx) / (bw * bh + 1e-6) < 0.75:
                continue
            if shape_ok(x, y, bw, bh, h, w):
                out.append((x, y, x + bw, y + bh))
    return out


def blob_candidates(img):
    """Rectangular blobs that are very dark (bezels / off screens) or very bright (on screens)."""
    h, w = img.shape[:2]
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2]
    out = []
    for m in [v < 60, v < 80, v > 170, v > 200]:
        blob = cv2.morphologyEx(m.astype(np.uint8) * 255,
                                cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            hull = cv2.convexHull(c)
            (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
            if cv2.contourArea(hull) / (rw * rh + 1e-6) < 0.8:
                continue
            x, y, bw, bh = cv2.boundingRect(hull)
            if shape_ok(x, y, bw, bh, h, w):
                out.append((x, y, x + bw, y + bh))
    return out


# 3. Hough verification + boundary snapping
def side_support(box, hm_d, vm_d):
    x1, y1, x2, y2 = box
    return np.array([hm_d[y1, x1:x2].mean(), hm_d[y2, x1:x2].mean(),
                     vm_d[y1:y2, x1].mean(), vm_d[y1:y2, x2].mean()]) / 255.0


def snap_box(box, hm, vm):
    """Move each side to the row/column with most Hough-line pixels nearby."""
    x1, y1, x2, y2 = box
    r = SIDE_TOL + 3

    def best(profile, base):
        if profile.size == 0 or profile.max() == 0:
            return base
        return base - r + int(np.argmax(profile))

    h, w = hm.shape
    ys = lambda y: slice(max(0, y - r), min(h, y + r + 1))
    xs = lambda x: slice(max(0, x - r), min(w, x + r + 1))
    ny1 = best(hm[ys(y1), x1:x2].sum(1), y1) if y1 - r >= 0 else y1
    ny2 = best(hm[ys(y2), x1:x2].sum(1), y2) if y2 + r < h else y2
    nx1 = best(vm[y1:y2, xs(x1)].sum(0), x1) if x1 - r >= 0 else x1
    nx2 = best(vm[y1:y2, xs(x2)].sum(0), x2) if x2 + r < w else x2
    return nx1, ny1, nx2, ny2


def nms(boxes, scores, overlap=0.3):
    order = sorted(range(len(boxes)),
                   key=lambda i: (-round(scores[i], 1),
                                  -(boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])))
    keep = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        ok = True
        for j in keep:
            a1, b1, a2, b2 = boxes[j]
            inter = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
            smaller = min((x2 - x1) * (y2 - y1), (a2 - a1) * (b2 - b1))
            if inter > overlap * smaller:
                ok = False
                break
        if ok:
            keep.append(i)
    return keep


def detect_screens(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lines_h, lines_v, _ = hough_lines(gray)
    hm_d, vm_d, hm, vm = line_masks(gray.shape, lines_h, lines_v)
    boxes, scores = [], []
    for b in contour_candidates(gray) + blob_candidates(img):
        b = (max(b[0], 0), max(b[1], 0), min(b[2], w - 1), min(b[3], h - 1))
        s = side_support(b, hm_d, vm_d)
        if (s >= SIDE_MIN_COVER).sum() < MIN_SUPPORTED_SIDES:
            continue
        b = snap_box(b, hm, vm)
        if not shape_ok(b[0], b[1], b[2] - b[0], b[3] - b[1], h, w):
            continue
        boxes.append(b)
        scores.append(float(side_support(b, hm_d, vm_d).mean()))
    keep = nms(boxes, scores)
    return [boxes[i] for i in keep], [scores[i] for i in keep], (lines_h, lines_v)


# 4. ON / OFF status
def screen_status(img, box):
    x1, y1, x2, y2 = box
    mx, my = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))   # ignore bezel
    roi = img[y1 + my:y2 - my, x1 + mx:x2 - mx]
    if roi.size == 0:
        return "OFF", 0.0, 0.0
    v = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., 2].mean()
    t = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).std()
    return ("ON" if (v > ON_BRIGHTNESS or t > ON_TEXTURE) else "OFF"), float(v), float(t)


# 5. Missing-screen anomalies
def find_missing(boxes):
    """Group screens into rows (by vertical overlap); flag unusually large gaps."""
    if len(boxes) < 3:
        return []
    rows = []
    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy, bh = (b[1] + b[3]) / 2, b[3] - b[1]
        for row in rows:
            rcy = np.mean([(r[1] + r[3]) / 2 for r in row])
            rh = np.median([r[3] - r[1] for r in row])
            if abs(cy - rcy) < 0.5 * max(bh, rh):
                row.append(b)
                break
        else:
            rows.append([b])
    widths = np.median([b[2] - b[0] for b in boxes])
    anomalies = []
    for row in rows:
        if len(row) < 2:
            continue
        row.sort(key=lambda b: b[0])
        spacing = np.diff([(b[0] + b[2]) / 2 for b in row])
        typical = np.median(spacing) if len(spacing) >= 2 else widths * 1.2
        for i, s in enumerate(spacing):
            if s > GAP_FACTOR * typical:
                left, right = row[i], row[i + 1]
                anomalies.append(dict(box=(int(left[2]), int(min(left[1], right[1])),
                                           int(right[0]), int(max(left[3], right[3]))),
                                      n=max(int(round(s / typical)) - 1, 1)))
    return anomalies


def analyse(img, name, save_lines=True):
    boxes, scores, (lh, lv) = detect_screens(img)
    vis = img.copy()
    if save_lines:
        lines_vis = img.copy()
        for x1, y1, x2, y2 in lh:
            cv2.line(lines_vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for x1, y1, x2, y2 in lv:
            cv2.line(lines_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"hough_lines_{name}"), lines_vis)

    report = []
    for k, b in enumerate(sorted(boxes, key=lambda b: (b[1] // 50, b[0])), 1):
        status, v, t = screen_status(img, b)
        color = (0, 255, 0) if status == "ON" else (0, 140, 255)
        cv2.rectangle(vis, b[:2], b[2:], color, 2)
        cv2.putText(vis, f"S{k} {status}", (b[0], max(b[1] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        report.append((f"S{k}", status, round(v, 1), round(t, 1), b))

    anomalies = find_missing(boxes)
    for a in anomalies:
        x1, y1, x2, y2 = a["box"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"MISSING? x{a['n']}", (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    on = sum(r[1] == "ON" for r in report)
    summary = f"{len(report)} screens | ON {on} | OFF {len(report) - on} | anomalies {len(anomalies)}"
    cv2.putText(vis, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(vis, summary, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, f"screens_{name}"), vis)
    return report, anomalies, summary


def missing_screen_test(name, remove_index):
    """Controlled anomaly test: erase one detected screen and re-run detection."""
    img = cv2.imread(os.path.join(DATA_DIR, name))
    boxes, _, _ = detect_screens(img)
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))
    if not boxes:
        return [], [], "no screens to remove"
    x1, y1, x2, y2 = boxes[min(remove_index, len(boxes) - 1)]
    mask = np.zeros(img.shape[:2], np.uint8)
    pad = 8
    mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 255
    erased = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    return analyse(erased, f"missing_test_{name}", save_lines=False)


MISSING_TESTS = [("Image_6.jpg", 5), ("Image_5.jpg", 3)]


def main():
    for name in IMAGES:
        img = cv2.imread(os.path.join(DATA_DIR, name))
        report, anomalies, summary = analyse(img, name)
        print(f"\n{name}: {summary}")
        for sid, status, v, t, b in report:
            print(f"   {sid:<4} {status:<3}  brightness={v:<6} texture={t:<6} box={b}")
    print("\n=== Controlled missing-screen test ===")
    for name, idx in MISSING_TESTS:
        _, anomalies, summary = missing_screen_test(name, idx)
        print(f"{name} with screen #{idx + 1} removed: {summary}")


if __name__ == "__main__":
    main()
