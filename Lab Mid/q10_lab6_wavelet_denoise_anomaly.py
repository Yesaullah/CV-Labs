"""
=====================================================================
Q10 |  LAB 6 — Wavelets: 1-D DWT denoising, anomaly detection,
                wavelet comparison, 2-D DWT image denoising & compression
     "Smart-Factory Vibration Monitor + Camera Compression"
=====================================================================
SCENARIO
  A vibration sensor on a motor streams a noisy signal with occasional
  faults (spikes). Clean it, flag the faults, and then reuse the same
  wavelet ideas to denoise and compress frames from the factory camera.

TASKS
  (a) Load a signal (CSV column 'value') or generate one:
      trend + 2 sinusoids + Gaussian noise + 5 injected spikes.
  (b) Decompose with pywt.wavedec('db4', level = pywt.dwt_max_level);
      print each coefficient array's name and length (cA_n, cD_n ... cD1).
  (c) Denoise with the UNIVERSAL threshold
          sigma = median(|cD1|) / 0.6745,  T = sigma * sqrt(2 ln N)
      using SOFT and HARD thresholding (approximation untouched). Report
      SNR before/after for both (signal is synthetic, so the clean truth
      is known).
  (d) Compare wavelets haar, db4, sym8, coif3 (soft) in an RMSE table.
  (e) Anomalies: residual = noisy - denoised; flag |residual - median| >
      3 * robust_std (robust_std = 1.4826 * MAD). Compute precision and
      recall against the injected indices (+-2 samples tolerance).
  (f) 2-D: one-level dwt2 ('haar') of an image -> show LL, LH, HL, HH.
      Then 3-level wavedec2 denoising (soft, universal threshold from the
      HH1 band) and report PSNR noisy vs denoised.
  (g) Compression: keep only the largest 5 % of all 2-D coefficients
      (others set to 0), reconstruct, report PSNR and the fraction kept.

CONCEPTS TESTED
  approximation vs detail coefficients, multi-resolution, why noise lives
  in the finest details, soft vs hard thresholding, waverec length trim,
  robust statistics for anomaly thresholds, LL/LH/HL/HH sub-bands,
  energy compaction (why JPEG2000 uses wavelets).
=====================================================================
"""
import os
import cv2
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

CSV_PATH = "sensor.csv"
IMAGE_PATH = "factory.jpg"

# ---------------- (a) signal ----------------
rng = np.random.default_rng(42)
N = 2048
t = np.linspace(0, 20, N)
clean = None
if os.path.exists(CSV_PATH):
    noisy = pd.read_csv(CSV_PATH)["value"].to_numpy(float)
    N = len(noisy); t = np.arange(N); true_idx = np.array([], int)
    print("[info] real data: SNR and precision/recall need ground truth -> skipped")
else:
    clean = 0.05 * t + np.sin(2 * np.pi * 0.5 * t) + 0.4 * np.sin(2 * np.pi * 2.0 * t)
    noisy = clean + rng.normal(0, 0.3, N)
    true_idx = np.array([300, 701, 1150, 1555, 1900])
    noisy[true_idx] += np.array([3.0, -2.5, 3.5, -3.0, 2.8])


def snr_db(ref, x):
    return 10 * np.log10(np.sum(ref ** 2) / np.sum((ref - x) ** 2))


# ---------------- (b) decomposition ----------------
wav = "db4"
level = pywt.dwt_max_level(N, pywt.Wavelet(wav).dec_len)
level = min(level, 6)
coeffs = pywt.wavedec(noisy, wav, level=level)
names = [f"cA{level}"] + [f"cD{level - i}" for i in range(level)]
print("(b) " + ", ".join(f"{n}:{len(c)}" for n, c in zip(names, coeffs)))


# ---------------- (c) universal threshold ----------------
def wavelet_denoise(x, wav="db4", level=6, mode="soft"):
    c = pywt.wavedec(x, wav, level=level)
    sigma = np.median(np.abs(c[-1])) / 0.6745             # noise estimate from finest detail
    T = sigma * np.sqrt(2 * np.log(len(x)))
    c = [c[0]] + [pywt.threshold(d, T, mode=mode) for d in c[1:]]
    return pywt.waverec(c, wav)[:len(x)], T                 # trim: may be 1 longer


soft, T = wavelet_denoise(noisy, wav, level, "soft")
hard, _ = wavelet_denoise(noisy, wav, level, "hard")
print(f"(c) universal threshold T = {T:.3f}")
if clean is not None:
    print(f"    SNR noisy = {snr_db(clean, noisy):.2f} dB | soft = {snr_db(clean, soft):.2f} dB | "
          f"hard = {snr_db(clean, hard):.2f} dB")
    print("    soft shrinks every surviving coefficient by T -> smoother; hard keeps them "
          "intact -> sharper but with small artefacts")

# ---------------- (d) wavelet comparison ----------------
if clean is not None:
    rmse = {w: np.sqrt(np.mean((clean - wavelet_denoise(noisy, w, level)[0]) ** 2))
            for w in ["haar", "db4", "sym8", "coif3"]}
    print("(d) RMSE by wavelet (soft):\n", pd.Series(rmse).round(4).to_string())

# ---------------- (e) anomalies ----------------
residual = noisy - soft
mad = np.median(np.abs(residual - np.median(residual)))
robust_std = 1.4826 * mad
flags = np.where(np.abs(residual - np.median(residual)) > 3 * robust_std)[0]
# merge flags that belong to the same event (within 10 samples)
events = [g[np.argmax(np.abs(residual[g]))] for g in np.split(flags, np.where(np.diff(flags) > 10)[0] + 1) if len(g)]
events = np.array(events, int)
print(f"(e) flagged samples = {len(flags)}, merged events = {len(events)}: {events.tolist()}")
if len(true_idx):
    tp = sum(np.any(np.abs(true_idx - e) <= 2) for e in events)
    rec = sum(np.any(np.abs(events - i) <= 2) for i in true_idx)
    print(f"    precision = {tp / max(len(events), 1):.2f}, recall = {rec / len(true_idx):.2f}")
    print("    (3-sigma on Gaussian noise alone flags ~0.3 % of samples -> some false alarms expected)")

plt.figure(figsize=(13, 7))
plt.subplot(211); plt.plot(t, noisy, lw=0.6, label="noisy")
plt.plot(t, soft, lw=1.5, label="soft denoised")
if clean is not None: plt.plot(t, clean, "k--", lw=1, label="clean")
plt.legend(); plt.title("Sensor signal")
plt.subplot(212); plt.plot(t, residual, lw=0.6, label="residual")
plt.axhline(3 * robust_std, color="r", ls="--"); plt.axhline(-3 * robust_std, color="r", ls="--")
plt.scatter(t[events], residual[events], color="r", zorder=3, label="anomaly")
plt.legend(); plt.title("Residual and 3-robust-sigma limits"); plt.tight_layout(); plt.show()

plt.figure(figsize=(12, 8))
for i, (n, c) in enumerate(zip(names, coeffs)):
    plt.subplot(len(coeffs), 1, i + 1); plt.plot(c, lw=0.7); plt.ylabel(n, rotation=0, labelpad=20)
plt.suptitle("Multi-level DWT coefficients (spikes show up in fine details)"); plt.tight_layout(); plt.show()


# ---------------- (f) 2-D DWT ----------------
def synthetic_image():
    img = np.zeros((256, 256), np.float64)
    img[:] = np.linspace(40, 200, 256)[None, :]
    cv2.rectangle(img, (40, 40), (120, 140), 230, -1)
    cv2.circle(img, (180, 170), 50, 20, -1)
    for x in range(140, 250, 12):
        cv2.line(img, (x, 20), (x, 100), 255, 2)
    return img.astype(np.uint8)


img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE) if os.path.exists(IMAGE_PATH) else None
if img is None:
    print("[info] using synthetic image for 2-D part")
    img = synthetic_image()
img = img[: img.shape[0] // 8 * 8, : img.shape[1] // 8 * 8]            # divisible by 2^3
f = img.astype(np.float64)

cA, (cH, cV, cD) = pywt.dwt2(f, "haar")
print(f"(f) dwt2 sub-band shape = {cA.shape} (half of {f.shape}); "
      f"perfect reconstruction: {np.allclose(pywt.idwt2((cA, (cH, cV, cD)), 'haar'), f)}")
print("    LL = approximation, LH = horizontal detail, HL = vertical detail, HH = diagonal detail")

noisy_img = f + rng.normal(0, 20, f.shape)
c2 = pywt.wavedec2(noisy_img, "db2", level=3)
sigma = np.median(np.abs(c2[-1][2])) / 0.6745                         # HH of finest level
T2 = sigma * np.sqrt(2 * np.log(noisy_img.size))
c2_th = [c2[0]] + [tuple(pywt.threshold(b, T2, "soft") for b in lvl) for lvl in c2[1:]]
den_img = np.clip(pywt.waverec2(c2_th, "db2")[: f.shape[0], : f.shape[1]], 0, 255)
print(f"    image denoising: PSNR noisy = {cv2.PSNR(f, np.clip(noisy_img, 0, 255)):.2f} dB -> "
      f"denoised = {cv2.PSNR(f, den_img):.2f} dB")

# ---------------- (g) compression ----------------
c3 = pywt.wavedec2(f, "db2", level=3)
arr, slices = pywt.coeffs_to_array(c3)
thr = np.quantile(np.abs(arr), 0.95)
arr_c = np.where(np.abs(arr) >= thr, arr, 0)
rec = pywt.waverec2(pywt.array_to_coeffs(arr_c, slices, output_format="wavedec2"), "db2")
rec = np.clip(rec[: f.shape[0], : f.shape[1]], 0, 255)
kept = np.count_nonzero(arr_c) / arr.size
print(f"(g) kept {kept * 100:.1f}% of coefficients -> PSNR = {cv2.PSNR(f, rec):.2f} dB "
      f"(energy is compacted into few large coefficients)")

plt.figure(figsize=(16, 7))
panels = [(f, "Original"), (cA, "LL (approx)"), (np.abs(cH), "LH (horizontal)"),
          (np.abs(cV), "HL (vertical)"), (np.abs(cD), "HH (diagonal)"),
          (np.clip(noisy_img, 0, 255), "Noisy"), (den_img, "Wavelet denoised"), (rec, f"5% coeffs")]
for i, (im, ti) in enumerate(panels):
    plt.subplot(2, 4, i + 1); plt.imshow(im, cmap="gray"); plt.title(ti); plt.axis("off")
plt.tight_layout(); plt.show()
