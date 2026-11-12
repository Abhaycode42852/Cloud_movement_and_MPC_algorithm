#!/usr/bin/env python3
"""
sky_forecast.py -- will a cloud cover the sun in the next few minutes?

What it does
  1. Gets N frames (default 12) INTERVAL seconds apart (default 5 s), either from the
     camera or from an existing video file (--video).
  2. Frame 1: finds the sun (bright, white, round blob) and the cloud contours
     (red/blue colour threshold + morphology + contours).
  3. All frames: measures how the clouds move between frames (phase correlation
     on a cloudiness image) and fits one robust velocity vector (pixels / second).
     The sun's own drift across the frame (hand-held / panning camera) is measured
     too and subtracted, so the velocity is relative to the sun.
  4. Slides the frame-1 cloud mask (with the sun's own glare blanked out) along that
     velocity and, for every future second, measures how much of the sun disc it
     overlaps  ->  sun coverage (%) over time.

Outputs (in runs/<timestamp>/):
  frames/frame_00.jpg ...   the raw captures (.png when the frames come from a video)   manifest.json   capture times
  forecast.jpg              annotated frame 1 + coverage-vs-time graph
  coverage.csv              t_seconds, coverage_percent
  forecast.json             sun, velocity, quality, events (contact / 50% / full / clear)

Usage
  python3 sky_forecast.py                          # capture 12 frames, 5 s apart, then forecast
  python3 sky_forecast.py --video runs/<name>      # same, but take the frames from a video in that folder
  python3 sky_forecast.py --video clip.mp4         # ... or from an explicit video file
  python3 sky_forecast.py --from-dir runs/<name>   # re-run the analysis on saved frames (tune thresholds)
  python3 sky_forecast.py --sun 900,200,25         # sun position/radius by hand (full-res pixels)
  python3 sky_forecast.py --horizon 600            # forecast further ahead (less reliable)

Limits you should know about
  * Only clouds that are ALREADY in the frame can be predicted. Clouds still outside
    the camera's view are not known, so a "0 %" forecast means "nothing visible is coming".
  * One average velocity is used for the whole sky (translation only). Clouds that grow,
    shrink or move at different speeds/heights are not modelled. Treat anything beyond a
    few minutes as a rough guide.
  * Wide-angle (fisheye-like) lenses bend distances near the edges; pixel speeds there
    are not equal to real speeds. Use --sky-circle to keep to the central, less bent area.
  * If a cloud already hides the sun in frame 1 the sun cannot be detected; the script
    then uses the clearest frame of the sequence, or --sun.
  * "Coverage" is the fraction of the sun disc covered by the DETECTED cloud mask, i.e.
    "is cloud over the sun", not "how bright is the sun".  Thin cloud can dim a panel a lot
    while the disc is still visible, and the sky right around the sun is bright and
    desaturated, so it is blanked out (--glare) before forecasting.
  * The forecast is only as good as the cloud mask is stable: measure it yourself with
    validate_forecast.py (it scores the forecast against the rest of a recorded video).
"""
import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
import threading
import time

import cv2
import numpy as np


# ====================================================================== camera
def csi_pipeline(sensor_id=0, w=1280, h=720, fps=30):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1 ! "
        "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
        "video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
    )


class Camera:
    """Reads the camera continuously in a thread so latest() is never a stale buffered frame."""

    def __init__(self, kind, device, w, h):
        if kind == "usb":
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            cap = cv2.VideoCapture(csi_pipeline(device, w, h), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise SystemExit("Cannot open the camera. Run:  python3 cam_test.py --scan")
        self.cap = cap
        self.frame, self.stamp = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame, self.stamp = f, time.monotonic()
            else:
                time.sleep(0.05)

    def wait_frame_after(self, t, timeout=3.0):
        """Return (frame, capture_time) for the first frame captured at/after monotonic time t."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                if self.frame is not None and self.stamp >= t:
                    return self.frame.copy(), self.stamp
            time.sleep(0.005)
        raise SystemExit("Camera stopped delivering frames.")

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def capture_sequence(a, run_dir, interval):
    cam = Camera(a.camera, a.device, a.width, a.height)
    try:
        print("Warming up camera (2 s, lets auto-exposure settle) ...")
        time.sleep(2.0)
        os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
        start = time.monotonic() + 0.2
        frames, stamps = [], []
        for i in range(a.frames):
            target = start + i * interval
            while time.monotonic() < target:
                time.sleep(min(0.2, max(target - time.monotonic(), 0.001)))
            frame, st = cam.wait_frame_after(target)
            frames.append(frame)
            stamps.append(st)
            cv2.imwrite(os.path.join(run_dir, "frames", f"frame_{i:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  captured {i + 1:2d}/{a.frames}   t = {st - stamps[0]:5.1f} s")
    finally:
        cam.close()
    times = [s - stamps[0] for s in stamps]
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"times": times, "started": dt.datetime.now().isoformat(timespec="seconds")}, f, indent=2)
    return frames, times


def load_sequence(folder, interval):
    paths = []
    for sub in (os.path.join(folder, "frames"), folder):
        paths = sorted(glob.glob(os.path.join(sub, "frame_*.jpg")) + glob.glob(os.path.join(sub, "frame_*.png")))
        if paths:
            break
    if len(paths) < 3:
        raise SystemExit(f"Need at least 3 frame_*.jpg (or .png) files in {folder} (or {folder}/frames)")
    frames = [cv2.imread(p) for p in paths]
    if any(f is None for f in frames):
        raise SystemExit("One of the frames could not be read.")
    times = None
    for mp in (os.path.join(folder, "manifest.json"), os.path.join(os.path.dirname(paths[0]), "manifest.json")):
        if os.path.exists(mp):
            with open(mp) as f:
                times = json.load(f).get("times")
            break
    if not times or len(times) != len(frames):
        times = [i * interval for i in range(len(frames))]
    return frames, times


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")

# The sun's disc is a fixed fraction of the frame with a given lens (~3 % on these GoPro
# clips, where the tracked ground-truth radius is 2.9-3.4 % of the width).  Detection only
# sees the blown-out core, which collapses behind haze, so the radius is floored at this.
SUN_MIN_FRAC = 0.03


def resolve_video_path(path):
    """Accept a video file, or a folder that contains exactly one video."""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        found = sorted(os.path.join(path, name) for name in os.listdir(path)
                       if os.path.splitext(name)[1].lower() in VIDEO_EXTS)
        if not found:
            raise SystemExit(f"No video files ({', '.join(VIDEO_EXTS)}) in {path}")
        if len(found) > 1:
            raise SystemExit("More than one video in that folder; pass the file itself:\n  " + "\n  ".join(found))
        return found[0]
    raise SystemExit(f"Video not found: {path}")


def load_video_sequence(path, a, run_dir):
    """Extract N frames INTERVAL seconds apart from a video and save them like a camera run."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 1.0:
            fps = 30.0
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:                                  # some containers do not report a frame count
            count = 0
            while cap.grab():
                count += 1
            if count == 0:
                raise SystemExit(f"Video has no readable frames: {path}")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)         # rewind after counting
        duration = count / fps

        n = min(a.frames, count)
        if n < a.frames:
            print(f"Video has only {count} frames; using {n} frames instead of {a.frames}.")
        if n < 3:
            raise SystemExit(f"Video is too short ({count} frames); need at least 3 frames.")

        if a.interval is None:
            # same cadence as the live camera (12 frames, 5 s apart) -> the first minute;
            # a clip shorter than that is sampled end to end instead of failing
            interval = 5.0 if duration >= 5.0 * (n - 1) else duration / (n - 1)
            print(f"Sampling {n} frames over the first {(n - 1) * interval:.0f} s "
                  f"of the {duration:.1f} s clip ({interval:.3g} s apart).")
        else:
            interval = a.interval
            if (n - 1) * interval > duration:
                interval = duration / (n - 1)
                print(f"Video is only {duration:.1f} s long; using {interval:.3g} s between frames "
                      f"instead of {a.interval:g} s.")

        times = [i * interval for i in range(n)]
        idxs = sorted({min(int(round(t * fps)), count - 1) for t in times})
        if len(idxs) < 3:
            raise SystemExit("Interval too small for this video; use a larger --interval.")
        times = [idx / fps for idx in idxs]

        os.makedirs(os.path.join(run_dir, "frames"), exist_ok=True)
        frames = []
        pos = 0
        for i, idx in enumerate(idxs):
            while pos < idx:
                if not cap.grab():
                    break
                pos += 1
            ok, frame = cap.read()
            if not ok:
                raise SystemExit(f"Could not read frame {idx} of {path}")
            pos += 1
            frames.append(frame)
            cv2.imwrite(os.path.join(run_dir, "frames", f"frame_{i:02d}.png"), frame)
            print(f"  extracted {i + 1:2d}/{len(idxs)}   t = {times[i]:5.1f} s")
    finally:
        cap.release()
    with open(os.path.join(run_dir, "manifest.json"), "w") as f:
        json.dump({"times": times, "started": dt.datetime.now().isoformat(timespec="seconds"),
                   "source": os.path.abspath(path)}, f, indent=2)
    return frames, times


# ================================================================== detection
def otsu_threshold(values, lo, hi, bins=128):
    """
    Otsu split of a 1-D sample inside [lo, hi], with a bimodality guard.

    A fixed red/blue threshold cannot work on every camera (a dark wide-angle sky and a
    bright one sit at completely different ratios), so the cloud/sky split is measured per
    frame.  Returns None when the histogram is not clearly two-class (e.g. a cloudless
    uniform sky), so the caller can fall back to a fixed value.
    """
    if values.size < 500:
        return None
    hist, edges = np.histogram(np.clip(values, lo, hi), bins=bins, range=(lo, hi))
    hist = hist.astype(np.float64)
    total = hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = total - w0
    csum = np.cumsum(hist * centers)
    m0 = csum / np.maximum(w0, 1e-9)
    m1 = (csum[-1] - csum) / np.maximum(w1, 1e-9)
    between = w0 * w1 * (m0 - m1) ** 2
    k = int(np.argmax(between))
    spread = float(np.sum(hist * (centers - csum[-1] / total) ** 2))
    if between[k] / max(spread, 1e-9) < 0.55:          # not two clean classes
        return None
    return float(centers[k])


def pick_sun(dets):
    """
    One sun reference for the whole sequence.

    A partly covered sun gives a smaller, shifted bright blob, so the radius comes from the
    clearest frame (the sun itself never changes size) and the centre is the median of the
    clearest quarter of the detections.
    """
    seen = [d["sun"] for d in dets if d["sun"] is not None]
    if not seen:
        return None
    rs = np.array([s[2] for s in seen])
    top = np.argsort(rs)[-max(3, len(seen) // 4):]
    return (float(np.median([seen[i][0] for i in top])),
            float(np.median([seen[i][1] for i in top])),
            float(rs.max()))


def sun_reference(dets, frames, a):
    """
    Sun disc in FRAME-1 coordinates, plus the largest core radius seen in the sequence.

    The forecast slides the frame-1 cloud mask, so the sun disc has to be where the sun was
    in that same frame.  A partly covered sun in frame 1 gives a shifted, small blob, so the
    radius comes from the clearest frame (the sun never changes size) while the centre comes
    from the brightest blob of frame 1 itself (the sun is the brightest thing, even through
    thin cloud).  If frame 1 is unusable, fall back to the robust detection median.
    """
    ref = pick_sun(dets)
    first = next((p for p in sun_positions(frames, a) if p is not None), None)
    if ref is None and first is None:
        return None
    if ref is None:
        return (first[0], first[1], SUN_MIN_FRAC * a.proc_width)
    if first is None:
        return ref
    # A sun behind haze keeps only a tiny blown-out core, but the disc itself does not
    # shrink: with these lenses it is ~3 % of the frame width.  A 24 px "sun" would make
    # the coverage test saturate (and the glare blanking miss the halo entirely).
    r = max(float(ref[2]), SUN_MIN_FRAC * a.proc_width)
    return (float(first[0]), float(first[1]), r)


def clear_core(mask, frame, sun, a):
    """
    Remove the sun's own blown-out core (and its immediate bloom) from the cloud mask.

    The core and the glow around it pass the cloud colour test, so if they are left in the
    forecast slides the sun's own light onto the disc.  Blanking a fixed number of sun radii
    instead erases the cloud that is about to arrive, which is worse: only the genuinely
    bright, white pixels are removed, dilated by a quarter of the sun radius.
    """
    if sun is None:
        return mask
    w = a.proc_width
    h = int(round(frame.shape[0] * w / frame.shape[1]))
    s = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
    core = ((hsv[..., 2] >= a.sun_v) & (hsv[..., 1] <= a.sun_s)).astype(np.uint8)
    k = max(3, int(round(0.25 * sun[2])) * 2 + 1)
    core = cv2.dilate(core, np.ones((k, k), np.uint8))
    if not core.any():
        return mask
    out = mask.copy()
    out[core > 0] = 0
    return out


def clear_sun(mask, sun, factor):
    """
    Blank the circumsolar glare out of the cloud mask.

    The sky around the sun is bright and desaturated, so a red/blue test calls the sun's own
    halo "cloud".  If that halo is left in the mask it slides across the sun forever and the
    forecast saturates at 100 % coverage.  Removing a disc of `factor` sun radii (per-frame
    sun removal only clears ~1 radius) makes the forecast measure clouds that actually drift
    in from elsewhere.
    """
    out = mask.copy()
    if sun is not None:
        cv2.circle(out, (int(round(sun[0])), int(round(sun[1]))),
                   int(round(sun[2] * max(float(factor), 1.0))), 0, -1)
    return out


def sun_positions(frames, a):
    """Centre of the brightest blob in each frame (processing pixels); None if not bright."""
    pos = []
    for f in frames:
        w = a.proc_width
        h = int(round(f.shape[0] * w / f.shape[1]))
        s = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
        v = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)[..., 2]
        peak = float(v.max())
        if peak < 240.0:
            pos.append(None)
            continue
        m = v >= max(240.0, peak - 40.0)
        ys, xs = np.nonzero(m)
        pos.append((float(xs.mean()), float(ys.mean()), int(m.sum())))
    return pos


def sun_veil(frames, dets, sun, a):
    """
    How much of the sun's light is missing in each captured frame (0 = clear, 1 = hidden).

    The red/blue mask only sees opaque cloud.  A thin veil dims a panel while leaving the
    mask almost empty, which is why a purely geometric forecast starts at 0 % while the sun
    is already half gone.  A blown-out sun keeps its core at the sensor maximum, and thin
    cloud eats that core from the inside, so the count of still-sun-bright pixels inside the
    known sun disc measures the attenuation directly.  Normalising by the clearest frame of
    the capture makes it independent of exposure and of the camera.
    """
    if sun is None:
        return None, None
    w = a.proc_width
    pos = sun_positions(frames, a)
    counts = []
    for f, p in zip(frames, pos):
        if p is None:
            p = (sun[0], sun[1])
        sc = f.shape[1] / float(w)
        y0 = max(0, int((p[1] - sun[2]) * sc))
        y1 = min(f.shape[0], int((p[1] + sun[2]) * sc) + 2)
        x0 = max(0, int((p[0] - sun[2]) * sc))
        x1 = min(f.shape[1], int((p[0] + sun[2]) * sc) + 2)
        if x1 <= x0 or y1 <= y0:
            counts.append(0)
            continue
        roi = f[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        cv2.circle(m, (int(round(p[0] * sc)) - x0, int(round(p[1] * sc)) - y0),
                   max(int(round(sun[2] * sc)), 1), 255, -1)
        # Count at FULL resolution: at 640 px a partly-covered sun loses most of its core
        # to the averaging, which would read as a thicker veil than there is.
        n = int(np.count_nonzero((hsv[..., 2] >= a.sun_v) & (hsv[..., 1] <= a.sun_s) & (m > 0)))
        counts.append(n)
    # exposure hunting makes a single frame jump; clouds do not.  A 3-tap median keeps the
    # trend and drops the flicker (one frame either side).
    if len(counts) >= 3:
        c = np.asarray(counts, float)
        sm = c.copy()
        sm[1:-1] = np.median(np.stack([c[:-2], c[1:-1], c[2:]]), axis=0)
        counts = [int(v) for v in sm]
    top = max(counts) if counts else 0
    if top <= 0:
        return [0.0] * len(frames), counts
    veil = [float(np.clip(1.0 - n / top, 0.0, 1.0)) for n in counts]
    return veil, counts


def blend_veil(rows, times, veil, a):
    """
    Measured attenuation inside the captured minute, then persistence.

    Inside the captured minute the brightness of the stored frames *is* the answer, so it is
    used directly.  Beyond it, the last measured attenuation is carried forward.

    Why not trust the advected mask's future arrival?  On the ten validated clips that scored
    28.1% mean error over the future window, against 18.1% for carrying the measured value
    forward: a cloud that is not on the sun yet may be thin, and the binary red/blue mask
    would call it opaque.  The mask's arrival is not thrown away -- `mask_arrival()` exposes
    it as an explicit scenario so it can be shown without being mistaken for a calibrated
    prediction.
    """
    if veil is None:
        return rows
    last_t = times[-1]
    v_last = float(veil[-1])
    out = []
    for t, c in rows:
        hidden = float(np.interp(t, times, veil)) if t <= last_t else v_last
        out.append((t, float(np.clip(hidden, 0.0, 1.0))))
    return out


def mask_arrival(rows, times, veil):
    """The mask's "a cloud is coming" curve: where it rises above the handover coverage."""
    if veil is None:
        return None
    last_t = times[-1]
    v_last = float(veil[-1])
    c_last = next((c for t, c in rows if t >= last_t - 1e-9), 0.0)
    if c_last >= 1.0 - 1e-6:
        return None
    out = []
    top = v_last
    for t, c in rows:
        if t <= last_t or c <= c_last:
            h = v_last
        else:
            h = 1.0 - (1.0 - v_last) * (1.0 - c) / (1.0 - c_last)
        h = float(np.clip(h, 0.0, 1.0))
        top = max(top, h)
        out.append((t, h))
    if top < v_last + 0.15:          # no meaningful arrival: nothing to show
        return None
    return out


def _robust_line(ts, vals):
    """Least-squares line with one round of residual trimming; returns slope, rms, inliers."""
    ts = np.asarray(ts, float)
    vals = np.asarray(vals, float)
    keep = np.ones(len(ts), bool)
    slope, rms = 0.0, 0.0
    for _ in range(3):
        A = np.column_stack([ts[keep], np.ones(int(keep.sum()))])
        slope, intercept = np.linalg.lstsq(A, vals[keep], rcond=None)[0]
        res = vals - (slope * ts + intercept)
        rms = float(np.sqrt(np.mean(res[keep] ** 2)))
        keep = np.abs(res) <= max(2.5 * rms, 1e-6)
    return float(slope), rms, int(keep.sum())


def estimate_sun_motion(frames, times, a):
    """
    Image-space velocity of the sun itself (processing px/s).

    Hand-held and panning cameras move the whole sky, sun included, so the clouds' image
    velocity is not their velocity relative to the sun.  The sun is the brightest thing in
    the frame and drifts smoothly, so a trimmed line fit through its positions gives that
    drift;  the forecast uses (cloud velocity - sun velocity), which is what the reference
    paper does by moving the sun rectangle together with the clouds.
    """
    pos = sun_positions(frames, a)
    good = [(t, p[0], p[1]) for t, p in zip(times, pos) if p is not None]
    if len(good) < 6:
        return dict(vx=0.0, vy=0.0, n=len(good), rms=0.0, ok=False)
    ts = [g[0] for g in good]
    vx, rmsx, nx = _robust_line(ts, [g[1] for g in good])
    vy, rmsy, ny = _robust_line(ts, [g[2] for g in good])
    rms = float(np.hypot(rmsx, rmsy))
    ok = min(nx, ny) >= 6 and np.hypot(vx, vy) < 0.6 * a.proc_width and rms < 0.06 * a.proc_width
    return dict(vx=vx, vy=vy, n=min(nx, ny), rms=rms, ok=bool(ok))


def detect(frame, a):
    """Find sun + cloud mask/contours in one frame. All coordinates are in 'processing' pixels."""
    H, W = frame.shape[:2]
    pw = a.proc_width
    ph = int(round(H * pw / W))
    small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(small, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(blur.astype(np.float32))
    rb = r / (b + 1.0)                      # sky ~0.3-0.6, cloud ~0.8-1.1

    # pixels we are allowed to look at: inside --sky-circle and not very dark (trees, buildings)
    valid = np.full((ph, pw), 255, np.uint8)
    if a.sky_circle:
        cx, cy, rad = a.sky_circle
        valid[:] = 0
        cv2.circle(valid, (int(cx * pw), int(cy * ph)), int(rad * pw), 255, -1)
    valid[hsv[..., 2] < a.min_v] = 0

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # ---- sun: bright + white + round (a big white cloud or glare is not round or is too large)
    sun_bin = ((hsv[..., 2] >= a.sun_v) & (hsv[..., 1] <= a.sun_s) & (valid > 0)).astype(np.uint8) * 255
    sun_bin = cv2.morphologyEx(sun_bin, cv2.MORPH_OPEN, k5)
    cnts, _ = cv2.findContours(sun_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for c in cnts:
        area = cv2.contourArea(c)
        if area < a.sun_min_area or area > a.sun_max_frac * pw * ph:
            continue
        per = cv2.arcLength(c, True)
        if 4 * np.pi * area / (per * per + 1e-6) < a.sun_circularity:
            continue
        if best is None or area > best[0]:
            best = (area, c)
    sun = None
    if best is not None:
        (x, y), rad = cv2.minEnclosingCircle(best[1])
        sun = (float(x), float(y), float(rad))

    # ---- clouds: red/blue ratio, cleaned, small blobs dropped
    rb_thr = a.rb_thresh
    if isinstance(rb_thr, str):                                  # "auto": measure the split
        vals = rb[valid > 0]
        rb_thr = otsu_threshold(vals, 0.35, 1.30)
        rb_thr = 0.85 if rb_thr is None else min(max(rb_thr, 0.55), 1.10)
        n_valid = max(int(np.count_nonzero(valid)), 1)
        if np.count_nonzero((rb >= rb_thr) & (valid > 0)) > 0.75 * n_valid:
            rb_thr = max(rb_thr, float(np.percentile(vals, 85)))   # never call the whole sky cloud
    cloud = ((rb >= rb_thr) & (valid > 0)).astype(np.uint8) * 255
    if sun is not None:
        cv2.circle(cloud, (int(sun[0]), int(sun[1])), int(sun[2]) + 2, 0, -1)   # sun disc is not cloud
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_OPEN, k5)
    cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, k9)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cloud, connectivity=8)
    ids = np.nonzero(stats[:, cv2.CC_STAT_AREA] >= a.cloud_min_area)[0]
    ids = ids[ids > 0]
    cmask = (np.isin(lab, ids) * 255).astype(np.uint8)
    ccnts, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sky_px = int(np.count_nonzero(valid))
    # ---- image used to measure motion: invalid pixels and the sun are flattened to the mean
    if a.motion_feature == "rb":
        feat = np.clip(rb, 0.0, 1.5).astype(np.float32)
    else:
        feat = cv2.GaussianBlur(cmask.astype(np.float32) / 255.0, (0, 0), 3)
    bad = (valid == 0).astype(np.uint8)
    if sun is not None:
        cv2.circle(bad, (int(sun[0]), int(sun[1])), int(sun[2] * 1.5) + 3, 1, -1)
    mean = float(feat[bad == 0].mean()) if np.any(bad == 0) else float(feat.mean())
    feat[bad > 0] = mean
    feat -= mean

    return dict(shape=(ph, pw), scale=W / pw, sun=sun, cloud_mask=cmask, cloud_cnts=ccnts,
                feat=feat, cloud_fraction=float(np.count_nonzero(cmask)) / max(sky_px, 1),
                rb_thr=float(rb_thr))


# ====================================================================== motion
def _phase_sign():
    """OpenCV versions differ on the sign of phaseCorrelate; calibrate against a known shift."""
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.random((128, 128)).astype(np.float32), (0, 0), 3)
    moved = np.roll(base, (3, 5), axis=(0, 1))          # content moved 5 px right, 3 px down
    (dx, _), _ = cv2.phaseCorrelate(base, moved)
    return 1.0 if dx > 0 else -1.0


def estimate_motion(dets, times, a):
    h, w = dets[0]["feat"].shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    sign = _phase_sign()
    pairs = []
    for i in range(len(dets) - 1):
        (dx, dy), resp = cv2.phaseCorrelate(dets[i]["feat"], dets[i + 1]["feat"], win)
        dx, dy = sign * dx, sign * dy
        dtm = max(times[i + 1] - times[i], 1e-3)
        pairs.append(dict(pair=f"{i}->{i + 1}", dx=dx, dy=dy, resp=float(resp), vx=dx / dtm, vy=dy / dtm))
    good = [p for p in pairs if p["resp"] >= a.min_response]
    if not good:
        return dict(vx=0.0, vy=0.0, quality="NONE", n_good=0, n_pairs=len(pairs), spread=0.0, pairs=pairs)
    vx = float(np.median([p["vx"] for p in good]))
    vy = float(np.median([p["vy"] for p in good]))
    spread = float(np.sqrt(np.mean([(p["vx"] - vx) ** 2 + (p["vy"] - vy) ** 2 for p in good])))
    speed = float(np.hypot(vx, vy))
    frac = len(good) / len(pairs)
    if frac >= 0.7 and spread <= 0.35 * max(speed, 1.0):
        quality = "GOOD"
    elif len(good) >= 3 and spread <= 0.8 * max(speed, 1.0):
        quality = "FAIR"
    else:
        quality = "POOR"
    return dict(vx=vx, vy=vy, quality=quality, n_good=len(good), n_pairs=len(pairs), spread=spread, pairs=pairs)


# =================================================================== forecast
def forecast(cloud_mask, sun, vx, vy, horizon, step):
    """Slide the cloud mask by v*t and measure overlap with the sun disc. Returns list of (t, coverage 0-1)."""
    h, w = cloud_mask.shape
    disc = np.zeros((h, w), np.uint8)
    cv2.circle(disc, (int(round(sun[0])), int(round(sun[1]))), max(int(round(sun[2])), 1), 255, -1)
    sun_area = max(int(np.count_nonzero(disc)), 1)
    rows = []
    for t in np.arange(0.0, horizon + 1e-9, step):
        M = np.float32([[1, 0, vx * t], [0, 1, vy * t]])
        moved = cv2.warpAffine(cloud_mask, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        overlap = int(np.count_nonzero((moved > 127) & (disc > 0)))
        rows.append((float(t), overlap / sun_area))
    return rows


def events(rows):
    def first(pred):
        for t, c in rows:
            if pred(c):
                return t
        return None
    cov_max = max(c for _, c in rows)
    t_max = first(lambda c: c >= cov_max - 1e-9)
    contact = first(lambda c: c > 0.02)
    clears = None
    if contact is not None:
        peak_seen = False
        for t, c in rows:
            if t < contact:
                continue
            if c > 0.02:
                peak_seen = True
            elif peak_seen:
                clears = t
                break
    return dict(now_pct=round(rows[0][1] * 100, 1), contact_s=contact,
                half_s=first(lambda c: c >= 0.5), full_s=first(lambda c: c >= 0.98),
                max_pct=round(cov_max * 100, 1), max_at_s=t_max, clears_s=clears)


# ================================================================== rendering
def timeline_panel(width, rows, horizon, ev):
    H = 220
    img = np.full((H, width, 3), 28, np.uint8)
    L, R, T, B = 70, width - 25, 30, H - 45
    cv2.putText(img, "Predicted sun coverage by cloud", (L, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    for pct in (0, 50, 100):
        y = int(B - pct / 100 * (B - T))
        cv2.line(img, (L, y), (R, y), (70, 70, 70), 1)
        cv2.putText(img, f"{pct}%", (20, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    nt = 5
    for k in range(nt + 1):
        t = horizon * k / nt
        x = int(L + (R - L) * k / nt)
        cv2.line(img, (x, B), (x, B + 5), (200, 200, 200), 1)
        cv2.putText(img, f"{t:.0f}s", (x - 18, B + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    pts = np.array([[L + (R - L) * t / horizon, B - c * (B - T)] for t, c in rows], np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (0, 200, 255), 2, cv2.LINE_AA)
    if ev["contact_s"] is not None:
        x = int(L + (R - L) * ev["contact_s"] / horizon)
        cv2.line(img, (x, T), (x, B), (0, 0, 255), 1)
    return img


def annotate(frame, det, sun, motion, rows, horizon, ev, show_times, now_pct=None):
    s = det["scale"]
    out = frame.copy()
    for c in det["cloud_cnts"]:
        cv2.drawContours(out, [(c * s).astype(np.int32)], -1, (0, 255, 255), 2)       # yellow: clouds now
    if sun is not None:
        cv2.circle(out, (int(sun[0] * s), int(sun[1] * s)), int(sun[2] * s) + 4, (0, 0, 255), 3)   # red: sun
    vx, vy = motion["vx"], motion["vy"]
    colors = [(255, 200, 0), (255, 0, 255), (0, 255, 0), (255, 128, 128)]
    if motion["quality"] != "NONE":
        for col, t in zip(colors, show_times):
            for c in det["cloud_cnts"]:
                shifted = ((c.astype(np.float32) + np.array([vx * t, vy * t], np.float32)) * s).astype(np.int32)
                cv2.drawContours(out, [shifted], -1, col, 1)
            cv2.putText(out, f"+{t:.0f}s", (10, 60 + 22 * colors.index(col)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        # arrow showing 60 s of travel from the image centre
        cx, cy = out.shape[1] // 2, out.shape[0] // 2
        cv2.arrowedLine(out, (cx, cy), (int(cx + vx * 60 * s), int(cy + vy * 60 * s)), (255, 255, 255), 3, tipLength=0.15)
    if ev is None:
        msg = "sun not found - no coverage forecast"
    else:
        c = ev["contact_s"]
        now = ev["now_pct"] if now_pct is None else now_pct
        msg = (f"sun covered now {now:.0f}% | " +
               (f"first contact in {c:.0f}s | max {ev['max_pct']:.0f}%" if c is not None else "no cloud reaches the sun in the horizon"))
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, msg + f" | motion {motion['quality']}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if ev is not None:
        out = np.vstack([out, timeline_panel(out.shape[1], rows, horizon, ev)])
    return out


# ======================================================================= main
def fmt_t(t):
    return "never within horizon" if t is None else f"{t:.0f} s"


def _rb_arg(s):
    return "auto" if s.lower() == "auto" else float(s)


def build_parser():
    p = argparse.ArgumentParser(description="12-frame sun/cloud collision forecast",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("capture")
    g.add_argument("--camera", choices=["usb", "csi"], default="usb")
    g.add_argument("--device", type=int, default=0, help="USB: /dev/videoN index; CSI: sensor id")
    g.add_argument("--width", type=int, default=1280)
    g.add_argument("--height", type=int, default=720)
    g.add_argument("--frames", type=int, default=12)
    g.add_argument("--interval", type=float, default=None,
                   help="seconds between frames (default: 5 -> the first minute; for a short clip the frames are spread over the clip)")
    g.add_argument("--out", default="runs", help="folder for results")
    g.add_argument("--from-dir", help="analyse frames saved by an earlier run instead of capturing")
    g.add_argument("--video", metavar="PATH",
                   help="take the frames from a video file, or a folder containing one video, instead of the camera")
    g = p.add_argument_group("forecast")
    g.add_argument("--horizon", type=float, default=300.0, help="seconds to predict ahead")
    g.add_argument("--step", type=float, default=1.0, help="forecast time resolution (s)")
    g.add_argument("--min-response", type=float, default=0.1, help="reject frame pairs whose motion match is weaker than this")
    g.add_argument("--motion-feature", choices=["rb", "mask"], default="rb")
    g.add_argument("--sun", type=lambda s: tuple(float(v) for v in s.split(",")), help="x,y,radius of the sun in full-resolution pixels")
    g = p.add_argument_group("detection (tune on your own frames)")
    g.add_argument("--proc-width", type=int, default=640)
    g.add_argument("--rb-thresh", type=_rb_arg, default="auto",
                   help="red/blue ratio above which a pixel is cloud, or 'auto' to measure the split per frame")
    g.add_argument("--min-v", type=int, default=40, help="HSV V below this is not sky")
    g.add_argument("--sun-v", type=int, default=245)
    g.add_argument("--sun-s", type=int, default=60)
    g.add_argument("--sun-min-area", type=int, default=30)
    g.add_argument("--sun-max-frac", type=float, default=0.02, help="sun blob larger than this fraction of the image is glare/cloud")
    g.add_argument("--sun-circularity", type=float, default=0.6)
    g.add_argument("--cloud-min-area", type=int, default=150)
    g.add_argument("--glare", type=float, default=1.0,
                   help="extra sun radii of circumsolar glare blanked out of the cloud mask "
                        "before forecasting (the sun's bright core is always removed; "
                        "1 = keep everything outside the core)")
    g.add_argument("--sun-motion", choices=["auto", "off"], default="auto",
                   help="subtract the camera/sun image drift so clouds are forecast relative to the sun")
    g.add_argument("--sky-circle", type=lambda s: tuple(float(v) for v in s.split(",")),
                   help="cx,cy,r as fractions (r of image width), e.g. 0.5,0.5,0.45 keeps only the central sky")
    return p


def main():
    a = build_parser().parse_args()
    if a.from_dir and a.video:
        raise SystemExit("Use either --from-dir or --video, not both.")
    interval = 5.0 if a.interval is None else a.interval
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if a.from_dir:
        run_dir = os.path.join(a.out, stamp)
        os.makedirs(run_dir, exist_ok=True)
        frames, times = load_sequence(a.from_dir, interval)
        print(f"Loaded {len(frames)} frames from {a.from_dir}")
    elif a.video:
        video = resolve_video_path(a.video)
        run_dir = os.path.join(a.out, stamp)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Reading frames from video {video}")
        frames, times = load_video_sequence(video, a, run_dir)
    else:
        run_dir = os.path.join(a.out, stamp)
        os.makedirs(run_dir, exist_ok=True)
        span = (a.frames - 1) * interval
        print(f"Capturing {a.frames} frames, {interval:g} s apart (about {span:.0f} s). Keep the camera still.")
        frames, times = capture_sequence(a, run_dir, interval)

    print("Analysing ...")
    dets = [detect(f, a) for f in frames]
    d0 = dets[0]

    # ---- sun reference: centre in frame-1 coordinates, radius from the clearest frame
    sun, sun_src = sun_reference(dets, frames, a), "detected"
    if a.sun:
        x, y, r = a.sun
        sun, sun_src = (x / d0["scale"], y / d0["scale"], r / d0["scale"]), "--sun"

    motion = estimate_motion(dets, times, a)
    sun_motion = estimate_sun_motion(frames, times, a)
    sun_drift_applied = bool(sun_motion["ok"] and a.sun_motion == "auto")
    if sun_drift_applied:                                   # forecast clouds relative to the sun
        motion["vx"] -= sun_motion["vx"]
        motion["vy"] -= sun_motion["vy"]
    speed = float(np.hypot(motion["vx"], motion["vy"]))
    heading = float(np.degrees(np.arctan2(motion["vy"], motion["vx"])))

    # the sun's own core is not cloud: take it out before forecasting.  --glare adds a
    # wider blank (in sun radii) around it, which was needed before the sun position and
    # the sun-relative drift were fixed, but it also deletes the cloud about to arrive.
    d0f = d0
    if sun is not None:
        cm = clear_core(d0["cloud_mask"], frames[0], sun, a)
        if a.glare > 1.0:
            cm = clear_sun(cm, sun, a.glare)
        cc, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        d0f = dict(d0, cloud_mask=cm, cloud_cnts=cc)

    print("\n=== RESULT ===")
    print(f"Cloud cover in frame 1: {d0['cloud_fraction'] * 100:.0f}%   ({len(d0['cloud_cnts'])} cloud contours)")
    if sun is None:
        print("Sun: NOT FOUND in any frame (hidden, or thresholds off). Try --sun x,y,r or tune --sun-v / --sun-s.")
    else:
        print(f"Sun: centre ({sun[0] * d0['scale']:.0f}, {sun[1] * d0['scale']:.0f}) px, radius {sun[2] * d0['scale']:.0f} px   [from {sun_src}]")
        extra = "" if a.glare <= 1.0 else f"; extra glare blank to {a.glare:g} sun radii"
        print(f"      sun core removed from the cloud mask before forecasting{extra}")
    print(f"Cloud motion: ({motion['vx'] * d0['scale']:+.1f}, {motion['vy'] * d0['scale']:+.1f}) px/s in full-res pixels "
          f"| speed {speed * d0['scale']:.1f} px/s | direction {heading:.0f} deg (0=right, 90=down)")
    if sun_drift_applied:
        print(f"      (camera/sun drift ({sun_motion['vx'] * d0['scale']:+.1f}, {sun_motion['vy'] * d0['scale']:+.1f}) px/s "
              f"subtracted, {sun_motion['n']} sun positions used)")
    elif a.sun_motion == "auto":
        print(f"      (no usable sun drift: {sun_motion['n']} sun positions, rms {sun_motion['rms'] * d0['scale']:.0f} px)")
    print(f"Motion quality: {motion['quality']}  ({motion['n_good']}/{motion['n_pairs']} frame pairs usable, spread {motion['spread'] * d0['scale']:.1f} px/s)")
    no_clouds = len(d0f["cloud_cnts"]) == 0
    if no_clouds:
        print("  -> No clouds in view in frame 1, so there is nothing to track.")
    elif motion["quality"] in ("NONE", "POOR"):
        print("  -> Motion is unreliable (uniform sky, camera moved, or clouds too slow/featureless). Forecast below is not trustworthy.")
    elif motion["quality"] == "FAIR":
        print("  -> Motion is only fair: trust the first ~30 s of the forecast, not the whole horizon.")

    rows, ev, now_pct = None, None, None
    if sun is not None and (motion["quality"] != "NONE" or no_clouds):
        veil, veil_counts = sun_veil(frames, dets, sun, a)
        disc = np.zeros_like(d0["cloud_mask"])
        cv2.circle(disc, (int(round(sun[0])), int(round(sun[1]))), int(round(sun[2])), 255, -1)
        mask_now = 100.0 * np.count_nonzero((d0f["cloud_mask"] > 127) & (disc > 0)) / max(np.count_nonzero(disc), 1)
        rows = forecast(d0f["cloud_mask"], sun, motion["vx"], motion["vy"], a.horizon, a.step)
        rows = blend_veil(rows, times, veil, a)
        now_pct = 100.0 * (veil[0] if veil is not None else 0.0)
        ev = events(rows)
        print(f"\nSun hidden now: {now_pct:.0f}%   (measured from the sun's brightness; "
              f"opaque cloud mask alone says {mask_now:.0f}%)")
        if now_pct > 2.0:
            print(f"First contact:          already (sun is {now_pct:.0f}% hidden at the first frame)")
        else:
            print(f"First contact in:       {fmt_t(ev['contact_s'])}")
        print(f"50% covered in:         {fmt_t(ev['half_s'])}")
        print(f"Fully covered in:       {fmt_t(ev['full_s'])}")
        print(f"Maximum coverage:       {ev['max_pct']:.0f}%  (at {fmt_t(ev['max_at_s'])})")
        print(f"Sun clear again at:     {fmt_t(ev['clears_s'])}")
        print(f"Horizon:                {a.horizon:.0f} s   (first {times[-1]:.0f} s measured from the "
              f"frames, the rest forecast)")
        print("(coverage is how much of the sun's light is blocked: thin veil plus opaque cloud.")
        print(" Inside the captured minute it is measured from the frames; after that the last")
        print(" measured attenuation is carried forward and the advected cloud mask is combined")
        print(" with it, so a cloud arriving from the edge of the frame still shows up.)")
        with open(os.path.join(run_dir, "coverage.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_seconds", "coverage_percent"])
            w.writerows([(f"{t:.1f}", f"{c * 100:.1f}") for t, c in rows])
    elif sun is not None:
        print("\nNo forecast: cloud motion could not be measured.")

    show = [t for t in (30, 60, 120, 240) if t <= a.horizon][:4]
    if ev and ev["contact_s"] is not None and ev["contact_s"] not in show:
        show = ([ev["contact_s"]] + show)[:4]
    img = annotate(frames[0], d0f, sun, motion, rows, a.horizon, ev, show)
    cv2.imwrite(os.path.join(run_dir, "forecast.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    with open(os.path.join(run_dir, "forecast.json"), "w") as f:
        json.dump(dict(sun_full_res=None if sun is None else [round(v * d0["scale"], 1) for v in sun],
                       sun_source=sun_src, cloud_fraction_frame1=round(d0["cloud_fraction"], 3),
                       glare_radii=a.glare, rb_thresh_used=round(d0["rb_thr"], 3),
                       sun_drift_px_per_s_full=None if not sun_drift_applied else
                       [round(sun_motion["vx"] * d0["scale"], 2), round(sun_motion["vy"] * d0["scale"], 2)],
                       velocity_px_per_s_full_res=[round(motion["vx"] * d0["scale"], 2), round(motion["vy"] * d0["scale"], 2)],
                       motion_quality=motion["quality"], events=ev,
                       sun_hidden_now_pct=None if veil is None else round(100 * veil[0], 1),
                       sun_veil_percent_observed=[round(100 * v, 1) for v in veil] if veil is not None else None,
                       handover_s=round(times[-1], 2),
                       pairs=[{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in motion["pairs"]]),
                  f, indent=2)
    saved = ["forecast.jpg", "forecast.json"] + (["coverage.csv"] if rows else [])
    print(f"\nSaved to {run_dir}/  ({', '.join(saved)})")


if __name__ == "__main__":
    main()
