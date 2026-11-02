"""Estimasi kecepatan kendaraan dari video CCTV — run lokal (CPU).

Pakai: ./.venv/bin/python local_speed.py [--video test_30s.mp4] [--seconds 30]
Butuh: yolov8n.pt sefolder dengan file ini (sudah ada).
Keluar: out_local.mp4 (anotasi) + speeds_local.csv
"""
import argparse
import sys

import cv2
import pandas as pd
from ultralytics import YOLO

CONFIG = {
    "model": "yolov8n.pt",
    "classes": [2, 3, 5, 7],  # car, motorcycle, bus, truck
    "conf": 0.35,
    "imgsz": 480,            # kecil = cepat (default 640); akurasi turun dikit
    "stride": 2,              # proses tiap frame ke-2 (~2x cepat); dt dikoreksi otomatis
    "line_a": 0.70,           # garis atas (fraksi tinggi frame)
    "line_b": 0.85,           # garis bawah
    "real_distance_m": 10.0,  # PLACEHOLDER — kalibrasi via marka jalan!
    "out_video": "out_local.mp4",
    "out_csv": "speeds_local.csv",
}


def process_video(video_path, max_seconds=None):
    model = YOLO(CONFIG["model"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"gagal buka video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, CONFIG["stride"])
    efps = fps / stride  # fps efektif untuk hitung waktu
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(max_seconds * fps) if max_seconds else None

    yA, yB = int(CONFIG["line_a"] * H), int(CONFIG["line_b"] * H)
    v_max, v_min = 250 / 3.6, 10 / 3.6
    f_min = max(2, int(CONFIG["real_distance_m"] / v_max * efps))
    f_max = int(CONFIG["real_distance_m"] / v_min * efps)
    X_TOL, MATCH_R = 80, 60

    listA, listB, prev = [], [], []  # crossing: (frame_efektif, x, w, h, tid)
    saved = {}  # frame_asli -> [(x1, y1, x2, y2, tid)]
    n, m = 0, 0
    # Pass 1: inference + catat crossing (tanpa tulis video)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if max_frames and n > max_frames:
            break
        if n % stride != 0:
            continue
        m += 1
        res = model.track(
            frame, persist=True, tracker="bytetrack.yaml", imgsz=CONFIG["imgsz"],
            classes=CONFIG["classes"], conf=CONFIG["conf"], verbose=False,
        )[0]
        curr, dets = [], []
        if res.boxes is not None and res.boxes.id is not None:
            for box, tid in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = box
                curr.append((float((x1 + x2) / 2), float(y2),
                             float(x2 - x1), float(y2 - y1), int(tid)))
                dets.append((int(x1), int(y1), int(x2), int(y2), int(tid)))
        saved[n] = dets
        used = set()
        for (x, yb, bw, bh, tid) in curr:
            bi = min(range(len(prev)),
                     key=lambda i: abs(prev[i][0] - x) + abs(prev[i][1] - yb),
                     default=None)
            if (bi is not None and bi not in used
                    and abs(prev[bi][0] - x) + abs(prev[bi][1] - yb) < MATCH_R * stride):
                used.add(bi)
                py = prev[bi][1]
                if (py - yA) * (yb - yA) < 0:
                    listA.append((m, x, bw, bh, tid))
                if (py - yB) * (yb - yB) < 0:
                    listB.append((m, x, bw, bh, tid))
        prev = curr
        if n % int(fps * 5) == 0:
            print(f"  {n / fps:.0f}s diproses...", flush=True)
    cap.release()

    # Matching antar-garis (lajur + ukuran bbox, tanpa tergantung ID)
    speeds, usedB, id_speed = [], set(), {}
    for fa, xa, wa, ha, ta in listA:
        cand = []
        for i, (fb, xb, wb, hb, tb) in enumerate(listB):
            if i in usedB or abs(xb - xa) >= X_TOL:
                continue
            dt = abs(fb - fa)
            if not (f_min <= dt <= f_max):
                continue
            size = abs(wb - wa) / max(wb, wa) + abs(hb - ha) / max(hb, ha)
            if size > 0.6:  # beda ukuran = beda kendaraan
                continue
            cand.append((size, i))
        if cand:
            cand.sort()
            usedB.add(cand[0][1])
            fb, _xb, _wb, _hb, tb = listB[cand[0][1]]
            kmh = round(CONFIG["real_distance_m"] / (abs(fb - fa) / efps) * 3.6, 1)
            speeds.append(kmh)
            id_speed[tb] = kmh
    print(f"crossA={len(listA)} crossB={len(listB)} matched={len(speeds)}")
    df = pd.DataFrame({"kmh": speeds})
    df.to_csv(CONFIG["out_csv"], index=False)

    # Pass 2: tulis video + label (cepat, tanpa inference)
    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        CONFIG["out_video"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
    )
    n, last = 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if max_frames and n > max_frames:
            break
        if saved.get(n):
            last = saved[n]  # bingkai terproses; selain itu pakai hasil terakhir
        for (x1, y1, x2, y2, tid) in last:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if tid in id_speed:
                cv2.putText(frame, f"{id_speed[tid]} km/h", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        for y, color in [(yA, (255, 0, 0)), (yB, (0, 0, 255))]:
            cv2.line(frame, (0, y), (W, y), color, 2)
        writer.write(frame)
    cap.release()
    writer.release()
    # Konversi ke H.264 agar bisa diputar di VS Code/QuickTime (mp4v mentah sering ditolak)
    import shutil
    import subprocess
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", CONFIG["out_video"],
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             CONFIG["out_video"].replace(".mp4", "_h264.mp4")],
            check=False,
        )
    return df, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="test_30s.mp4")
    ap.add_argument("--seconds", type=float, default=30,
                    help="batasi durasi proses (detik); 0 = full")
    args = ap.parse_args()
    print(f"video: {args.video} | model: {CONFIG['model']}")
    df, fps = process_video(args.video,
                            max_seconds=args.seconds or None)
    print(f"fps: {fps:.1f} | kendaraan terukur: {len(df)}")
    print(df.describe().to_string() if len(df) else "KOSONG")
    print(f"tersimpan: {CONFIG['out_video']}, {CONFIG['out_csv']}")


if __name__ == "__main__":
    main()
