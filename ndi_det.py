#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import collections
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import NDIlib as ndi  # pip install ndi-python

# ======== CONFIG ========
SOURCE_MATCH = "BIRDDOG"  # part of NDI source name
VIDEO_RES = (480, 288)    # W,H for processing/recording
RECORD_DURATION = 10      # seconds
PTZ_COOLDOWN = 1.0        # sec between moves
DEAD_ZONE = 40            # +/- px around center tolerated
THRESHOLD = 100           # extra before pan
PAN_SPEED = 0.3           # -1..+1
SHOW_WINDOW = True
# ========================

# --- Prepare folders ---
os.makedirs("videos/with_box", exist_ok=True)
os.makedirs("videos/without_box", exist_ok=True)

# --- YOLO + DeepSort ---
model = YOLO("yolov8s.pt")
if torch.cuda.is_available():
    model.to("cuda")
    print(" Using CUDA")
else:
    print(" Using CPU")
tracker = DeepSort(max_age=30)
fps_window = collections.deque(maxlen=30)

# ======== NDI helpers ========
def ndi_init_and_find_source(match):
    if not ndi.initialize():
        raise RuntimeError("NDI failed to initialize")

    finder = ndi.find_create_v2()
    if not finder:
        raise RuntimeError("NDI finder create failed")

    print(" Discovering NDI sources...")
    time.sleep(2)
    sources = ndi.find_get_current_sources(finder)
    if not sources:
        raise RuntimeError("No NDI sources found")

    chosen = None
    for s in sources:
        name = s.ndi_name  # already string
        if match.lower() in name.lower():
            chosen = s
            break

    if not chosen:
        print("Available sources:")
        for s in sources:
            print(" -", s.ndi_name)
        raise RuntimeError(f"Could not find source containing '{match}'")

    return finder, chosen

def ndi_create_receiver_and_connect(source):
    recv = ndi.recv_create_v3()
    if not recv:
        raise RuntimeError("NDI recv create failed")
    ndi.recv_connect(recv, source)
    return recv

def ndi_frame_to_bgr(v_frame):
    """Convert NDI frame (BGRA or UYVY) to BGR numpy array."""
    h, w = v_frame.yres, v_frame.xres
    frame_bytes = bytes(v_frame.data)
    size = len(frame_bytes)

    bgra_size = w * h * 4
    uyvy_size = w * h * 2

    if size == bgra_size:
        img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif size == uyvy_size:
        img = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 2))
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    else:
        raise ValueError(f"Unsupported frame size {size} for {w}x{h}")

def ndi_send_xml(recv, xml_str):
    """Send PTZ XML metadata (old ndi-python API: string directly)."""
    ndi.recv_send_metadata(recv, xml_str)

def ptz_pan_tilt_speed(recv, pan_speed, tilt_speed):
    xml = f'<ntk_ptz_pan_tilt_speed pan_speed="{pan_speed:.3f}" tilt_speed="{tilt_speed:.3f}"/>'
    ndi_send_xml(recv, xml)
# =============================

def main():
    finder, source = None, None
    recv = None
    video_writer_with_box = None
    video_writer_no_box = None
    record_start_time = None
    is_recording = False
    last_move_time = 0

    try:
        finder, source = ndi_init_and_find_source(SOURCE_MATCH)
        recv = ndi_create_receiver_and_connect(source)
        print(" Connected to:", source.ndi_name)

        if SHOW_WINDOW:
            cv2.namedWindow("NDI Tracking + PTZ", cv2.WINDOW_NORMAL)

        print(" Waiting for frames...")
        while True:
            t, v, a, m = ndi.recv_capture_v2(recv, 1000)  # 1000 ms timeout

            if t == ndi.FRAME_TYPE_VIDEO:
                bgr = ndi_frame_to_bgr(v)
                ndi.recv_free_video_v2(recv, v)  # match capture_v2()

                frame_no_box = cv2.resize(bgr, VIDEO_RES)
                frame = frame_no_box.copy()

                now = time.time()
                fps_window.append(now)
                if len(fps_window) >= 2:
                    real_fps = len(fps_window) / (fps_window[-1] - fps_window[0])
                    VIDEO_FPS_DYNAMIC = max(5, min(30, round(real_fps)))
                else:
                    VIDEO_FPS_DYNAMIC = 30

                results = model(frame, classes=[0], verbose=False)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

                tracks = tracker.update_tracks(detections, frame=frame)
                frame_cx = frame.shape[1] // 2
                detected = False

                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    detected = True

                    x1, y1, x2, y2 = map(int, track.to_ltrb())
                    cx = (x1 + x2) // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {track.track_id}", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

                    dx = cx - frame_cx
                    move = None
                    if dx > THRESHOLD + DEAD_ZONE:
                        move = +PAN_SPEED
                    elif dx < -THRESHOLD - DEAD_ZONE:
                        move = -PAN_SPEED

                    if move is not None and (time.time() - last_move_time > PTZ_COOLDOWN):
                        print(" Pan:", "right" if move > 0 else "left")
                        ptz_pan_tilt_speed(recv, move, 0.0)
                        time.sleep(0.3)
                        ptz_pan_tilt_speed(recv, 0.0, 0.0)
                        last_move_time = time.time()

                    break  # only first confirmed track

                if detected and not is_recording:
                    record_start_time = time.time()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn_with = f"videos/with_box/output_{ts}_with_box.mp4"
                    fn_no = f"videos/without_box/output_{ts}_no_box.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    print(f" Start recording @ ~{VIDEO_FPS_DYNAMIC} FPS")
                    video_writer_with_box = cv2.VideoWriter(fn_with, fourcc, VIDEO_FPS_DYNAMIC, VIDEO_RES)
                    video_writer_no_box = cv2.VideoWriter(fn_no, fourcc, VIDEO_FPS_DYNAMIC, VIDEO_RES)
                    is_recording = True

                if is_recording:
                    video_writer_with_box.write(frame)
                    video_writer_no_box.write(frame_no_box)
                    if time.time() - record_start_time >= RECORD_DURATION:
                        print(" Finished 10-second recording.")
                        video_writer_with_box.release()
                        video_writer_no_box.release()
                        is_recording = False

                if SHOW_WINDOW:
                    cv2.imshow("NDI Tracking + PTZ", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            elif t == ndi.FRAME_TYPE_METADATA and m is not None:
                ndi.recv_free_metadata(recv, m)  # no _v2 in your binding

            elif t == ndi.FRAME_TYPE_NONE:
                continue

    finally:
        try:
            if recv:
                ptz_pan_tilt_speed(recv, 0.0, 0.0)
        except Exception:
            pass
        if SHOW_WINDOW:
            cv2.destroyAllWindows()
        if finder:
            ndi.find_destroy(finder)
        if recv:
            ndi.recv_destroy(recv)
        ndi.destroy()

if __name__ == "__main__":
    main()


# In[ ]:




