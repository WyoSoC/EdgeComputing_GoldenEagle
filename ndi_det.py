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
SOURCE_MATCH = "birddog"   # part of NDI source name (case-insensitive)
VIDEO_RES = (480, 288)     # (W, H) for processing
REC_RES = (480, 288)       # (W, H) for recording (match to avoid implicit resizes)
RECORD_DURATION = 20       # seconds
PTZ_COOLDOWN = 0.35        # sec between moves (rate-limit metadata)
DEAD_ZONE = 40             # +/- px around center tolerated
THRESHOLD = 100            # extra before pan
PAN_SPEED = 0.30           # -1..+1 , change as needed
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
def ndi_init_and_find_source(match: str):
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
        name = s.ndi_name  # already a Python str
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

def ndi_send_xml(recv, xml_str: str):
    """
    Send PTZ XML metadata to the camera via the receiver handle.
    Advanced SDK bindings require a MetadataFrame object (not a plain str).
    This helper supports both Advanced and legacy bindings.
    """
    # Advanced-style
    try:
        meta = ndi.MetadataFrame()
        meta.data = xml_str      # plain string is OK
        meta.timecode = 0        # 0 => synthesize
        ndi.recv_send_metadata(recv, meta)
        return
    except Exception:
        pass

    # Legacy-style (C-struct)
    try:
        meta = ndi.NDIlib_metadata_frame_t()
        payload = xml_str.encode("utf-8")
        meta.p_data = payload
        meta.length = len(payload)
        meta.timecode = 0
        ndi.recv_send_metadata(recv, meta)
        return
    except Exception as e:
        raise RuntimeError(f"Failed to send PTZ metadata: {e}")

def ptz_pan_tilt_speed(recv, pan_speed, tilt_speed):
    # Clamp to [-1.0, +1.0] just to be safe
    pan_speed = max(-1.0, min(1.0, float(pan_speed)))
    tilt_speed = max(-1.0, min(1.0, float(tilt_speed)))
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
    last_move_time = 0.0

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

                # Estimate FPS in a robust way
                now = time.time()
                fps_window.append(now)
                if len(fps_window) >= 2:
                    real_fps = len(fps_window) / (fps_window[-1] - fps_window[0])
                    VIDEO_FPS_DYNAMIC = max(5, min(30, round(real_fps)))
                else:
                    VIDEO_FPS_DYNAMIC = 30

                # YOLO inference (class 0 = person)
                results = model(frame, classes=[0], verbose=False)[0]
                detections = []
                for box in results.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0].item())
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "person"))

                # DeepSort tracking
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

                    # Pan decision
                    dx = cx - frame_cx
                    move = None
                    if dx > THRESHOLD + DEAD_ZONE:
                        move = +PAN_SPEED
                    elif dx < -THRESHOLD - DEAD_ZONE:
                        move = -PAN_SPEED

                    # Send PTZ command (rate-limited)
                    if move is not None and (time.time() - last_move_time > PTZ_COOLDOWN):
                        print(" Pan:", "right" if move > 0 else "left")
                        ptz_pan_tilt_speed(recv, -move, 0.0)  # -move because the movement works correctly when mirrored
                        time.sleep(0.30)  # short burst
                        ptz_pan_tilt_speed(recv, 0.0, 0.0)
                        last_move_time = time.time()

                    break  # only act on first confirmed track

                # Recording control
                if detected and not is_recording:
                    record_start_time = time.time()
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn_with = f"videos/with_box/output_{ts}_with_box.mp4"
                    fn_no = f"videos/without_box/output_{ts}_no_box.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    print(f" Start recording @ ~{VIDEO_FPS_DYNAMIC} FPS")
                    video_writer_with_box = cv2.VideoWriter(fn_with, fourcc, VIDEO_FPS_DYNAMIC, REC_RES)
                    video_writer_no_box = cv2.VideoWriter(fn_no, fourcc, VIDEO_FPS_DYNAMIC, REC_RES)
                    is_recording = True

                if is_recording:
                    video_writer_with_box.write(frame)
                    video_writer_no_box.write(frame_no_box)
                    if time.time() - record_start_time >= RECORD_DURATION:
                        print(" Finished 20-second recording.")
                        video_writer_with_box.release()
                        video_writer_no_box.release()
                        is_recording = False

                if SHOW_WINDOW:
                    cv2.imshow("NDI Tracking + PTZ", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            elif t == ndi.FRAME_TYPE_METADATA and m is not None:
                # Free incoming metadata frame from camera if any
                try:
                    ndi.recv_free_metadata(recv, m)  # works in many builds
                except Exception:
                    # Some bindings use a different free signature/no recv arg
                    try:
                        ndi.recv_free_metadata(m)
                    except Exception:
                        pass

            elif t == ndi.FRAME_TYPE_NONE:
                continue

    finally:
        # Best-effort stop motion
        try:
            if recv:
                ptz_pan_tilt_speed(recv, 0.0, 0.0)
        except Exception:
            pass
        if SHOW_WINDOW:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if finder:
            try:
                ndi.find_destroy(finder)
            except Exception:
                pass
        if recv:
            try:
                ndi.recv_destroy(recv)
            except Exception:
                pass
        try:
            ndi.destroy()
        except Exception:
            pass

if __name__ == "__main__":
    main()
