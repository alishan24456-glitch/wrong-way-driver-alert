#!/usr/bin/env python3
"""
Wrong-Way Driver Alert -- direction check on a divided highway
================================================================

An elevated camera watches a divided motorway. YOLOv8n vehicle
detection (OpenCV DNN, no torch / ultralytics at run time) plus a
plain centroid tracker follows every vehicle. For each tracked vehicle
the system measures its direction of travel from the centroid movement
(a least-squares velocity fit over the last ~1.5 s) and compares it
with the road's legal direction for that carriageway:

* the motorway is split by its central reservation (a line from the
  bottom of the frame up to the vanishing point): the LEFT carriageway
  and the RIGHT carriageway each have their own legal direction,
  measured on the demo clip (left -20 degrees, right +42 degrees --
  the road curves, so the two directions are not simply opposite);
* direction is judged by the DOT PRODUCT of the vehicle's unit
  velocity with the legal unit vector: above +0.5 is LEGAL, below
  -0.5 is WRONG WAY, in between is UNCERTAIN (too slow or turning);
* a vehicle needs 1 s of tracking and 25 px/s of speed before it is
  classified at all -- a parked or barely-moving vehicle is never
  flagged;
* WRONG WAY raises a red banner, a per-vehicle direction arrow and a
  timestamped event log. The board shows vehicles tracked, legal /
  wrong-way counts and wrong-way events.

Weights: ``yolov8n.onnx`` next to this script (12.8 MB, ships in this
folder). If it is missing, the script tries to export it from
``yolov8n.pt`` with Ultralytics (downloading the .pt if needed) and
otherwise exits with instructions.

Simulation mode (``--simulate``) renders the same divided highway from
the elevated camera: cars on the left carriageway drive up the road
and cars on the right carriageway drive down it, and ONE car drives
DOWN the left carriageway, against the traffic -- it is correctly
flagged WRONG WAY. The simulation's vehicle source finds the cars
with a colour mask + contours -- the direction, state and alert logic
are the exact same code as real mode, and the detector never sees the
simulation's ground truth. Simulation is an internal logic check; the
delivered demo is real footage (see README).

Honest framing: direction is judged against a documented legal
direction per carriageway for the scene; the system does not know the
road's real signage. A vehicle that is turning, changing lanes across
the reservation or crawling in congestion can read UNCERTAIN -- it is
never flagged on a guess.

Examples
--------
    python wrong_way_driver_alert.py                 (opens a file picker)
    python wrong_way_driver_alert.py --source pick   (same picker, explicit)
    python wrong_way_driver_alert.py --source road.mp4
    python wrong_way_driver_alert.py --source 0      (webcam)
    python wrong_way_driver_alert.py --simulate --max-seconds 30 --no-display

Press Q to quit. The video is written to ``output.mp4``, the
timestamped event log to ``output_events.log`` and the final counters
to ``output_summary.txt`` next to it.

Author: Abdul Hanan -- Computer Vision & AI Developer
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import urllib.request

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Layout (the output video is always 1280x720)
# ---------------------------------------------------------------------------
W, H = 1280, 720
TITLE_H = 46
PANEL_H = 150                     # stats panel at the bottom
VIDEO_W, VIDEO_H = W, H - TITLE_H - PANEL_H
BG = (24, 26, 32)
PANEL_BG = (32, 35, 46)
WHITE = (255, 255, 255)
GREY = (160, 166, 178)
GREEN = (90, 200, 90)
AMBER = (60, 170, 255)
RED = (60, 60, 235)
YELLOW = (0, 210, 235)
BLUE = (220, 160, 60)

# ---------------------------------------------------------------------------
# Detection + direction tuning
# ---------------------------------------------------------------------------
DET_W, DET_H = 1280, 720          # detection runs on this frame size
DETECT_EVERY = 3                  # run YOLO every 3rd frame (10 fps)
CONF = 0.35
NMS_THRESH = 0.30
VEHICLE_CLASSES = {2, 3, 5, 7}    # car, motorcycle, bus, truck (COCO)
MATCH_DIST = 80                   # px, centroid match radius
MAX_MISSED = 15                   # detection rounds before a track drops
PT_URL = ("https://github.com/ultralytics/assets/releases/"
          "download/v8.2.0/yolov8n.pt")

# Central reservation: the line splitting the two carriageways, from
# the bottom of the frame up toward the vanishing point. Measured on
# the demo clip (Pexels 12420164): the reservation runs from x=660 at
# the bottom to about x=800 near the top of the frame.
SPLIT_X0 = 660.0                  # x of the split at the bottom (y=720)
SPLIT_SLOPE = 0.245               # px the split moves right per px up
# Legal directions per carriageway, in degrees (0 = right, 90 = down,
# image coordinates). Measured on the demo clip with an IoU-gated
# tracker: left carriageway traffic moves up-right at about -20 deg,
# right carriageway traffic moves down-right at about +42 deg -- the
# road curves, so the two are not simply opposite.
LEGAL_LEFT_DEG = -20.0
LEGAL_RIGHT_DEG = 42.0
# Classification: dot(unit velocity, legal unit) above this is LEGAL,
# below minus this is WRONG WAY, between is UNCERTAIN.
DOT_THRESHOLD = 0.5
MIN_SPEED = 25.0                  # px/s: slower than this is not judged
MIN_AGE_S = 1.0                   # seconds of tracking before judging
VEL_WINDOW_S = 1.5                # velocity fit window
ALERT_HOLD_S = 1.0                # keep the banner this long after the
                                  # last wrong-way vehicle leaves


def _unit(deg):
    r = math.radians(deg)
    return (math.cos(r), math.sin(r))


LEGAL_LEFT = _unit(LEGAL_LEFT_DEG)
LEGAL_RIGHT = _unit(LEGAL_RIGHT_DEG)


def split_x(y):
    """x of the central reservation at image row y."""
    return SPLIT_X0 + (DET_H - y) * SPLIT_SLOPE


# ---------------------------------------------------------------------------
# Vehicle sources -- real mode (YOLOv8 via OpenCV DNN) and simulation
# (colour mask). Both return boxes in detection-frame coordinates.
# ---------------------------------------------------------------------------
def ensure_weights(path):
    """Return a usable yolov8n.onnx path, exporting it if needed."""
    if os.path.exists(path):
        return path
    print(f"Weights not found at {path}; trying Ultralytics export ...")
    try:
        from ultralytics import YOLO
        pt = os.path.join(os.path.dirname(path) or ".", "yolov8n.pt")
        if not os.path.exists(pt):
            print("Downloading yolov8n.pt ...")
            urllib.request.urlretrieve(PT_URL, pt)
        print("Exporting yolov8n.pt -> yolov8n.onnx (one-time) ...")
        YOLO(pt).export(format="onnx", imgsz=640)
        exported = os.path.join(os.path.dirname(pt) or ".", "yolov8n.onnx")
        if os.path.exists(exported) and exported != path:
            os.replace(exported, path)
        return path
    except Exception as exc:  # noqa: BLE001 - report and stop cleanly
        print(f"Could not obtain weights: {exc}")
        print("Download yolov8n.onnx and place it next to this script, or")
        print("install ultralytics (`pip install ultralytics`) and re-run.")
        sys.exit(1)


class YoloVehicleDetector:
    """YOLOv8n vehicle detection through OpenCV DNN."""

    def __init__(self, weights_path):
        self.net = cv2.dnn.readNetFromONNX(ensure_weights(weights_path))

    def detect(self, frame):
        """frame is a DET_W x DET_H image; returns [x, y, w, h, conf]."""
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # [8400, 84]: cx cy w h + 80 scores
        scores = out[:, 4:].max(axis=1)
        classes = out[:, 4:].argmax(axis=1)
        keep = (scores > CONF) & np.isin(classes, list(VEHICLE_CLASSES))
        sh, sw = frame.shape[:2]
        boxes = []
        for (cx, cy, bw, bh), s in zip(out[keep, :4], scores[keep]):
            boxes.append([float((cx - bw / 2) * sw / 640),
                          float((cy - bh / 2) * sh / 640),
                          float(bw * sw / 640), float(bh * sh / 640),
                          float(s)])
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes([b[:4] for b in boxes],
                               [b[4] for b in boxes], CONF, NMS_THRESH)
        return [boxes[int(i)] for i in idx.flatten()] if len(idx) else []


def sim_vehicle_boxes(frame):
    """Colour-blob vehicle source for the rendered highway."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([95, 120, 80]),
                       np.array([115, 255, 255]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        if cv2.contourArea(c) < 300:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([float(x), float(y), float(w), float(h), 1.0])
    return boxes


# ---------------------------------------------------------------------------
# Direction tracker -- centroid tracking + per-vehicle travel direction
# ---------------------------------------------------------------------------
def _fit_velocity(history):
    """Least-squares (vx, vy) in px/s over [(t, x, y), ...]; None if thin."""
    if len(history) < 5:
        return None
    t0 = history[0][0]
    ts = [t - t0 for t, _, _ in history]
    if ts[-1] - ts[0] < 0.5:
        return None
    n = len(ts)
    mt = sum(ts) / n
    mx = sum(x for _, x, _ in history) / n
    my = sum(y for _, _, y in history) / n
    den = sum((tt - mt) ** 2 for tt in ts)
    if den <= 0:
        return None
    vx = sum((tt - mt) * (xx - mx)
             for tt, xx in zip(ts, [x for _, x, _ in history])) / den
    vy = sum((tt - mt) * (yy - my)
             for tt, yy in zip(ts, [y for _, _, y in history])) / den
    return (vx, vy)


class DirectionTracker:
    """Tracks vehicles, judges each one's direction of travel."""

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = 0
        self.video_frames = 0
        self.events = []
        self.detections = 0
        self.vehicles_seen = 0
        self.wrong_way_events = 0
        self.wrong_way_seconds = 0.0
        self.wrong_way_ids = set()
        self.legal_count = 0
        self.first_wrong_way_t = None
        self.alert = "CLEAR"
        self._alert_since = None
        self._last_t = None
        self.wrong_way_now = 0
        self.legal_now = 0

    def _log(self, t, text):
        self.events.append((t, text))

    @staticmethod
    def _centre(box):
        x, y, w, h, _ = box
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _iou(a, b):
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _band(cx, cy):
        return "LEFT" if cx < split_x(cy) else "RIGHT"

    def reset(self, t):
        """Drop all tracks: the source looped, so the scene restarted."""
        n = len(self.tracks)
        self.tracks.clear()
        self._log(t, f"source looped -- {n} in-flight tracks reset "
                     f"(directions restart from zero)")

    def update(self, boxes, t):
        """One detection round. Returns the alert state + live stats."""
        self.frames += 1
        self.detections += len(boxes)
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        used = set()
        for box in boxes:
            cx, cy = self._centre(box)
            best, best_d = None, MATCH_DIST
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                d = math.hypot(cx - tr["cx"], cy - tr["cy"])
                if d >= best_d:
                    continue
                # A detection must also overlap the track's last box, so
                # a track cannot jump onto a different car beside it.
                if self._iou(box, tr["box"]) <= 0.05:
                    continue
                best, best_d = tid, d
            if best is None:
                tid = self.next_id
                self.next_id += 1
                self.vehicles_seen += 1
                self.tracks[tid] = {"id": tid, "cx": cx, "cy": cy,
                                    "box": box, "missed": 0,
                                    "born_t": t, "history": [],
                                    "vx": 0.0, "vy": 0.0, "speed": 0.0,
                                    "state": "MEASURING", "band": None,
                                    "angle": None}
                best = tid
            used.add(best)
            tr = self.tracks[best]
            tr.update({"cx": cx, "cy": cy, "box": box, "missed": 0})
            tr["band"] = self._band(cx, cy)
            tr["history"].append((t, cx, cy))
            tr["history"] = [(tt, xx, yy) for tt, xx, yy in tr["history"]
                             if t - tt <= VEL_WINDOW_S]
            old_state = tr["state"]
            v = _fit_velocity(tr["history"])
            age = t - tr["born_t"]
            if v is None or age < MIN_AGE_S:
                new_state = "MEASURING"
                tr["vx"], tr["vy"] = (0.0, 0.0) if v is None else v
                tr["speed"] = 0.0 if v is None else math.hypot(*v)
                tr["angle"] = None
            else:
                vx, vy = v
                tr["vx"], tr["vy"] = vx, vy
                speed = math.hypot(vx, vy)
                tr["speed"] = speed
                tr["angle"] = math.degrees(math.atan2(vy, vx))
                if speed < MIN_SPEED:
                    new_state = "TOO SLOW"
                else:
                    legal = (LEGAL_LEFT if tr["band"] == "LEFT"
                             else LEGAL_RIGHT)
                    ux, uy = vx / speed, vy / speed
                    dot = ux * legal[0] + uy * legal[1]
                    if dot >= DOT_THRESHOLD:
                        new_state = "LEGAL"
                    elif dot <= -DOT_THRESHOLD:
                        new_state = "WRONG WAY"
                    else:
                        new_state = "UNCERTAIN"
            tr["state"] = new_state
            if new_state != old_state:
                if new_state == "WRONG WAY":
                    self.wrong_way_events += 1
                    self.wrong_way_ids.add(tr["id"])
                    if self.first_wrong_way_t is None:
                        self.first_wrong_way_t = t
                    legal_deg = (LEGAL_LEFT_DEG if tr["band"] == "LEFT"
                                 else LEGAL_RIGHT_DEG)
                    self._log(t, f"WRONG WAY -- vehicle #{tr['id']} in "
                                 f"the {tr['band']} carriageway travelling "
                                 f"{tr['angle']:.0f} deg "
                                 f"(legal {legal_deg:.0f} deg)")
                elif old_state == "WRONG WAY":
                    self._log(t, f"vehicle #{tr['id']} no longer wrong "
                                 f"way ({new_state})")
                elif new_state == "LEGAL" and old_state == "MEASURING":
                    self.legal_count += 1
        for tid in list(self.tracks):
            if tid in used:
                continue
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > MAX_MISSED:
                del self.tracks[tid]
        # -- live stats ------------------------------------------------------
        wrong = [tr for tr in self.tracks.values()
                 if tr["state"] == "WRONG WAY"]
        legal = [tr for tr in self.tracks.values()
                 if tr["state"] == "LEGAL"]
        self.wrong_way_now = len(wrong)
        self.legal_now = len(legal)
        self.wrong_way_seconds += len(wrong) * dt
        # -- alert: WRONG WAY while any vehicle is wrong way, held 1 s
        if wrong:
            self._alert_since = t
            new_alert = "WRONG WAY"
        elif (self._alert_since is not None
              and t - self._alert_since < ALERT_HOLD_S):
            new_alert = self.alert
        else:
            new_alert = "CLEAR"
            self._alert_since = None
        if new_alert != self.alert:
            self.alert = new_alert
            if new_alert == "CLEAR":
                self._log(t, "road clear -- WRONG WAY banner down")
        return {"alert": self.alert, "t": t}

    def summary(self):
        lines = [
            "WRONG-WAY DRIVER ALERT -- RUN SUMMARY",
            f"frames processed: {self.video_frames}",
            f"tracker updates: {self.frames}",
            f"vehicle detections (summed): {self.detections}",
            f"vehicles tracked: {self.vehicles_seen}",
            f"vehicles judged LEGAL: {self.legal_count}",
            f"WRONG WAY events: {self.wrong_way_events}",
            f"distinct wrong-way vehicles: {len(self.wrong_way_ids)}",
            f"seconds with a wrong-way vehicle: "
            f"{self.wrong_way_seconds:.1f} s",
            ("first WRONG WAY at: "
             + (f"{self.first_wrong_way_t:.1f} s"
                if self.first_wrong_way_t is not None else "never")),
            f"final alert: {self.alert}",
            f"events logged: {len(self.events)}",
        ]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Simulation renderer -- a divided highway from the elevated camera.
# The tracker only ever sees these rendered pixels.
# ---------------------------------------------------------------------------
class HighwaySim:
    """Renders a divided highway; one car drives against the traffic."""

    def __init__(self, w, h, fps=30):
        self.w, self.h = w, h
        self.fps = fps
        self.frame_idx = 0
        self.rng = random.Random(20261119)
        # Lanes: (band, base position along the direction, speed px/s,
        # size). Left carriageway traffic moves up-right (-20 deg);
        # right carriageway traffic moves down-right (+42 deg).
        self.cars = []
        self.next_spawn = {"LEFT": 0.5, "RIGHT": 1.0}
        self.spawned = {"LEFT": 0, "RIGHT": 0}
        # The wrong-way car: drives DOWN the left carriageway (opposite
        # to its legal -20 deg) starting near the top at t=8 s. It makes
        # exactly one pass -- it never respawns.
        self.wrong_way = {"active": False, "x": 0.0, "y": 0.0,
                          "done": False}

    def _draw_road(self, img):
        cv2.rectangle(img, (0, 0), (self.w, self.h), (70, 72, 66), -1)
        # Sky / trees at the top.
        cv2.rectangle(img, (0, 0), (self.w, 90), (110, 130, 120), -1)
        # Left carriageway: polygon from bottom-left to top-centre.
        left = np.array([(0, self.h), (0, 320), (560, 90), (760, 90),
                         (760, self.h)], dtype=np.int32)
        cv2.fillPoly(img, [left], (64, 64, 68))
        # Right carriageway: polygon from bottom-centre-right to top-right.
        right = np.array([(800, self.h), (800, 90), (1000, 90),
                          (self.w, 300), (self.w, self.h)],
                         dtype=np.int32)
        cv2.fillPoly(img, [right], (64, 64, 68))
        # Central reservation between them.
        res = np.array([(760, 90), (800, 90), (800, self.h), (760, self.h)],
                       dtype=np.int32)
        cv2.fillPoly(img, [res], (90, 95, 80))
        for y in range(100, self.h, 60):
            cv2.circle(img, (780, y), 5, (180, 180, 170), -1)
        # Lane dashes on both carriageways.
        for x0, y0, x1, y1 in ((380, 320, 180, 700), (880, 200, 1150, 650)):
            for i in range(6):
                fx = x0 + (x1 - x0) * i / 5
                fy = y0 + (y1 - y0) * i / 5
                cv2.circle(img, (int(fx), int(fy)), 4, (200, 200, 190), -1)

    @staticmethod
    def _draw_car(img, x, y, size, colour=(255, 130, 30)):
        w = int(64 * size)
        h = int(34 * size)
        x0, y0 = int(x - w / 2), int(y - h / 2)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), colour, -1)
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (200, 100, 20), 2)
        cv2.rectangle(img, (x0 + int(w * 0.15), y0 + int(h * 0.2)),
                      (x0 + int(w * 0.85), y0 + int(h * 0.8)),
                      (40, 40, 46), -1)

    def frame(self):
        idx = self.frame_idx
        self.frame_idx += 1
        t = idx / self.fps
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._draw_road(img)

        # Spawn legal traffic: left carriageway cars travel up-right
        # (-20 deg), right carriageway cars travel down-right (+42 deg).
        # Each car starts at its carriageway's entry edge and drives
        # along its direction until it leaves the frame.
        for band in ("LEFT", "RIGHT"):
            if t >= self.next_spawn[band] and self.spawned[band] < 10:
                if band == "LEFT":
                    x, y = self.rng.uniform(60, 500), self.h + 20
                    vx, vy = 46.0, -17.0      # -20 deg, ~49 px/s
                else:
                    x, y = self.rng.uniform(900, 1100), 110
                    vx, vy = 33.0, 30.0       # +42 deg, ~45 px/s
                size = self.rng.uniform(0.8, 1.1)
                self.cars.append({"band": band, "x": x, "y": y,
                                  "vx": vx, "vy": vy, "size": size})
                self.spawned[band] += 1
                self.next_spawn[band] = t + 2.5
        # The wrong-way car enters the left carriageway from the top at
        # t=8 s and drives DOWN it (opposite to the legal -20 deg).
        if t >= 8.0 and not self.wrong_way["active"] \
                and not self.wrong_way["done"]:
            self.wrong_way.update({"active": True, "x": 420.0, "y": 110.0})
        for car in list(self.cars):
            car["x"] += car["vx"] / self.fps
            car["y"] += car["vy"] / self.fps
            self._draw_car(img, car["x"], car["y"], car["size"])
            if (car["x"] < -80 or car["x"] > self.w + 80
                    or car["y"] < 40 or car["y"] > self.h + 80):
                self.cars.remove(car)
        if self.wrong_way["active"]:
            ww = self.wrong_way
            # Opposite to the left carriageway's legal -20 deg: down-left
            # (~160 deg), i.e. back down the carriageway toward the
            # camera, against the traffic.
            ww["x"] -= 46.0 / self.fps
            ww["y"] += 17.0 / self.fps
            self._draw_car(img, ww["x"], ww["y"], 1.0,
                           colour=(255, 130, 30))
            if ww["y"] > self.h + 40 or ww["x"] < -40:
                self.wrong_way["active"] = False
                self.wrong_way["done"] = True
        return img


# ---------------------------------------------------------------------------
# Rendering -- draws tracker output onto the 1280x720 canvas
# ---------------------------------------------------------------------------
_ARROWS = {}


def _arrow_for(angle_deg):
    """An 8-way arrow character for a travel direction (degrees)."""
    if angle_deg is None:
        return "?"
    a = (angle_deg + 360) % 360
    if a < 22.5 or a >= 337.5:
        return "->"
    if a < 67.5:
        return "v>"   # down-right (image y grows downward)
    if a < 112.5:
        return "v"
    if a < 157.5:
        return "<v"
    if a < 202.5:
        return "<-"
    if a < 247.5:
        return "<^"
    if a < 292.5:
        return "^"
    return "^>"


def draw_canvas(det_frame, tracker, res, elapsed, mode_label, boxes):
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = BG
    cv2.rectangle(canvas, (0, 0), (W, TITLE_H), PANEL_BG, -1)
    cv2.putText(canvas, "WRONG-WAY DRIVER ALERT  --  divided highway",
                (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, WHITE, 2)
    cv2.putText(canvas, mode_label, (W - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    vh, vw = det_frame.shape[:2]
    scale = min(VIDEO_W / vw, VIDEO_H / vh)
    nw, nh = int(vw * scale), int(vh * scale)
    resized = cv2.resize(det_frame, (nw, nh))
    ox = (VIDEO_W - nw) // 2
    oy = TITLE_H + (VIDEO_H - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    sx, sy = nw / vw, nh / vh

    def _mx(x):
        return ox + int(x * sx)

    def _my(y):
        return oy + int(y * sy)

    # Central reservation line.
    pts = np.array([(_mx(split_x(y)), _my(y)) for y in range(0, DET_H, 20)],
                   dtype=np.int32)
    cv2.polylines(canvas, [pts], False, YELLOW, 2)
    cv2.putText(canvas, "CENTRAL RESERVATION",
                (_mx(split_x(360)) + 8, _my(360)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
    cv2.putText(canvas, "LEFT  legal -20 deg",
                (ox + 16, oy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 2)
    cv2.putText(canvas, "RIGHT  legal +42 deg",
                (_mx(split_x(360)) + 16, oy + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 2)

    # Vehicle boxes with their direction arrow + verdict. The far end of
    # the motorway packs many small cars close together, so a label is
    # only drawn when the box is big enough for it to be read -- small
    # cars still get their coloured box and still count on the board.
    for tid, tr in tracker.tracks.items():
        x, y, w, h, _ = tr["box"]
        state = tr["state"]
        colour = {"LEGAL": GREEN, "WRONG WAY": RED, "UNCERTAIN": AMBER,
                  "TOO SLOW": GREY, "MEASURING": GREY}[state]
        cv2.rectangle(canvas, (_mx(x), _my(y)), (_mx(x + w), _my(y + h)),
                      colour, 2 if state != "WRONG WAY" else 3)
        if w < 100:
            continue
        arrow = _arrow_for(tr["angle"])
        tag = f"#{tid} {arrow}"
        if tr["angle"] is not None:
            tag += f" {tr['angle']:.0f} deg"
        tag += f"  {state}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lx = _mx(x)
        if lx + tw + 8 > W - 4:
            lx = max(4, W - 4 - tw - 8)
        ly = _my(y) - 8
        if ly - th < oy + 4:
            ly = _my(y + h) + th + 10
        cv2.rectangle(canvas, (lx, ly - th - 4),
                      (lx + tw + 8, ly + 4), (24, 26, 32), -1)
        cv2.putText(canvas, tag, (lx + 4, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    # Bottom panel: board + alert banner.
    py = TITLE_H + VIDEO_H
    cv2.rectangle(canvas, (0, py), (W, H), PANEL_BG, -1)
    alert = tracker.alert
    acol = RED if alert == "WRONG WAY" else GREEN
    cv2.rectangle(canvas, (0, py), (W, py + 6), acol, -1)
    banner = f"STATUS: {alert}"
    if alert == "WRONG WAY":
        banner += "  !!! VEHICLE DRIVING AGAINST TRAFFIC !!!"
    cv2.putText(canvas, banner, (16, py + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, acol, 2)
    cv2.putText(canvas, f"t = {elapsed:.1f} s", (16, py + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas, f"legal now: {tracker.legal_now}",
                (300, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    cv2.putText(canvas, f"wrong way now: {tracker.wrong_way_now}",
                (520, py + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                RED if tracker.wrong_way_now else WHITE, 2)
    cv2.putText(canvas,
                f"vehicles: {tracker.vehicles_seen}   judged legal: "
                f"{tracker.legal_count}   WRONG WAY events: "
                f"{tracker.wrong_way_events}   wrong-way seconds: "
                f"{tracker.wrong_way_seconds:.1f}",
                (16, py + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    return canvas


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Choose a video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                       ("All files", "*.*")])
        root.destroy()
        return path
    except Exception:
        return ""


def open_source(source):
    if source == "pick":
        path = pick_file()
        if not path:
            print("No file chosen.")
            sys.exit(0)
        source = path
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    if not cap.isOpened():
        print(f"Could not open source: {source}")
        sys.exit(1)
    return cap


def main():
    global LEGAL_LEFT, LEGAL_RIGHT, LEGAL_LEFT_DEG, LEGAL_RIGHT_DEG
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--source", default="pick",
                    help="video file, webcam index, or 'pick'")
    ap.add_argument("--simulate", action="store_true",
                    help="rendered divided highway instead of a camera/video")
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--loop", action="store_true",
                    help="loop a short source until --max-seconds is reached")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--weights", default=None,
                    help="path to yolov8n.onnx (default: next to the script)")
    ap.add_argument("--legal-left", type=float, default=LEGAL_LEFT_DEG,
                    help="legal direction (deg) for the left carriageway")
    ap.add_argument("--legal-right", type=float, default=LEGAL_RIGHT_DEG,
                    help="legal direction (deg) for the right carriageway")
    args = ap.parse_args()

    LEGAL_LEFT_DEG = args.legal_left
    LEGAL_RIGHT_DEG = args.legal_right
    LEGAL_LEFT = _unit(args.legal_left)
    LEGAL_RIGHT = _unit(args.legal_right)

    fps = 30
    here = os.path.dirname(os.path.abspath(__file__))
    detector = None if args.simulate else YoloVehicleDetector(
        args.weights or os.path.join(here, "yolov8n.onnx"))
    tracker = DirectionTracker()
    sim = HighwaySim(DET_W, DET_H, fps) if args.simulate else None
    cap = None if args.simulate else open_source(args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap is not None else fps
    if not src_fps or src_fps <= 0:
        src_fps = fps
    step = max(1, round(src_fps / fps))  # sample the source at ~30 fps

    out_path = os.path.abspath(args.out)
    base, _ = os.path.splitext(out_path)
    events_path = base + "_events.log"
    summary_path = base + "_summary.txt"
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    mode_label = "SIMULATION (logic check)" if args.simulate else "REAL MODE"

    n_out = 0
    max_frames = int(args.max_seconds * fps)
    src_idx = 0
    boxes = []
    while n_out < max_frames:
        t = n_out / fps
        if args.simulate:
            raw = sim.frame()
            det_frame = raw
            boxes = sim_vehicle_boxes(det_frame)
            res = tracker.update(boxes, t)
        else:
            ok, frame = cap.read()
            src_idx += 1
            if not ok:
                if args.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    src_idx = 1
                    tracker.reset(t)
                if not ok:
                    break
            if (src_idx - 1) % step != 0:
                continue
            det_frame = cv2.resize(frame, (DET_W, DET_H))
            if n_out % DETECT_EVERY == 0:
                boxes = detector.detect(det_frame)
                res = tracker.update(boxes, t)
            else:
                res = {"alert": tracker.alert, "t": t}
        canvas = draw_canvas(det_frame, tracker, res, t, mode_label, boxes)
        writer.write(canvas)
        n_out += 1
        if not args.no_display:
            try:
                cv2.imshow("Wrong-Way Driver Alert", canvas)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
            except cv2.error:
                pass

    writer.release()
    if cap is not None:
        cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    tracker.video_frames = n_out
    with open(events_path, "w") as f:
        for t, text in tracker.events:
            f.write(f"{t:6.1f}s  {text}\n")
    with open(summary_path, "w") as f:
        f.write(tracker.summary())
    print(tracker.summary())
    print(f"video:   {out_path}")
    print(f"events:  {events_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
