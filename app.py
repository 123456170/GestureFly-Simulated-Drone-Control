"""GestureFly Simulated Drone Control — single-file Streamlit app.

Privacy: all processing is local. No network upload. Recording happens only when the user presses Record.
License notes embedded because only two files were requested:
- Streamlit: Apache-2.0
- NumPy: BSD-3-Clause
- pandas: BSD-3-Clause
- OpenCV-Python: Apache-2.0
- MediaPipe: Apache-2.0
- imageio: MIT
- imageio-ffmpeg: BSD-style wrapper; bundled FFmpeg components are LGPL/GPL depending on build
- Pillow: MIT/CUDA-style historical license terms vary by component, but it is free/open-source

This file contains simulated control only. Do not connect to a real drone.
"""

from __future__ import annotations

import copy
import csv
import glob
import io
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    import imageio
except Exception:
    imageio = None


CONFIG: Dict[str, Any] = {
    "fps": 30,                           # target perception/control frequency in Hz
    "frame_width": 640,                  # camera capture width in pixels
    "frame_height": 480,                 # camera capture height in pixels
    "deadzone": 0.12,                    # normalized hand-offset deadzone before motion
    "curve_p": 1.6,                      # exponent for nonlinear offset curve, normally 1..2
    "k_xy": 1.8,                         # horizontal velocity scale in (m/s)/normalized-unit
    "k_z": 1.2,                          # vertical velocity scale in (m/s)/normalized-unit
    "k_yaw": 1.5,                        # yaw-rate scale in (rad/s)/normalized-unit
    "tilt_deadzone": 0.08,               # normalized palm-tilt deadzone
    "tilt_p": 1.4,                       # exponent for palm-tilt curve
    "max_vxy": 2.0,                      # maximum horizontal speed in m/s
    "max_vz": 1.0,                       # maximum vertical speed in m/s
    "max_yaw_rate": 1.2,                 # maximum yaw rate in rad/s
    "accel_limit": 3.0,                  # maximum linear acceleration in m/s^2
    "yaw_accel_limit": 2.5,              # maximum angular acceleration in rad/s^2
    "hover_timeout_s": 0.5,              # hand-lost time before hover failsafe
    "land_timeout_s": 3.0,               # hand-lost time before automatic landing
    "land_rate": 0.45,                   # safe landing descent rate in m/s
    "crash_vz": 1.2,                     # downward speed treated as crash in m/s
    "geofence_radius_m": 12.0,           # horizontal geofence radius in metres
    "geofence_alt_min_m": 0.0,           # minimum allowed altitude in metres
    "geofence_alt_max_m": 15.0,          # maximum allowed altitude in metres
    "arming_hold_s": 1.0,                # open-palm hold time to arm risky actions
    "land_hold_s": 0.7,                  # additional open-palm hold time to request landing
    "refractory_s": 0.45,                # minimum time between repeated gesture events
    "hysteresis": 0.08,                  # score margin required to switch gesture
    "pinch_open": 0.35,                  # thumb-index pinch ratio considered open
    "pinch_close": 0.18,                 # thumb-index pinch ratio considered closed
    "finger_extension_threshold": 1.12,  # tip/wrist distance ratio vs MCP/wrist for extended finger
    "one_euro_cutoff": 1.0,              # One Euro filter base cutoff in Hz
    "one_euro_beta": 0.02,               # One Euro filter speed adaptation factor
    "one_euro_d_cutoff": 1.0,            # One Euro filter derivative cutoff in Hz
    "demo_cycle_s": 45.0,                # synthetic demo loop duration in seconds
    "gate_radius_m": 0.9,                # gate passing radius in metres
    "gate_alt_band_m": 1.0,              # gate altitude tolerance in metres
    "task_kp_xy": 1.1,                   # task autopilot horizontal proportional gain
    "task_kp_z": 0.9,                    # task autopilot vertical proportional gain
    "keyboard_latency_s": 0.10,          # assumed human keyboard latency in evaluation
    "gesture_latency_s": 0.18,           # assumed gesture pipeline latency in evaluation
    "hand_landmarker_model_path": os.environ.get("GESTUREFLY_HAND_MODEL", ""),     # optional local Hand Landmarker .task path
    "gesture_recognizer_model_path": os.environ.get("GESTUREFLY_GESTURE_MODEL", ""), # optional local Gesture Recognizer .task path
    "media_dir": "gesturefly_media",     # local folder for generated video files
    "preview_duration_s": 8.0,           # auto-generated startup demo video length in seconds
    "preview_fps": 20,                   # frame rate for generated preview videos
    "capture_fps": 15,                   # frame rate for live session video capture
    "video_width": 640,                  # generated video width in pixels
    "video_height": 360,                 # generated video height in pixels
}

GATES: List[Tuple[float, float, float]] = [(3.0, 0.0, 1.8), (3.0, 4.0, 1.8), (-3.0, 4.0, 1.8)]
PAD: Tuple[float, float, float] = (0.0, 0.0, 0.0)
START_POS: Tuple[float, float, float] = (0.0, 0.0, 1.5)

ROUTE: List[np.ndarray] = [np.array(START_POS[:2], dtype=float)] + [np.array(g[:2], dtype=float) for g in GATES] + [np.array(PAD[:2], dtype=float)]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def vec2list(v: np.ndarray) -> List[float]:
    return [float(v[0]), float(v[1])]


def dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def palm_width(lm: np.ndarray) -> float:
    lm = np.asarray(lm, dtype=float)
    return max(dist(lm[5], lm[17]), 1e-6)


def palm_center(lm: np.ndarray) -> np.ndarray:
    lm = np.asarray(lm, dtype=float)
    return np.mean(lm[[0, 5, 9, 13, 17]], axis=0)


def pinch_ratio(lm: np.ndarray) -> float:
    lm = np.asarray(lm, dtype=float)
    return dist(lm[4], lm[8]) / palm_width(lm)


def shaped(x: float, deadzone: float, p: float, k: float) -> float:
    x = float(x)
    mag = max(0.0, abs(x) - float(deadzone))
    return float(k) * math.copysign(mag ** float(p), x)


def angle_diff(a: float, b: float) -> float:
    return float((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    t = clamp(float(np.dot(p - a, ab) / denom), 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def min_distance_point_polyline(p: np.ndarray, pts: List[np.ndarray]) -> float:
    p = np.asarray(p, dtype=float)
    if len(pts) == 0:
        return 0.0
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))
    return min(point_segment_distance(p, pts[i], pts[i + 1]) for i in range(len(pts) - 1))


class LowPassFilter:
    def __init__(self) -> None:
        self.s = 0.0
        self.initialized = False

    def filter(self, x: float, alpha: float) -> float:
        alpha = clamp(alpha, 0.0, 1.0)
        if not self.initialized:
            self.s = float(x)
            self.initialized = True
        else:
            self.s = alpha * float(x) + (1.0 - alpha) * self.s
        return self.s


class OneEuroFilter:
    def __init__(self, cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.cutoff = float(cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_raw: Optional[float] = None
        self.last_time: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        if dt <= 0.0:
            return 1.0
        tau = 1.0 / (2.0 * math.pi * max(float(cutoff), 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: Optional[float] = None, dx: Optional[float] = None) -> float:
        x = float(x)
        t = time.perf_counter() if t is None else float(t)

        if self.last_time is None or self.last_raw is None:
            dt = 1.0 / 60.0
            dx = 0.0
        else:
            dt = t - self.last_time
            if dx is None:
                dx = (x - self.last_raw) / dt if dt > 0 else 0.0

        self.last_raw = x
        self.last_time = t

        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff, dt))


class OneEuroVector:
    def __init__(self, size: int, cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0) -> None:
        self.filters = [OneEuroFilter(cutoff=cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(int(size))]

    def filter(self, arr: np.ndarray, t: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        original_shape = arr.shape
        flat = arr.reshape(-1)
        out = np.empty_like(flat, dtype=float)
        for i, value in enumerate(flat):
            out[i] = self.filters[i].filter(float(value), t)
        return out.reshape(original_shape)


@dataclass
class GestureEvent:
    type: str
    hand: str
    t_ms: int
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)


class GestureDetector:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.filter: Optional[OneEuroVector] = None
        self.current = "none"
        self.previous_current = "none"
        self.conf = 0.0
        self.open_start: Optional[float] = None
        self.last_emit: Dict[str, float] = {}
        self.armed = False

    def _can_emit(self, typ: str, t: float) -> bool:
        return (t - self.last_emit.get(typ, -1e9)) > float(self.params.get("refractory_s", 0.45))

    def update(self, lm: Optional[np.ndarray], t: float, hand: str = "right") -> Tuple[List[GestureEvent], Dict[str, Any]]:
        events: List[GestureEvent] = []
        features: Dict[str, Any] = {
            "present": False,
            "gesture": "none",
            "confidence": 0.0,
            "pinch_ratio": 0.0,
            "palm_width": 0.0,
            "palm_center": [0.0, 0.0],
            "armed": self.armed,
            "extensions": [False, False, False, False],
        }

        if lm is None:
            self.current = "none"
            self.previous_current = "none"
            self.conf = 0.0
            self.open_start = None
            features["armed"] = self.armed
            return events, features

        lm_arr = np.asarray(lm, dtype=float)
        if lm_arr.shape != (21, 2):
            features["armed"] = self.armed
            return events, features

        if self.filter is None:
            self.filter = OneEuroVector(
                lm_arr.size,
                cutoff=float(self.params.get("one_euro_cutoff", 1.0)),
                beta=float(self.params.get("one_euro_beta", 0.02)),
                d_cutoff=float(self.params.get("one_euro_d_cutoff", 1.0)),
            )

        fsm = self.filter.filter(lm_arr, t)
        pw = palm_width(fsm)
        pc = palm_center(fsm)
        pr = pinch_ratio(fsm)

        extensions = []
        for tip, mcp in ((8, 5), (12, 9), (16, 13), (20, 17)):
            tip_wrist = dist(fsm[tip], fsm[0])
            mcp_wrist = dist(fsm[mcp], fsm[0])
            extensions.append(bool(tip_wrist > mcp_wrist * float(self.params.get("finger_extension_threshold", 1.12))))

        ext_count = sum(extensions)

        open_score = (ext_count / 4.0) * clamp((pr - float(self.params.get("pinch_open", 0.35))) / 0.25, 0.0, 1.0)
        fist_score = (1.0 - ext_count / 4.0) * clamp((0.55 - pr) / 0.25, 0.0, 1.0)
        pinch_score = clamp((float(self.params.get("pinch_close", 0.18)) + 0.08 - pr) / 0.16, 0.0, 1.0) * (0.5 + 0.5 * (ext_count / 4.0))

        scores = {"open": open_score, "fist": fist_score, "pinch": pinch_score}
        best = max(scores, key=scores.get)
        conf = float(scores[best])

        if conf < 0.35:
            if self.conf < 0.35:
                self.current = "none"
                self.conf = 0.0
            best = self.current
            conf = self.conf
        elif best != self.current:
            if conf > self.conf + float(self.params.get("hysteresis", 0.08)):
                self.current = best
                self.conf = conf
            else:
                best = self.current
                conf = self.conf
        else:
            self.conf = conf

        just_entered = best != self.previous_current

        if best == "open":
            if self.open_start is None:
                self.open_start = t
            hold = t - self.open_start
            if (not self.armed) and hold >= float(self.params.get("arming_hold_s", 1.0)) and self._can_emit("arm", t):
                self.armed = True
                self.last_emit["arm"] = t
                events.append(GestureEvent("arm", hand, int(t * 1000), conf, {"hold_s": hold}))
            elif self.armed and hold >= float(self.params.get("arming_hold_s", 1.0)) + float(self.params.get("land_hold_s", 0.7)) and self._can_emit("land_request", t):
                self.armed = False
                self.open_start = t
                self.last_emit["land_request"] = t
                events.append(GestureEvent("land_request", hand, int(t * 1000), conf, {"hold_s": hold}))
        else:
            self.open_start = None

        if best == "fist" and just_entered and self._can_emit("hover", t):
            self.last_emit["hover"] = t
            events.append(GestureEvent("hover", hand, int(t * 1000), conf, {}))

        if best == "pinch" and just_entered and self._can_emit("toggle_task", t):
            self.last_emit("toggle_task", t)
            events.append(GestureEvent("toggle_task", hand, int(t * 1000), conf, {}))

        features.update({
            "present": True,
            "gesture": best,
            "confidence": conf,
            "pinch_ratio": float(pr),
            "palm_width": float(pw),
            "palm_center": vec2list(pc),
            "armed": bool(self.armed),
            "extensions": [bool(x) for x in extensions],
        })

        self.previous_current = best
        return events, features


def transform_local(local: np.ndarray, center: np.ndarray, scale: float, rot: float, pitch: float) -> np.ndarray:
    pts = np.asarray(local, dtype=float).copy()
    pts[:, 1] *= (1.0 - 0.22 * clamp(pitch, -1.0, 1.0))
    c, s = math.cos(rot), math.sin(rot)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    pts = np.column_stack((x, y))
    pts = pts * float(scale) + np.asarray(center, dtype=float)
    return np.clip(pts, 0.02, 0.98)


class DemoHand:
    def __init__(self) -> None:
        self.open = np.array([
            [0.00, 0.00],   # 0 wrist
            [0.04, 0.02],   # 1 thumb CMC
            [0.08, 0.00],   # 2 thumb MCP
            [0.12, -0.01],  # 3 thumb IP
            [0.16, -0.02],  # 4 thumb tip
            [0.08, -0.05],  # 5 index MCP
            [0.09, -0.11],  # 6 index PIP
            [0.10, -0.17],  # 7 index DIP
            [0.10, -0.23],  # 8 index tip
            [0.02, -0.07],  # 9 middle MCP
            [0.02, -0.14],  # 10 middle PIP
            [0.02, -0.21],  # 11 middle DIP
            [0.02, -0.27],  # 12 middle tip
            [-0.04, -0.06], # 13 ring MCP
            [-0.05, -0.12], # 14 ring PIP
            [-0.06, -0.18], # 15 ring DIP
            [-0.06, -0.23], # 16 ring tip
            [-0.10, -0.04], # 17 pinky MCP
            [-0.12, -0.09], # 18 pinky PIP
            [-0.13, -0.14], # 19 pinky DIP
            [-0.13, -0.18], # 20 pinky tip
        ], dtype=float)

        self.fist = self.open.copy()
        self.fist[4] = [0.05, 0.02]
        self.fist[6] = [0.08, -0.04]
        self.fist[7] = [0.08, -0.02]
        self.fist[8] = [0.07, -0.02]
        self.fist[10] = [0.02, -0.05]
        self.fist[11] = [0.02, -0.04]
        self.fist[12] = [0.02, -0.03]
        self.fist[14] = [-0.04, -0.04]
        self.fist[15] = [-0.04, -0.03]
        self.fist[16] = [-0.04, -0.02]
        self.fist[18] = [-0.10, -0.03]
        self.fist[19] = [-0.10, -0.02]
        self.fist[20] = [-0.09, -0.01]

        self.pinch = self.open.copy()
        self.pinch[4] = [0.10, -0.20]
        self.pinch[8] = [0.10, -0.20]

    def static(self, pose: str) -> np.ndarray:
        if pose == "fist":
            base = self.fist
        elif pose == "pinch":
            base = self.pinch
        else:
            base = self.open
        return transform_local(base, np.array([0.50, 0.62], dtype=float), 1.0, 0.0, 0.0)

    def generate(self, t_abs: float, cycle_s: float = 45.0) -> Tuple[Optional[np.ndarray], str, str]:
        c = float(t_abs) % float(cycle_s)
        center = np.array([0.50, 0.62], dtype=float)
        scale = 1.0
        rot = 0.0
        pitch = 0.0
        pose = "open"
        present = True
        phase = "Idle"

        if c < 2.0:
            phase = "Open palm: arming"
        elif c < 8.0:
            u = (c - 2.0) / 6.0
            phase = "Gesture flight: gate 1"
            center += np.array([0.10 * math.sin(math.pi * u), -0.08 * u], dtype=float)
            rot = 0.18 * math.sin(2.0 * math.pi * u)
            pitch = 0.25 * u
            scale = 1.0 + 0.05 * math.sin(math.pi * u)
        elif c < 9.0:
            pose = "fist"
            phase = "Fist: hover"
        elif c < 15.0:
            u = (c - 9.0) / 6.0
            phase = "Gesture flight: gate 2"
            center += np.array([0.04, -0.14 - 0.04 * u], dtype=float)
            rot = -0.12
            pitch = -0.18 + 0.10 * u
            scale = 1.05
        elif c < 16.0:
            pose = "pinch"
            phase = "Pinch: task toggle"
        elif c < 22.0:
            u = (c - 16.0) / 6.0
            phase = "Gesture flight: gate 3"
            center += np.array([-0.16 * u, -0.16 + 0.05 * math.sin(math.pi * u)], dtype=float)
            rot = 0.22 * math.sin(math.pi * u)
            pitch = 0.12
            scale = 0.95
        elif c < 24.0:
            pose = "open"
            phase = "Open palm: land request"
            center += np.array([0.0, 0.02], dtype=float)
        elif c < 25.0:
            present = False
            phase = "Hand lost: hover failsafe"
        elif c < 29.0:
            present = False
            phase = "Hand lost: land failsafe"
        elif c < 31.0:
            phase = "Recovery: hand returned"
            center = np.array([0.50, 0.62], dtype=float)
        else:
            u = (c - 31.0) / max(1e-6, float(cycle_s) - 31.0)
            phase = "Idle sway"
            center += np.array([0.03 * math.sin(2.0 * math.pi * u), 0.02 * math.cos(3.0 * math.pi * u)], dtype=float)
            rot = 0.05 * math.sin(2.0 * math.pi * u)

        if not present:
            return None, "none", phase

        if pose == "fist":
            local = self.fist.copy()
        elif pose == "pinch":
            local = self.pinch.copy()
        else:
            local = self.open.copy()

        local = local + np.random.normal(0.0, 0.0012, local.shape)
        lm = transform_local(local, center, scale, rot, pitch)
        intended = pose
        return lm, intended, phase


def compute_neutral(lm: np.ndarray) -> Dict[str, Any]:
    lm = np.asarray(lm, dtype=float)
    pw = palm_width(lm)
    axis = lm[17] - lm[5]
    yaw = math.atan2(float(axis[1]), float(axis[0]))
    return {
        "center": vec2list(palm_center(lm)),
        "palm_width": float(pw),
        "roll": float(axis[1] / pw),
        "pitch": float((lm[9][1] - lm[0][1]) / pw),
        "yaw": float(yaw),
    }


def landmarks_to_command(lm: Optional[np.ndarray], neutral: Optional[Dict[str, Any]], params: Dict[str, Any]) -> np.ndarray:
    if lm is None or not neutral:
        return np.zeros(4, dtype=float)

    lm = np.asarray(lm, dtype=float)
    center = palm_center(lm)
    pw = palm_width(lm)

    dx = float(center[0] - float(neutral["center"][0]))
    dy = float(center[1] - float(neutral["center"][1]))
    width_ratio = float(pw / max(float(neutral["palm_width"]), 1e-6))

    axis = lm[17] - lm[5]
    roll = float(axis[1] / pw)
    pitch = float((lm[9][1] - lm[0][1]) / pw)
    yaw = float(math.atan2(float(axis[1]), float(axis[0])))

    roll_sig = roll - float(neutral["roll"])
    pitch_sig = pitch - float(neutral["pitch"])
    yaw_sig = angle_diff(yaw, float(neutral["yaw"])) / math.pi

    vx = shaped(dx, params["deadzone"], params["curve_p"], params["k_xy"]) + 0.45 * shaped(roll_sig, params["tilt_deadzone"], params["tilt_p"], params["k_xy"])
    vy = shaped(-dy, params["deadzone"], params["curve_p"], params["k_xy"]) + 0.45 * shaped(pitch_sig, params["tilt_deadzone"], params["tilt_p"], params["k_xy"])
    vz = shaped(width_ratio - 1.0, params["deadzone"] * 0.65, params["curve_p"], params["k_z"])
    yaw_rate = shaped(yaw_sig, params["tilt_deadzone"], params["tilt_p"], params["k_yaw"])

    return np.array([vx, vy, vz, yaw_rate], dtype=float)


class DroneSim:
    def __init__(self, start: Tuple[float, float, float] = START_POS) -> None:
        self.reset(start)

    def reset(self, start: Tuple[float, float, float] = START_POS) -> None:
        self.pos = np.array(start, dtype=float)
        self.vel = np.zeros(3, dtype=float)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.crash = False
        self.landed = bool(start[2] <= 0.02)

    def step(self, desired: np.ndarray, dt: float, mode: str, params: Dict[str, Any]) -> None:
        if self.crash:
            return

        cmd = np.asarray(desired, dtype=float).copy()

        if mode == "HOVER":
            cmd[:] = 0.0
        elif mode == "WARNING":
            cmd[:3] *= 0.5
            cmd[3] *= 0.5
        elif mode == "LAND":
            cmd[:] = [0.0, 0.0, -float(params["land_rate"]), 0.0]
        elif mode == "GEOFENCE":
            center_err = -self.pos[:2]
            cmd[0] = clamp(center_err[0] * 1.0, -float(params["max_vxy"]), float(params["max_vxy"]))
            cmd[1] = clamp(center_err[1] * 1.0, -float(params["max_vxy"]), float(params["max_vxy"]))
            cmd[2] = clamp((1.5 - self.pos[2]) * 0.8, -float(params["max_vz"]), float(params["max_vz"]))
            cmd[3] = 0.0
        elif mode == "LANDED":
            cmd[:] = 0.0

        cmd[0] = clamp(cmd[0], -float(params["max_vxy"]), float(params["max_vxy"]))
        cmd[1] = clamp(cmd[1], -float(params["max_vxy"]), float(params["max_vxy"]))
        cmd[2] = clamp(cmd[2], -float(params["max_vz"]), float(params["max_vz"]))
        cmd[3] = clamp(cmd[3], -float(params["max_yaw_rate"]), float(params["max_yaw_rate"]))

        max_dv = float(params["accel_limit"]) * dt
        self.vel = self.vel + np.clip(cmd[:3] - self.vel, -max_dv, max_dv)

        max_dw = float(params["yaw_accel_limit"]) * dt
        self.yaw_rate = self.yaw_rate + clamp(cmd[3] - self.yaw_rate, -max_dw, max_dw)

        self.pos += self.vel * dt
        self.yaw = (self.yaw + self.yaw_rate * dt + math.pi) % (2.0 * math.pi) - math.pi

        if self.pos[2] <= 0.0:
            impact = -float(self.vel[2])
            self.pos[2] = 0.0
            if impact > float(params["crash_vz"]):
                self.crash = True
            self.vel[2] = 0.0
            self.landed = True

        r = float(np.linalg.norm(self.pos[:2]))
        if r > float(params["geofence_radius_m"]):
            self.pos[:2] *= float(params["geofence_radius_m"]) / max(r, 1e-9)
            self.vel[:2] *= 0.1

        self.pos[2] = clamp(float(self.pos[2]), float(params["geofence_alt_min_m"]), float(params["geofence_alt_max_m"]))


class FailsafeMachine:
    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.lost_start: Optional[float] = None
        self.mode = "NORMAL"

    def update(self, hand_present: bool, t: float, pos: np.ndarray, params: Dict[str, Any]) -> str:
        if not hand_present:
            if self.lost_start is None:
                self.lost_start = t
            lost = t - self.lost_start
            if lost >= float(params["land_timeout_s"]):
                self.mode = "LAND"
            elif lost >= float(params["hover_timeout_s"]):
                self.mode = "HOVER"
            else:
                self.mode = "WARNING"
        else:
            self.lost_start = None
            if self.mode in ("WARNING", "HOVER"):
                self.mode = "NORMAL"

        r = float(np.linalg.norm(np.asarray(pos, dtype=float)[:2]))
        outside = (
            r > float(params["geofence_radius_m"])
            or float(pos[2]) < float(params["geofence_alt_min_m"]) - 0.05
            or float(pos[2]) > float(params["geofence_alt_max_m"])
        )

        if self.mode != "LAND" and outside:
            self.mode = "GEOFENCE"
        elif self.mode == "GEOFENCE" and r <= 0.95 * float(params["geofence_radius_m"]) and float(params["geofence_alt_min_m"]) <= float(pos[2]) <= float(params["geofence_alt_max_m"]):
            self.mode = "NORMAL"

        if float(pos[2]) <= 0.02 and self.mode == "LAND":
            self.mode = "LANDED"

        return self.mode


@dataclass
class TaskState:
    active: bool = False
    next_gate: int = 0
    start_time: float = 0.0
    elapsed: float = 0.0
    path_error: float = 0.0
    crashes: int = 0
    finished: bool = False
    success: bool = False
    message: str = "Idle"

    def reset(self, t: float) -> None:
        self.active = True
        self.next_gate = 0
        self.start_time = float(t)
        self.elapsed = 0.0
        self.path_error = 0.0
        self.crashes = 0
        self.finished = False
        self.success = False
        self.message = "Running 3-gate task"

    def update(self, pos: np.ndarray, t: float, dt: float, crashed: bool, params: Dict[str, Any]) -> None:
        if not self.active:
            return

        pos = np.asarray(pos, dtype=float)
        self.elapsed = max(0.0, float(t) - float(self.start_time))
        cross = min_distance_point_polyline(pos[:2], ROUTE)
        self.path_error += float(cross) * float(dt)

        if crashed:
            self.crashes += 1
            self.finished = True
            self.success = False
            self.message = "Task failed: crash"
            return

        if self.next_gate < len(GATES):
            target = np.array(GATES[self.next_gate], dtype=float)
            gate_r = float(params.get("gate_radius_m", 0.9))
            alt_band = float(params.get("gate_alt_band_m", 1.0))
            if float(np.linalg.norm(pos - target)) <= gate_r and abs(float(pos[2] - target[2])) <= alt_band:
                self.next_gate += 1
                if self.next_gate <= len(GATES):
                    self.message = f"Gate {self.next_gate} cleared"
                else:
                    self.message = "Returning to pad"
        elif self.next_gate == len(GATES):
            pad = np.array(PAD, dtype=float)
            if float(np.linalg.norm(pos[:2] - pad[:2])) <= 1.0 and float(pos[2]) <= 0.08:
                self.finished = True
                self.success = True
                self.message = "Task complete: landed"

        if self.elapsed > 120.0 and not self.finished:
            self.finished = True
            self.success = False
            self.message = "Task failed: timeout"


def autopilot_command(pos: np.ndarray, task: TaskState, params: Dict[str, Any]) -> np.ndarray:
    if not task.active:
        return np.zeros(4, dtype=float)

    pos = np.asarray(pos, dtype=float)
    if task.next_gate < len(GATES):
        target = np.array(GATES[task.next_gate], dtype=float)
    else:
        target = np.array(PAD, dtype=float)

    err = target - pos
    cmd = np.zeros(4, dtype=float)
    cmd[0] = clamp(err[0] * float(params["task_kp_xy"]), -float(params["max_vxy"]), float(params["max_vxy"]))
    cmd[1] = clamp(err[1] * float(params["task_kp_xy"]), -float(params["max_vxy"]), float(params["max_vxy"]))

    if task.next_gate < len(GATES):
        cmd[2] = clamp(err[2] * float(params["task_kp_z"]), -float(params["max_vz"]), float(params["max_vz"]))
    else:
        cmd[2] = -float(params["land_rate"])

    return cmd


def run_self_tests(params: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        rows.append({"test": name, "result": "PASS" if bool(condition) else "FAIL", "detail": detail})

    check("deadzone outputs zero", abs(shaped(0.05, 0.12, 1.6, 1.8)) < 1e-9, "small input inside deadzone")
    check("nonlinear curve positive", shaped(0.30, 0.12, 1.6, 1.8) > 0.0, "input outside deadzone")
    check("clamp works", clamp(5.0, -2.0, 2.0) == 2.0, "upper clamp")

    fs = FailsafeMachine(params)
    fs.update(False, 0.0, np.array([0, 0, 1.5]), params)
    mode_hover = fs.update(False, 0.60, np.array([0, 0, 1.5]), params)
    mode_land = fs.update(False, 3.10, np.array([0, 0, 1.5]), params)
    check("hover failsafe after 0.5 s", mode_hover == "HOVER", f"mode={mode_hover}")
    check("land failsafe after 3 s", mode_land == "LAND", f"mode={mode_land}")

    fs.reset()
    mode_geo = fs.update(True, 0.0, np.array([20.0, 0.0, 1.5]), params)
    check("geofence triggers", mode_geo == "GEOFENCE", f"mode={mode_geo}")

    sim = DroneSim()
    sim.step(np.array([10.0, 0.0, 0.0, 0.0]), 0.5, "NORMAL", params)
    check("horizontal speed clamped", float(np.linalg.norm(sim.vel[:2])) <= float(params["max_vxy"]) + 1e-6, f"vxy={np.linalg.norm(sim.vel[:2]):.3f}")

    det = GestureDetector(params)
    demo = DemoHand()
    lm_open = demo.static("open")
    all_events: List[GestureEvent] = []
    t = 0.0
    for _ in range(30):
        evs, _ = det.update(lm_open, t)
        all_events.extend(evs)
        t += 0.05
    check("open palm arms after hold", any(e.type == "arm" for e in all_events), "arm event emitted")

    lm_fist = demo.static("fist")
    all_events = []
    for _ in range(20):
        evs, _ = det.update(lm_fist, t)
        all_events.extend(evs)
        t += 0.05
    check("fist emits hover", any(e.type == "hover" for e in all_events), "hover event emitted")

    task = TaskState()
    task.reset(0.0)
    task.update(np.array([3.0, 0.0, 1.8]), 0.1, 0.1, False, params)
    check("gate 1 detection", task.next_gate == 1, f"next_gate={task.next_gate}")

    check("video backend available", (imageio is not None) or (cv2 is not None), "imageio or OpenCV")

    return pd.DataFrame(rows)


def run_comparison(params: Dict[str, Any], flights: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    dt = 0.05

    for method in ["gesture", "keyboard"]:
        for flight in range(1, int(flights) + 1):
            rng = np.random.default_rng(1000 + flight + (0 if method == "gesture" else 500))
            sim = DroneSim()
            task = TaskState()
            fs = FailsafeMachine(params)
            task.reset(0.0)

            latency = float(params["gesture_latency_s"] if method == "gesture" else params["keyboard_latency_s"])
            lag = max(1, int(latency / dt))
            cmd_buf = deque([np.zeros(4, dtype=float) for _ in range(lag)], maxlen=lag)

            t = 0.0
            max_t = 120.0

            while t < max_t and not task.finished:
                ideal = autopilot_command(sim.pos, task, params)

                if method == "gesture":
                    noise = rng.normal(0.0, 0.10, 4)
                    hand_present = not (25.0 < t < 25.6)
                    dz_scale = 0.55
                else:
                    noise = rng.normal(0.0, 0.05, 4)
                    hand_present = True
                    dz_scale = 0.20

                cmd_buf.append(ideal + noise)
                raw = cmd_buf[0]

                cmd = np.array([
                    shaped(raw[0], float(params["deadzone"]) * dz_scale, float(params["curve_p"]), 1.0),
                    shaped(raw[1], float(params["deadzone"]) * dz_scale, float(params["curve_p"]), 1.0),
                    shaped(raw[2], float(params["deadzone"]) * dz_scale * 0.5, float(params["curve_p"]), 1.0),
                    shaped(raw[3], float(params["tilt_deadzone"]) * dz_scale, float(params["tilt_p"]), 1.0),
                ], dtype=float)

                mode = fs.update(hand_present, t, sim.pos, params)
                sim.step(cmd, dt, mode, params)
                task.update(sim.pos, t, dt, sim.crash, params)
                t += dt

            mean_path_error = task.path_error / max(task.elapsed, 1e-6)
            rows.append({
                "method": method,
                "flight": flight,
                "time_s": round(float(task.elapsed), 3),
                "mean_path_error_m": round(float(mean_path_error), 3),
                "crash": bool(sim.crash or task.crashes > 0),
                "success": bool(task.success),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("method").agg(
        mean_time_s=("time_s", "mean"),
        mean_path_error_m=("mean_path_error_m", "mean"),
        crashes=("crash", "sum"),
        success_rate=("success", "mean"),
    ).reset_index()
    summary["mean_time_s"] = summary["mean_time_s"].round(3)
    summary["mean_path_error_m"] = summary["mean_path_error_m"].round(3)
    summary["success_rate"] = summary["success_rate"].round(3)
    return df, summary


def ensure_media_dir(params: Dict[str, Any]) -> str:
    path = str(params.get("media_dir", "gesturefly_media"))
    os.makedirs(path, exist_ok=True)
    return path


class VideoWriterWrapper:
    def __init__(self, base_path: str, fps: int, width: int, height: int) -> None:
        self.base_path = base_path
        self.actual_path: Optional[str] = None
        self.width = int(width)
        self.height = int(height)
        self._writer: Any = None
        self._kind: Optional[str] = None

        mp4_path = base_path + ".mp4"

        if imageio is not None:
            try:
                self._writer = imageio.get_writer(
                    mp4_path,
                    fps=float(fps),
                    codec="libx264",
                    quality=8,
                    pixelformat="yuv420p",
                    macro_block_size=1,
                )
                self._kind = "imageio"
                self.actual_path = mp4_path
                return
            except Exception:
                self._writer = None

        if cv2 is not None:
            candidates = [
                ("mp4v", ".mp4"),
                ("avc1", ".mp4"),
                ("XVID", ".avi"),
                ("MJPG", ".avi"),
            ]
            for codec, ext in candidates:
                p = base_path + ext
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    writer = cv2.VideoWriter(p, fourcc, float(fps), (self.width, self.height))
                    if writer.isOpened():
                        self._writer = writer
                        self._kind = "cv2"
                        self.actual_path = p
                        return
                    writer.release()
                except Exception:
                    pass

        raise RuntimeError("No usable local video backend. Install imageio-ffmpeg and/or opencv-python.")

    def write(self, frame_bgr: np.ndarray) -> None:
        if self._kind == "imageio":
            if cv2 is not None:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                rgb = frame_bgr[:, :, ::-1]
            self._writer.append_data(np.ascontiguousarray(rgb))
        elif self._kind == "cv2":
            self._writer.write(frame_bgr)

    def release(self) -> None:
        try:
            if self._kind == "imageio":
                self._writer.close()
            elif self._kind == "cv2":
                self._writer.release()
        except Exception:
            pass


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.45, color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 1) -> None:
    if cv2 is None:
        return
    cv2.putText(img, str(text), (int(org[0]), int(org[1])), cv2.FONT_HERSHEY_SIMPLEX, float(scale), color, int(thickness), cv2.LINE_AA)


def draw_composite(
    lm: Optional[Any],
    state: Dict[str, Any],
    trail: List[Tuple[float, float, float]],
    phase: str,
    gesture: str,
    confidence: float,
    armed: bool,
    failsafe: str,
    command: List[float],
    task_message: str,
    params: Dict[str, Any],
) -> np.ndarray:
    w = int(params.get("video_width", 640))
    h = int(params.get("video_height", 360))
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    if cv2 is None:
        return frame

    hand_w = w // 2

    cv2.rectangle(frame, (0, 0), (hand_w - 1, h - 1), (18, 22, 35), -1)
    cv2.rectangle(frame, (hand_w, 0), (w - 1, h // 2 - 1), (12, 16, 28), -1)
    cv2.rectangle(frame, (hand_w, h // 2), (w - 1, h - 1), (10, 14, 24), -1)

    if lm is not None:
        arr = np.asarray(lm, dtype=float)
        pts = []
        for x, y in arr:
            px = int(15 + float(x) * (hand_w - 30))
            py = int(35 + float(y) * (h - 55))
            pts.append((px, py))

        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], (240, 201, 76), 2, cv2.LINE_AA)

        for pt in pts:
            cv2.circle(frame, pt, 3, (249, 37, 247), -1, cv2.LINE_AA)
    else:
        put_text(frame, "No hand detected", (15, 50), 0.55, (180, 180, 180), 1)

    put_text(frame, "GestureCam local view", (10, 18), 0.45, (255, 255, 255), 1)
    put_text(frame, f"Phase: {str(phase)[:34]}", (10, h - 55), 0.38, (255, 255, 255), 1)
    put_text(frame, f"Gesture: {gesture} conf={float(confidence):.2f}", (10, h - 38), 0.38, (0, 255, 160), 1)
    put_text(frame, f"Armed: {'yes' if armed else 'no'} failsafe: {failsafe}", (10, h - 21), 0.38, (102, 209, 255), 1)

    px = hand_w
    py = 0
    pw = w - hand_w
    ph = h // 2

    R = max(float(params.get("geofence_radius_m", 12.0)), 1.0)
    margin = 24
    scale = min((pw - 2 * margin) / (2 * R), (ph - 2 * margin) / (2 * R))
    cx = px + pw / 2
    cy = py + ph / 2

    cv2.circle(frame, (int(cx), int(cy)), int(R * scale), (240, 201, 76), 1, cv2.LINE_AA)
    cv2.line(frame, (px + margin, int(cy)), (w - margin - 1, int(cy)), (60, 70, 90), 1, cv2.LINE_AA)
    cv2.line(frame, (int(cx), py + margin), (int(cx), ph - margin - 1), (60, 70, 90), 1, cv2.LINE_AA)
    put_text(frame, "Top-down X/Y (m)", (px + 6, py + 16), 0.42, (255, 255, 255), 1)

    for i, g in enumerate(GATES, start=1):
        gx = int(cx + float(g[0]) * scale)
        gy = int(cy - float(g[1]) * scale)
        gr = max(3, int(float(params.get("gate_radius_m", 0.9)) * scale))
        cv2.circle(frame, (gx, gy), gr, (102, 209, 255), 2, cv2.LINE_AA)
        put_text(frame, f"G{i}", (gx + 4, gy - 4), 0.35, (102, 209, 255), 1)

    prev = None
    for p in trail:
        x = int(cx + float(p[0]) * scale)
        y = int(cy - float(p[1]) * scale)
        if prev is not None:
            cv2.line(frame, prev, (x, y), (0, 255, 160), 2, cv2.LINE_AA)
        prev = (x, y)

    dx = int(cx + float(state.get("x", 0.0)) * scale)
    dy = int(cy - float(state.get("y", 0.0)) * scale)
    color = (111, 71, 239) if not state.get("crash", False) else (128, 128, 128)
    cv2.circle(frame, (dx, dy), 6, color, -1, cv2.LINE_AA)

    yaw = float(state.get("yaw", 0.0))
    hx = int(dx + math.cos(yaw) * 14)
    hy = int(dy - math.sin(yaw) * 14)
    cv2.line(frame, (dx, dy), (hx, hy), (255, 255, 255), 2, cv2.LINE_AA)

    bx = hand_w
    by = h // 2
    bw = w - hand_w
    bh = h - h // 2

    alt_max = max(float(params.get("geofence_alt_max_m", 15.0)), 1.0)
    margin_x = 24
    ground_y = by + bh - 24
    scale_x = (bw - 2 * margin_x) / (2 * R)
    scale_z = (bh - 58) / alt_max
    origin_x = bx + bw / 2

    put_text(frame, "Side X/Z (m)", (bx + 6, by + 16), 0.42, (255, 255, 255), 1)
    cv2.line(frame, (bx + margin_x, ground_y), (w - margin_x - 1, ground_y), (180, 180, 180), 2, cv2.LINE_AA)

    prev = None
    for p in trail:
        x = int(origin_x + float(p[0]) * scale_x)
        y = int(ground_y - float(p[2]) * scale_z)
        if prev is not None:
            cv2.line(frame, prev, (x, y), (176, 138, 17), 2, cv2.LINE_AA)
        prev = (x, y)

    sx = int(origin_x + float(state.get("x", 0.0)) * scale_x)
    sy = int(ground_y - float(state.get("z", 0.0)) * scale_z)
    cv2.circle(frame, (sx, sy), 5, color, -1, cv2.LINE_AA)

    lines = [
        f"cmd vx={float(command[0]):+.2f} vy={float(command[1]):+.2f} vz={float(command[2]):+.2f} yaw={float(command[3]):+.2f}",
        f"vel vx={float(state.get('vx', 0.0)):+.2f} vy={float(state.get('vy', 0.0)):+.2f} vz={float(state.get('vz', 0.0)):+.2f}",
        f"pos x={float(state.get('x', 0.0)):.2f} y={float(state.get('y', 0.0)):.2f} z={float(state.get('z', 0.0)):.2f} yaw={float(state.get('yaw', 0.0)):.2f}",
        f"task: {str(task_message)[:42]}",
    ]

    yy = by + 34
    for line in lines:
        put_text(frame, line, (bx + 8, yy), 0.34, (220, 220, 220), 1)
        yy += 14

    return frame


def draw_composite_from_snapshot(snap: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    state = dict(snap.get("state", {}))
    state.setdefault("x", 0.0)
    state.setdefault("y", 0.0)
    state.setdefault("z", 0.0)
    state.setdefault("yaw", 0.0)
    state.setdefault("vx", 0.0)
    state.setdefault("vy", 0.0)
    state.setdefault("vz", 0.0)
    state.setdefault("yaw_rate", 0.0)
    state.setdefault("crash", False)

    command = snap.get("command", [0.0, 0.0, 0.0, 0.0])
    if len(command) != 4:
        command = [0.0, 0.0, 0.0, 0.0]

    return draw_composite(
        snap.get("landmarks"),
        state,
        snap.get("trail", []),
        snap.get("phase", ""),
        snap.get("gesture", "none"),
        float(snap.get("features", {}).get("confidence", 0.0)),
        bool(snap.get("armed", False)),
        snap.get("failsafe_mode", "NORMAL"),
        command,
        snap.get("task", {}).get("message", ""),
        params,
    )


def render_preview_video(
    params: Dict[str, Any],
    duration_s: float,
    fps: int,
    base_path: str,
    mode: str = "demo",
    progress_callback: Optional[Any] = None,
) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV is required to render video.")

    duration_s = max(1.0, float(duration_s))
    fps = max(5, int(fps))
    total_frames = max(1, int(duration_s * fps))
    dt = 1.0 / float(fps)

    writer = VideoWriterWrapper(
        base_path,
        fps=fps,
        width=int(params.get("video_width", 640)),
        height=int(params.get("video_height", 360)),
    )

    sim = DroneSim()
    demo = DemoHand()
    detector = GestureDetector(params)
    fs = FailsafeMachine(params)
    task = TaskState()
    neutral: Optional[Dict[str, Any]] = None
    trail: deque = deque(maxlen=400)
    override: Optional[Tuple[str, Optional[float]]] = None
    t = 0.0

    if mode == "task":
        task.reset(0.0)

    try:
        for i in range(total_frames):
            if progress_callback is not None:
                progress_callback(float(i) / float(max(1, total_frames)))

            lm, intended, phase = demo.generate(t, float(params.get("demo_cycle_s", 45.0)))
            events, features = detector.update(lm, t)

            if lm is not None and neutral is None:
                neutral = compute_neutral(lm)

            if mode != "task":
                for ev in events:
                    if ev.type == "hover":
                        override = ("HOVER", t + 2.0)
                    elif ev.type == "land_request":
                        override = ("LAND", None)
                    elif ev.type == "toggle_task":
                        if task.active:
                            task.active = False
                        else:
                            task.reset(t)
                            sim.reset(START_POS)
                            trail.clear()
                            fs.reset()
                            override = None
                            detector.armed = True

            if override is not None and override[1] is not None and t > override[1]:
                override = None

            hand_present = bool(features.get("present"))

            if mode == "task":
                desired = autopilot_command(sim.pos, task, params)
                phase = f"Task autopilot: next gate {min(task.next_gate + 1, len(GATES))}/{len(GATES)}"
                hand_present = True
            else:
                desired = landmarks_to_command(lm, neutral, params) if hand_present else np.zeros(4, dtype=float)

            fs_mode = fs.update(hand_present, t, sim.pos, params)
            effective = fs_mode

            if override is not None:
                ov_mode, _ = override
                if ov_mode == "LAND":
                    effective = "LAND"
                elif fs_mode not in ("GEOFENCE", "LAND"):
                    effective = ov_mode

            sim.step(desired, dt, effective, params)

            if task.active:
                task.update(sim.pos, t, dt, sim.crash, params)
                if task.finished:
                    task.active = False
                    if mode == "task":
                        phase = task.message

            trail.append((float(sim.pos[0]), float(sim.pos[1]), float(sim.pos[2])))

            state = {
                "x": float(sim.pos[0]),
                "y": float(sim.pos[1]),
                "z": float(sim.pos[2]),
                "yaw": float(sim.yaw),
                "vx": float(sim.vel[0]),
                "vy": float(sim.vel[1]),
                "vz": float(sim.vel[2]),
                "yaw_rate": float(sim.yaw_rate),
                "crash": bool(sim.crash),
            }

            frame = draw_composite(
                lm,
                state,
                list(trail),
                phase,
                features.get("gesture", "none"),
                float(features.get("confidence", 0.0)),
                bool(detector.armed),
                effective,
                desired.tolist(),
                task.message,
                params,
            )

            writer.write(frame)
            t += dt

    finally:
        writer.release()

    if progress_callback is not None:
        progress_callback(1.0)

    if writer.actual_path is None:
        raise RuntimeError("Video writer did not produce a file path.")

    return writer.actual_path


class FrameBus:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.t = 0.0
        self.seq = 0

    def put(self, frame: np.ndarray, t: float) -> None:
        with self.lock:
            self.frame = frame
            self.t = float(t)
            self.seq += 1

    def get(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self.lock:
            return self.frame, self.t, self.seq

    def clear(self) -> None:
        with self.lock:
            self.frame = None
            self.t = 0.0
            self.seq = 0


class Engine:
    def __init__(self) -> None:
        self.params = dict(CONFIG)
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.running = True
        self.mode = "demo"
        self.start_time = time.perf_counter()

        self.frame_bus = FrameBus()
        self.camera_stop: Optional[threading.Event] = None
        self.camera_thread: Optional[threading.Thread] = None

        self.demo = DemoHand()
        self.detector = GestureDetector(self.params)
        self.sim = DroneSim()
        self.fs = FailsafeMachine(self.params)
        self.task = TaskState()

        self.neutral: Optional[Dict[str, Any]] = None
        self.last_lm: Optional[np.ndarray] = None
        self.control_source = "gesture"
        self.manual_cmd = np.zeros(4, dtype=float)
        self.override: Optional[Tuple[str, Optional[float]]] = None

        self.trail = deque(maxlen=400)
        self.event_log = deque(maxlen=120)
        self.telemetry_buffer = deque(maxlen=2500)
        self.errors = deque(maxlen=30)

        self.recording = False
        self.record_path: Optional[str] = None

        self.metrics: Dict[str, Any] = {
            "fps": 0.0,
            "latency_ms": 0.0,
            "tracking_fps": 0.0,
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
        }

        self.mp_error = "MediaPipe Tasks models are optional. Demo mode works without them."
        self.hand_landmarker = None
        self.gesture_recognizer = None
        self.mp_ready = False

        self.snapshot = self._blank_snapshot()

        self.media_dir = ensure_media_dir(self.params)

        self.preview_lock = threading.Lock()
        self.auto_preview_path: Optional[str] = None

        self.render_lock = threading.Lock()
        self.render_status: Dict[str, Any] = {
            "state": "idle",
            "message": "No render job.",
            "progress": 0.0,
            "path": None,
            "mode": None,
        }

        self.capture_lock = threading.Lock()
        self.capture_stop: Optional[threading.Event] = None
        self.capture_thread: Optional[threading.Thread] = None
        self.capture_writer: Optional[VideoWriterWrapper] = None
        self.capture_path: Optional[str] = None
        self.capture_status: Dict[str, Any] = {
            "active": False,
            "path": None,
            "message": "Live capture idle.",
        }

        self.eval_lock = threading.Lock()
        self.eval_status: Dict[str, Any] = {
            "state": "idle",
            "message": "No evaluation run.",
        }
        self.eval_df: Optional[pd.DataFrame] = None
        self.eval_summary: Optional[pd.DataFrame] = None

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

        self.preview_thread = threading.Thread(target=self._auto_preview_worker, daemon=True)
        self.preview_thread.start()

    def _blank_snapshot(self) -> Dict[str, Any]:
        return {
            "t": 0.0,
            "phase": "Starting synthetic demo",
            "intended": "none",
            "features": {
                "present": False,
                "gesture": "none",
                "confidence": 0.0,
                "pinch_ratio": 0.0,
                "palm_width": 0.0,
                "palm_center": [0.0, 0.0],
                "armed": False,
                "extensions": [False, False, False, False],
            },
            "events": [],
            "command": [0.0, 0.0, 0.0, 0.0],
            "state": {
                "x": START_POS[0], "y": START_POS[1], "z": START_POS[2],
                "yaw": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "yaw_rate": 0.0,
                "crash": False,
            },
            "trail": [],
            "landmarks": None,
            "failsafe_mode": "NORMAL",
            "override": None,
            "armed": False,
            "gesture": "none",
            "feedback": "Starting GestureFly synthetic demo...",
            "metrics": dict(self.metrics),
            "task": asdict(TaskState()),
            "neutral": None,
            "mode": "demo",
            "running": True,
            "control_source": "gesture",
            "mp_error": self.mp_error,
            "recording": False,
            "record_path": None,
            "errors": [],
        }

    def _loop(self) -> None:
        last = time.perf_counter()
        while not self.stop_event.is_set():
            now = time.perf_counter()
            dt = min(0.10, max(1e-4, now - last))
            last = now

            with self.lock:
                running = self.running
                fps_target = float(self.params.get("fps", 30))

            if running:
                try:
                    self._step(now, dt)
                except Exception as exc:
                    with self.lock:
                        self.errors.append(f"{type(exc).__name__}: {exc}")

            sleep_s = max(0.0, (1.0 / max(1.0, fps_target)) - (time.perf_counter() - now))
            time.sleep(sleep_s)

    def _perceive(self, now: float, dt: float, params: Dict[str, Any], mode: str) -> Tuple[Optional[np.ndarray], str, str, float]:
        if mode == "camera":
            frame, frame_t, _ = self.frame_bus.get()
            latency = max(0.0, (now - frame_t) * 1000.0) if frame_t > 0 else 0.0
            if frame is None:
                return None, "none", "Waiting for local camera frames", latency
            lm = self._detect_mp(frame, frame_t)
            if lm is None:
                return None, "none", self.mp_error, latency
            return lm, "unknown", "Local camera tracking", latency

        lm, intended, phase = self.demo.generate(now - self.start_time, float(params.get("demo_cycle_s", 45.0)))
        return lm, intended, phase, 0.0

    def _detect_mp(self, frame: np.ndarray, frame_t: float) -> Optional[np.ndarray]:
        if not self.mp_ready or self.hand_landmarker is None or mp is None or cv2 is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(max(0, (frame_t - self.start_time) * 1000))
            res = self.hand_landmarker.detect_for_video(mp_img, ts)
            if res.hand_landmarks:
                return np.array([[p.x, p.y] for p in res.hand_landmarks[0]], dtype=float)
        except Exception as exc:
            with self.lock:
                self.errors.append(f"MediaPipe: {exc}")
        return None

    def _init_mediapipe(self) -> None:
        try:
            if mp is None:
                raise RuntimeError("mediapipe is not installed")
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            hand_model = str(self.params.get("hand_landmarker_model_path", "") or "")
            gesture_model = str(self.params.get("gesture_recognizer_model_path", "") or "")

            if hand_model and os.path.exists(hand_model):
                base = mp_python.BaseOptions(model_asset_path=hand_model)
                opts = vision.HandLandmarkerOptions(
                    base_options=base,
                    num_hands=1,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self.hand_landmarker = vision.HandLandmarker.create_from_options(opts)

            if gesture_model and os.path.exists(gesture_model):
                base = mp_python.BaseOptions(model_asset_path=gesture_model)
                opts = vision.GestureRecognizerOptions(
                    base_options=base,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self.gesture_recognizer = vision.GestureRecognizer.create_from_options(opts)

            self.mp_ready = self.hand_landmarker is not None or self.gesture_recognizer is not None
            if not self.mp_ready:
                self.mp_error = "No local MediaPipe Tasks .task model paths found. Camera landmarks disabled; demo still works."
            else:
                self.mp_error = "Local MediaPipe Tasks initialized."
        except Exception as exc:
            self.mp_ready = False
            self.mp_error = f"MediaPipe Tasks unavailable: {exc}"

    def _camera_loop(self, stop: threading.Event) -> None:
        if cv2 is None:
            with self.lock:
                self.errors.append("OpenCV is not installed")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            with self.lock:
                self.errors.append("Cannot open local webcam")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.params.get("frame_width", 640)))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.params.get("frame_height", 480)))
        while not stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                self.frame_bus.put(frame, time.perf_counter())
            else:
                time.sleep(0.03)
        cap.release()

    def enable_camera(self) -> None:
        self.disable_camera()
        if cv2 is None:
            with self.lock:
                self.mp_error = "OpenCV is not installed; camera disabled."
            return
        stop = threading.Event()
        self.camera_stop = stop
        self.camera_thread = threading.Thread(target=self._camera_loop, args=(stop,), daemon=True)
        self.camera_thread.start()
        self._init_mediapipe()

    def disable_camera(self) -> None:
        if self.camera_stop is not None:
            self.camera_stop.set()
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=0.5)
        self.camera_stop = None
        self.camera_thread = None
        self.frame_bus.clear()

    def _step(self, now: float, dt: float) -> None:
        with self.lock:
            if not self.running:
                return
            params = dict(self.params)
            mode = self.mode
            manual_cmd = np.array(self.manual_cmd, dtype=float)

        lm, intended, phase, latency = self._perceive(now, dt, params, mode)
        events, features = self.detector.update(lm, now)

        if intended and intended not in ("none", "unknown") and bool(features.get("present")):
            with self.lock:
                self.metrics["total"] += 1
                if features.get("gesture") == intended:
                    self.metrics["correct"] += 1
                self.metrics["accuracy"] = float(self.metrics["correct"]) / max(1, int(self.metrics["total"]))

        if lm is not None:
            with self.lock:
                if self.neutral is None:
                    self.neutral = compute_neutral(lm)

        for ev in events:
            ed = {
                "type": ev.type,
                "hand": ev.hand,
                "t_ms": int(ev.t_ms),
                "confidence": round(float(ev.confidence), 3),
                "params": ev.params,
            }
            with self.lock:
                self.event_log.appendleft(ed)

            if ev.type == "hover":
                self.override = ("HOVER", now + 2.0)
            elif ev.type == "land_request":
                self.override = ("LAND", None)
            elif ev.type == "toggle_task":
                if self.task.active:
                    self.stop_task()
                else:
                    self.start_task(now)

        if self.override is not None and self.override[1] is not None and now > self.override[1]:
            self.override = None

        with self.lock:
            control_source = self.control_source

        if control_source == "task" and self.task.active:
            desired = autopilot_command(self.sim.pos, self.task, params)
        elif control_source == "manual":
            desired = manual_cmd
        elif features.get("present"):
            desired = landmarks_to_command(lm, self.neutral, params)
        else:
            desired = np.zeros(4, dtype=float)

        fs_mode = self.fs.update(bool(features.get("present")), now, self.sim.pos, params)
        effective_mode = fs_mode

        if self.override is not None:
            ov_mode, _ = self.override
            if ov_mode == "LAND":
                effective_mode = "LAND"
            elif fs_mode not in ("GEOFENCE", "LAND"):
                effective_mode = ov_mode

        if mode == "demo" and isinstance(phase, str) and phase.startswith("Recovery") and float(self.sim.pos[2]) <= 0.03:
            with self.lock:
                self.sim.reset(START_POS)
                self.fs.reset()
                self.trail.clear()
                self.override = None
                self.task.active = False
                self.control_source = "gesture"
            effective_mode = "NORMAL"

        self.sim.step(desired, dt, effective_mode, params)

        if self.task.active:
            self.task.update(self.sim.pos, now, dt, self.sim.crash, params)
            if self.task.finished:
                with self.lock:
                    self.task.active = False
                    self.control_source = "gesture"

        if dt > 0:
            with self.lock:
                self.metrics["fps"] = float(0.9 * self.metrics.get("fps", 0.0) + 0.1 * (1.0 / dt))
                self.metrics["tracking_fps"] = self.metrics["fps"]
                self.metrics["latency_ms"] = float(latency)

        row = {
            "t_s": round(float(now - self.start_time), 3),
            "x_m": round(float(self.sim.pos[0]), 4),
            "y_m": round(float(self.sim.pos[1]), 4),
            "z_m": round(float(self.sim.pos[2]), 4),
            "yaw_rad": round(float(self.sim.yaw), 4),
            "vx": round(float(desired[0]), 4),
            "vy": round(float(desired[1]), 4),
            "vz": round(float(desired[2]), 4),
            "yaw_rate": round(float(desired[3]), 4),
            "gesture": features.get("gesture", "none"),
            "failsafe": effective_mode,
            "phase": phase,
        }

        with self.lock:
            self.last_lm = lm
            self.telemetry_buffer.append(row)

            if self.recording and self.record_path:
                write_header = not os.path.exists(self.record_path)
                with open(self.record_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(row)

            feedback = (
                f"{phase} | Gesture: {features.get('gesture', 'none')} "
                f"({float(features.get('confidence', 0.0)):.2f}) | "
                f"Armed: {'yes' if features.get('armed') else 'no'} | "
                f"Failsafe: {effective_mode} | Source: {control_source}"
            )

            self.snapshot = {
                "t": float(now - self.start_time),
                "phase": phase,
                "intended": intended,
                "features": features,
                "events": [ed for ed in list(self.event_log)[:8]],
                "command": [float(x) for x in desired.tolist()],
                "state": {
                    "x": float(self.sim.pos[0]),
                    "y": float(self.sim.pos[1]),
                    "z": float(self.sim.pos[2]),
                    "yaw": float(self.sim.yaw),
                    "vx": float(self.sim.vel[0]),
                    "vy": float(self.sim.vel[1]),
                    "vz": float(self.sim.vel[2]),
                    "yaw_rate": float(self.sim.yaw_rate),
                    "crash": bool(self.sim.crash),
                },
                "trail": [tuple(p) for p in list(self.trail)],
                "landmarks": None if lm is None else np.asarray(lm, dtype=float).tolist(),
                "failsafe_mode": effective_mode,
                "override": None if self.override is None else self.override[0],
                "armed": bool(self.detector.armed),
                "gesture": features.get("gesture", "none"),
                "feedback": feedback,
                "metrics": dict(self.metrics),
                "task": asdict(self.task),
                "neutral": copy.deepcopy(self.neutral),
                "mode": self.mode,
                "running": self.running,
                "control_source": self.control_source,
                "mp_error": self.mp_error,
                "recording": self.recording,
                "record_path": self.record_path,
                "errors": list(self.errors)[-6:],
            }

            self.trail.append((float(self.sim.pos[0]), float(self.sim.pos[1]), float(self.sim.pos[2])))

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.snapshot)

    def get_params_copy(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.params)

    def get_param(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self.params.get(key, CONFIG.get(key, default))

    def set_param(self, key: str, value: Any) -> None:
        with self.lock:
            self.params[key] = value

    def set_mode(self, mode: str) -> None:
        with self.lock:
            if mode == self.mode:
                return
            self.mode = mode
        if mode == "camera":
            self.enable_camera()
        else:
            self.disable_camera()
            with self.lock:
                self.detector.reset()
                self.neutral = None

    def set_running(self, running: bool) -> None:
        with self.lock:
            self.running = bool(running)

    def is_running(self) -> bool:
        with self.lock:
            return bool(self.running)

    def is_recording(self) -> bool:
        with self.lock:
            return bool(self.recording)

    def get_control_source(self) -> str:
        with self.lock:
            return str(self.control_source)

    def set_control_source(self, src: str) -> None:
        with self.lock:
            if src == self.control_source and (src != "task" or self.task.active):
                return
            self.control_source = src
            if src == "task":
                if not self.task.active:
                    self.task.reset(time.perf_counter())
                    self.sim.reset(START_POS)
                    self.trail.clear()
                    self.fs.reset()
                    self.override = None
                    self.detector.armed = True
            else:
                self.task.active = False

    def set_manual_command(self, values: List[float]) -> None:
        with self.lock:
            self.manual_cmd = np.array([float(v) for v in values], dtype=float)

    def calibrate(self) -> None:
        with self.lock:
            if self.last_lm is not None:
                self.neutral = compute_neutral(np.asarray(self.last_lm, dtype=float))

    def reset_drone(self) -> None:
        with self.lock:
            self.sim.reset(START_POS)
            self.trail.clear()
            self.fs.reset()
            self.task.active = False
            self.control_source = "gesture"
            self.override = None
            self.detector.reset()
            self.neutral = None

    def start_task(self, t: Optional[float] = None) -> None:
        t = time.perf_counter() if t is None else float(t)
        with self.lock:
            self.task.reset(t)
            self.control_source = "task"
            self.sim.reset(START_POS)
            self.trail.clear()
            self.fs.reset()
            self.override = None
            self.detector.armed = True

    def stop_task(self) -> None:
        with self.lock:
            self.task.active = False
            self.control_source = "gesture"

    def start_recording(self) -> None:
        with self.lock:
            if not self.recording:
                self.recording = True
                self.record_path = f"gesturefly_session_{int(time.time() * 1000)}.csv"

    def stop_recording(self) -> None:
        with self.lock:
            self.recording = False

    def delete_all(self) -> None:
        with self.lock:
            self.recording = False
            self.telemetry_buffer.clear()
            self.event_log.clear()
            self.errors.clear()
            self.trail.clear()
            if self.record_path and os.path.exists(self.record_path):
                try:
                    os.remove(self.record_path)
                except Exception:
                    pass
            self.record_path = None

        self.delete_media_files()

        with self.eval_lock:
            self.eval_df = None
            self.eval_summary = None
            self.eval_status = {"state": "idle", "message": "Evaluation data deleted."}

    def _auto_preview_worker(self) -> None:
        try:
            base = os.path.join(self.media_dir, "gesturefly_auto_preview")
            path = render_preview_video(
                self.get_params_copy(),
                duration_s=float(self.get_param("preview_duration_s", 8.0)),
                fps=int(self.get_param("preview_fps", 20)),
                base_path=base,
                mode="demo",
            )
            with self.preview_lock:
                self.auto_preview_path = path
        except Exception as exc:
            with self.preview_lock:
                self.auto_preview_path = None
            with self.lock:
                self.errors.append(f"Auto preview: {exc}")

    def get_auto_preview_path(self) -> Optional[str]:
        with self.preview_lock:
            return self.auto_preview_path

    def start_render_job(self, mode: str, duration_s: float) -> None:
        with self.render_lock:
            if self.render_status.get("state") == "running":
                return
            base = os.path.join(self.media_dir, f"gesturefly_{mode}_{int(time.time() * 1000)}")
            self.render_status = {
                "state": "running",
                "message": f"Rendering {mode} video in background...",
                "progress": 0.0,
                "path": None,
                "mode": mode,
            }
        threading.Thread(target=self._render_worker, args=(mode, float(duration_s), base), daemon=True).start()

    def _render_worker(self, mode: str, duration_s: float, base_path: str) -> None:
        try:
            def cb(p: float) -> None:
                with self.render_lock:
                    self.render_status["progress"] = float(p)
                    self.render_status["message"] = f"Rendering {mode} video... {int(float(p) * 100)}%"

            path = render_preview_video(
                self.get_params_copy(),
                duration_s=float(duration_s),
                fps=int(self.get_param("preview_fps", 20)),
                base_path=base_path,
                mode=mode,
                progress_callback=cb,
            )
            with self.render_lock:
                self.render_status.update({
                    "state": "done",
                    "message": "Render complete.",
                    "progress": 1.0,
                    "path": path,
                })
        except Exception as exc:
            with self.render_lock:
                self.render_status.update({
                    "state": "error",
                    "message": str(exc),
                    "progress": 0.0,
                    "path": None,
                })
            with self.lock:
                self.errors.append(f"Render: {exc}")

    def get_render_status(self) -> Dict[str, Any]:
        with self.render_lock:
            return copy.deepcopy(self.render_status)

    def start_live_capture(self) -> None:
        with self.capture_lock:
            if self.capture_status.get("active"):
                return
            try:
                base = os.path.join(self.media_dir, f"gesturefly_live_{int(time.time() * 1000)}")
                writer = VideoWriterWrapper(
                    base,
                    fps=int(self.get_param("capture_fps", 15)),
                    width=int(self.get_param("video_width", 640)),
                    height=int(self.get_param("video_height", 360)),
                )
            except Exception as exc:
                self.capture_status = {
                    "active": False,
                    "path": None,
                    "message": f"Capture failed: {exc}",
                }
                with self.lock:
                    self.errors.append(f"Capture start: {exc}")
                return

            self.capture_writer = writer
            self.capture_path = writer.actual_path
            self.capture_stop = threading.Event()
            self.capture_status = {
                "active": True,
                "path": self.capture_path,
                "message": "Recording local video...",
            }

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self) -> None:
        fps = max(1, int(self.get_param("capture_fps", 15)))
        interval = 1.0 / float(fps)

        while self.capture_stop is not None and not self.capture_stop.is_set():
            start = time.perf_counter()
            with self.capture_lock:
                writer = self.capture_writer

            if writer is None:
                break

            try:
                snap = self.get_snapshot()
                frame = draw_composite_from_snapshot(snap, self.get_params_copy())
                writer.write(frame)
            except Exception as exc:
                with self.lock:
                    self.errors.append(f"Capture: {exc}")

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, interval - elapsed))

        with self.capture_lock:
            if self.capture_writer is not None:
                self.capture_writer.release()
            self.capture_writer = None
            self.capture_stop = None
            self.capture_thread = None
            self.capture_status = {
                "active": False,
                "path": self.capture_path,
                "message": "Live capture stopped.",
            }

    def stop_live_capture(self) -> None:
        with self.capture_lock:
            stop = self.capture_stop
            thread = self.capture_thread
        if stop is not None:
            stop.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def get_capture_status(self) -> Dict[str, Any]:
        with self.capture_lock:
            return copy.deepcopy(self.capture_status)

    def list_media_files(self) -> List[Dict[str, Any]]:
        files: List[Dict[str, Any]] = []
        patterns = ["*.mp4", "*.avi", "*.mkv", "*.mov"]
        seen = set()

        for pat in patterns:
            for fp in glob.glob(os.path.join(self.media_dir, pat)):
                ap = os.path.abspath(fp)
                if ap in seen or not os.path.isfile(fp):
                    continue
                seen.add(ap)
                try:
                    stat = os.stat(fp)
                    files.append({
                        "name": os.path.basename(fp),
                        "path": fp,
                        "size_bytes": int(stat.st_size),
                        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    })
                except Exception:
                    pass

        files.sort(key=lambda x: x["path"])
        return files

    def delete_media_files(self) -> int:
        with self.capture_lock:
            active = self.capture_path

        removed = 0
        for item in self.list_media_files():
            p = item["path"]
            if active and os.path.abspath(p) == os.path.abspath(active):
                continue
            try:
                os.remove(p)
                removed += 1
            except Exception:
                pass

        with self.preview_lock:
            if self.auto_preview_path and not os.path.exists(self.auto_preview_path):
                self.auto_preview_path = None

        return removed

    def start_evaluation_job(self, flights: int = 10) -> None:
        with self.eval_lock:
            if self.eval_status.get("state") == "running":
                return
            self.eval_status = {
                "state": "running",
                "message": f"Running {int(flights)} flights per method in background...",
            }
        threading.Thread(target=self._eval_worker, args=(int(flights),), daemon=True).start()

    def _eval_worker(self, flights: int) -> None:
        try:
            df, summary = run_comparison(self.get_params_copy(), int(flights))
            with self.eval_lock:
                self.eval_df = df
                self.eval_summary = summary
                self.eval_status = {
                    "state": "done",
                    "message": f"Completed {int(flights)} flights per method.",
                }
        except Exception as exc:
            with self.eval_lock:
                self.eval_status = {
                    "state": "error",
                    "message": str(exc),
                }
            with self.lock:
                self.errors.append(f"Evaluation: {exc}")

    def get_eval_status(self) -> Dict[str, Any]:
        with self.eval_lock:
            return copy.deepcopy(self.eval_status)

    def get_eval_results(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Dict[str, Any]]:
        with self.eval_lock:
            return self.eval_df, self.eval_summary, copy.deepcopy(self.eval_status)


def live_fragment_impl() -> None:
    engine: Optional[Engine] = st.session_state.get("engine")
    if engine is None:
        return

    snap = engine.get_snapshot()
    params = engine.get_params_copy()

    st.subheader("Live simulated flight")
    if snap.get("mode") == "camera":
        st.error("CAMERA IS ON. Local processing only. Press Delete all data to erase stored telemetry.")
    st.markdown(svg_viewer(snap, params), unsafe_allow_html=True)

    m = snap.get("metrics", {})
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("FPS", f"{float(m.get('fps', 0.0)):.1f}")
    c2.metric("Latency", f"{float(m.get('latency_ms', 0.0)):.1f} ms")
    c3.metric("Gesture acc.", f"{float(m.get('accuracy', 0.0)) * 100.0:.0f}%")
    c4.metric("Failsafe", snap.get("failsafe_mode", "NORMAL"))
    c5.metric("Altitude", f"{float(snap['state']['z']):.2f} m")

    fb = snap.get("feedback", "")
    mode = snap.get("failsafe_mode", "NORMAL")
    if mode in ("LAND", "GEOFENCE"):
        st.error(fb)
    elif mode in ("HOVER", "WARNING"):
        st.warning(fb)
    else:
        st.success(fb)

    cmd = snap.get("command", [0, 0, 0, 0])
    st.write(
        f"Command: vx={cmd[0]:+.2f} m/s, vy={cmd[1]:+.2f} m/s, vz={cmd[2]:+.2f} m/s, yaw_rate={cmd[3]:+.2f} rad/s"
    )

    with st.expander("Raw state and recent events"):
        st.dataframe(pd.DataFrame([snap["state"]]), use_container_width=True)
        if snap.get("events"):
            st.dataframe(pd.DataFrame(snap["events"]), use_container_width=True)
        else:
            st.caption("No gesture events yet.")


def video_fragment_impl() -> None:
    engine: Optional[Engine] = st.session_state.get("engine")
    if engine is None:
        return

    st.subheader("Local video studio")
    st.caption("Videos are generated and stored only on your computer.")

    auto = engine.get_auto_preview_path()
    if auto and os.path.exists(auto):
        st.video(auto)
        st.caption(f"Auto-generated local demo video: {os.path.basename(auto)}")
    else:
        st.info("Auto-generated demo video is being prepared in a background thread. The live simulation viewer is already running.")

    c1, c2, c3, c4 = st.columns(4)
    c1.button("Generate 10s demo video", key="gen_demo_video", on_click=engine.start_render_job, args=("demo", 10.0), use_container_width=True)
    c2.button("Generate 20s task video", key="gen_task_video", on_click=engine.start_render_job, args=("task", 20.0), use_container_width=True)

    capture = engine.get_capture_status()
    if capture.get("active"):
        c3.button("Stop live capture", key="stop_live_capture", on_click=engine.stop_live_capture, use_container_width=True)
    else:
        c3.button("Start live capture", key="start_live_capture", on_click=engine.start_live_capture, use_container_width=True)

    c4.button("Delete media files", key="delete_media_files", on_click=engine.delete_media_files, use_container_width=True)

    render = engine.get_render_status()
    st.progress(float(render.get("progress", 0.0)))
    st.caption(str(render.get("message", "")))

    if render.get("path") and os.path.exists(str(render["path"])):
        st.video(str(render["path"]))
        st.caption(f"Latest rendered video: {os.path.basename(str(render['path']))}")

    st.write("Capture status:", capture.get("message", ""))

    files = engine.list_media_files()[:5]
    if files:
        st.dataframe(pd.DataFrame(files)[["name", "size_bytes", "modified"]], use_container_width=True)
        for idx, item in enumerate(files):
            try:
                with open(item["path"], "rb") as f:
                    data = f.read()
                mime = "video/mp4" if str(item["path"]).lower().endswith(".mp4") else "video/octet-stream"
                st.download_button(
                    f"Download {item['name']}",
                    data=data,
                    file_name=item["name"],
                    mime=mime,
                    key=f"download_media_{idx}_{item['name']}",
                )
            except Exception as exc:
                st.error(f"Could not read {item['name']}: {exc}")
    else:
        st.caption("No media files yet.")


def eval_fragment_impl() -> None:
    engine: Optional[Engine] = st.session_state.get("engine")
    if engine is None:
        return

    st.subheader("Gesture vs keyboard comparison")
    st.caption("Evaluation runs in a background thread so the GUI thread is not blocked.")

    c1, c2 = st.columns(2)
    c1.button("Run 10 flights per method", key="run_eval_10", on_click=engine.start_evaluation_job, args=(10,), use_container_width=True)
    c2.button("Run 5 flights per method", key="run_eval_5", on_click=engine.start_evaluation_job, args=(5,), use_container_width=True)

    status = engine.get_eval_status()
    st.caption(status.get("message", ""))

    if status.get("state") == "running":
        st.info("Evaluation running in background. Interact with the app or switch tabs to refresh.")

    df, summary, _ = engine.get_eval_results()
    if summary is not None:
        st.dataframe(summary, use_container_width=True)
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No evaluation results yet.")


def _make_fragment(func):
    frag = getattr(st, "fragment", None)
    if frag is None:
        return func
    try:
        return frag(run_every=0.10)(func)
    except TypeError:
        try:
            return frag()(func)
        except Exception:
            return func
    except Exception:
        return func


live_fragment = _make_fragment(live_fragment_impl)
video_fragment = _make_fragment(video_fragment_impl)
eval_fragment = _make_fragment(eval_fragment_impl)


def svg_viewer(snap: Dict[str, Any], params: Dict[str, Any]) -> str:
    state = snap["state"]
    trail = snap.get("trail", [])
    R = max(float(params.get("geofence_radius_m", 12.0)), 1.0)
    alt_max = max(float(params.get("geofence_alt_max_m", 15.0)), 1.0)

    sc = 130.0 / R
    scz = 110.0 / alt_max

    def tx(x: float) -> float:
        return 170.0 + float(x) * sc

    def ty(y: float) -> float:
        return 170.0 - float(y) * sc

    def sx(x: float) -> float:
        return 170.0 + float(x) * sc

    def sz(z: float) -> float:
        return 145.0 - float(z) * scz

    gate_html = ""
    for i, g in enumerate(GATES, start=1):
        gate_html += (
            f"<circle cx='{tx(g[0]):.1f}' cy='{ty(g[1]):.1f}' r='{float(params.get('gate_radius_m', 0.9)) * sc:.1f}' "
            f"fill='none' stroke='#ffd166' stroke-width='2' stroke-dasharray='4 3'/>"
            f"<text x='{tx(g[0]) + 8:.1f}' y='{ty(g[1]) - 8:.1f}' fill='#ffd166' font-family='monospace' font-size='11'>G{i}</text>"
        )

    trail_points = " ".join(f"{tx(p[0]):.1f},{ty(p[1]):.1f}" for p in trail)
    drone_color = "#ef476f" if not state.get("crash") else "#8d99ae"
    top = f"""
<svg width='340' height='340' style='background:#0b1020;border-radius:12px'>
  <text x='10' y='18' fill='#e0e0e0' font-family='monospace' font-size='12'>Top-down X/Y metres</text>
  <circle cx='170' cy='170' r='{R * sc:.1f}' fill='none' stroke='#4cc9f0' stroke-width='1.5' stroke-dasharray='5 4'/>
  <line x1='20' y1='170' x2='320' y2='170' stroke='#24304a' stroke-width='1'/>
  <line x1='170' y1='20' x2='170' y2='320' stroke='#24304a' stroke-width='1'/>
  {gate_html}
  <polyline points='{trail_points}' fill='none' stroke='#06d6a0' stroke-width='2' opacity='0.85'/>
  <circle cx='{tx(state['x']):.1f}' cy='{ty(state['y']):.1f}' r='7' fill='{drone_color}'/>
  <text x='{tx(state['x']) + 10:.1f}' y='{ty(state['y']) + 4:.1f}' fill='#ffffff' font-family='monospace' font-size='11'>drone</text>
</svg>
"""

    side_trail = " ".join(f"{sx(p[0]):.1f},{sz(p[2]):.1f}" for p in trail)
    side = f"""
<svg width='340' height='180' style='background:#0b1020;border-radius:12px'>
  <text x='10' y='18' fill='#e0e0e0' font-family='monospace' font-size='12'>Side X/Z metres</text>
  <line x1='20' y1='145' x2='320' y2='145' stroke='#8d99ae' stroke-width='2'/>
  <polyline points='{side_trail}' fill='none' stroke='#118ab2' stroke-width='2' opacity='0.85'/>
  <circle cx='{sx(state['x']):.1f}' cy='{sz(state['z']):.1f}' r='6' fill='{drone_color}'/>
  <text x='250' y='30' fill='#ffd166' font-family='monospace' font-size='11'>z={state['z']:.2f} m</text>
</svg>
"""

    hand = "<svg width='230' height='230' style='background:#0b1020;border-radius:12px'><text x='10' y='18' fill='#e0e0e0' font-family='monospace' font-size='12'>Hand landmarks</text></svg>"
    lms = snap.get("landmarks")
    if lms:
        pts = [(20.0 + float(p[0]) * 190.0, 20.0 + float(p[1]) * 190.0) for p in lms]
        lines = "".join(
            f"<line x1='{pts[a][0]:.1f}' y1='{pts[a][1]:.1f}' x2='{pts[b][0]:.1f}' y2='{pts[b][1]:.1f}' stroke='#4cc9f0' stroke-width='2' opacity='0.85'/>"
            for a, b in HAND_CONNECTIONS
        )
        circles = "".join(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='2.7' fill='#f72585'/>" for x, y in pts
        )
        hand = f"""
<svg width='230' height='230' style='background:#0b1020;border-radius:12px'>
  <text x='10' y='18' fill='#e0e0e0' font-family='monospace' font-size='12'>Hand landmarks</text>
  {lines}
  {circles}
</svg>
"""

    return f"<div style='display:flex;flex-wrap:wrap;gap:12px;align-items:flex-start'>{top}{side}{hand}</div>"


def main() -> None:
    st.set_page_config(page_title="GestureFly Simulated Drone Control", layout="wide", initial_sidebar_state="expanded")
    st.title("GestureFly: Simulated Drone Control")
    st.caption("Local-only gesture-to-flight simulator. No real drone connection. No API key. No upload.")

    if "engine" not in st.session_state:
        st.session_state.engine = Engine()

    engine: Engine = st.session_state.engine
    snap = engine.get_snapshot()

    st.sidebar.header("Session control")

    source_labels = {"Demo (synthetic hand)": "demo", "Camera (local only)": "camera"}
    selected_source_label = st.sidebar.radio(
        "Input source",
        list(source_labels.keys()),
        index=list(source_labels.values()).index(engine.mode if engine.mode in source_labels.values() else "demo"),
    )
    selected_source = source_labels[selected_source_label]
    if selected_source != engine.mode:
        engine.set_mode(selected_source)

    running = st.sidebar.checkbox("Simulation running", value=engine.is_running())
    engine.set_running(running)

    cmd_labels = {"Gesture": "gesture", "Task autopilot": "task", "Manual (keyboard-style)": "manual"}
    current_cmd = engine.get_control_source()
    if current_cmd not in cmd_labels.values():
        current_cmd = "gesture"
    selected_cmd_label = st.sidebar.radio(
        "Command source",
        list(cmd_labels.keys()),
        index=list(cmd_labels.values()).index(current_cmd),
    )
    engine.set_control_source(cmd_labels[selected_cmd_label])

    if engine.get_control_source() == "manual":
        st.sidebar.subheader("Manual keyboard-style joystick")
        vx = st.sidebar.slider("vx m/s", -float(engine.get_param("max_vxy", 2.0)), float(engine.get_param("max_vxy", 2.0)), 0.0, 0.05)
        vy = st.sidebar.slider("vy m/s", -float(engine.get_param("max_vxy", 2.0)), float(engine.get_param("max_vxy", 2.0)), 0.0, 0.05)
        vz = st.sidebar.slider("vz m/s", -float(engine.get_param("max_vz", 1.0)), float(engine.get_param("max_vz", 1.0)), 0.0, 0.05)
        yr = st.sidebar.slider("yaw rate rad/s", -float(engine.get_param("max_yaw_rate", 1.2)), float(engine.get_param("max_yaw_rate", 1.2)), 0.0, 0.05)
        engine.set_manual_command([vx, vy, vz, yr])

    b1, b2 = st.sidebar.columns(2)
    b1.button("Reset drone", on_click=engine.reset_drone, use_container_width=True)
    b2.button("Calibrate", on_click=engine.calibrate, use_container_width=True)

    st.sidebar.button("Run 3-gate task", on_click=engine.start_task, use_container_width=True)
    st.sidebar.button("Stop task", on_click=engine.stop_task, use_container_width=True)

    st.sidebar.header("Privacy and data")
    st.sidebar.caption("Recording writes a local CSV only after you press Record.")
    rec_col1, rec_col2 = st.sidebar.columns(2)
    rec_col1.button("Record CSV", on_click=engine.start_recording, use_container_width=True)
    rec_col2.button("Stop rec.", on_click=engine.stop_recording, use_container_width=True)
    st.sidebar.button("Delete all data", on_click=engine.delete_all, use_container_width=True)

    with st.sidebar.expander("Tuning thresholds"):
        engine.set_param("deadzone", st.slider("Hand offset deadzone", 0.0, 0.35, float(engine.get_param("deadzone")), 0.01))
        engine.set_param("curve_p", st.slider("Offset curve exponent p", 1.0, 2.0, float(engine.get_param("curve_p")), 0.05))
        engine.set_param("k_xy", st.slider("Horizontal velocity gain k_xy", 0.2, 4.0, float(engine.get_param("k_xy")), 0.05))
        engine.set_param("k_z", st.slider("Vertical velocity gain k_z", 0.2, 3.0, float(engine.get_param("k_z")), 0.05))
        engine.set_param("k_yaw", st.slider("Yaw rate gain k_yaw", 0.2, 3.0, float(engine.get_param("k_yaw")), 0.05))
        engine.set_param("tilt_deadzone", st.slider("Palm tilt deadzone", 0.0, 0.25, float(engine.get_param("tilt_deadzone")), 0.01))
        engine.set_param("hover_timeout_s", st.slider("Hover timeout s", 0.1, 2.0, float(engine.get_param("hover_timeout_s")), 0.05))
        engine.set_param("land_timeout_s", st.slider("Land timeout s", 1.0, 8.0, float(engine.get_param("land_timeout_s")), 0.1))
        engine.set_param("geofence_radius_m", st.slider("Geofence radius m", 3.0, 30.0, float(engine.get_param("geofence_radius_m")), 0.5))
        engine.set_param("arming_hold_s", st.slider("Arming hold s", 0.3, 3.0, float(engine.get_param("arming_hold_s")), 0.05))
        engine.set_param("refractory_s", st.slider("Refractory s", 0.05, 1.5, float(engine.get_param("refractory_s")), 0.05))

    tab_live, tab_video, tab_map, tab_task, tab_eval, tab_tests, tab_priv = st.tabs(
        ["Live demo", "Video studio", "Gesture mapping", "Task course", "Evaluation", "Tests and limits", "Privacy and logs"]
    )

    with tab_live:
        live_fragment()

    with tab_video:
        video_fragment()

    with tab_map:
        st.subheader("Gesture and mapping feedback")
        st.write("This page shows the normalized hand features, current gesture, arming state, and emitted events.")
        c1, c2 = st.columns(2)
        with c1:
            st.json(snap.get("features", {}))
        with c2:
            st.write("Neutral calibration")
            st.json(snap.get("neutral") or {"status": "Not calibrated yet. Hand presence auto-calibrates; press Calibrate to reset."})
            st.write("Recent events")
            if snap.get("events"):
                st.dataframe(pd.DataFrame(snap["events"]), use_container_width=True)
            else:
                st.caption("No events yet.")

    with tab_task:
        st.subheader("3-gate task")
        st.dataframe(pd.DataFrame(GATES, columns=["x_m", "y_m", "z_m"]), use_container_width=True)
        task = snap.get("task", {})
        st.metric("Task active", "Yes" if task.get("active") else "No")
        st.metric("Next gate", f"{int(task.get('next_gate', 0))}/{len(GATES)}")
        st.metric("Elapsed", f"{float(task.get('elapsed', 0.0)):.2f} s")
        st.metric("Path error", f"{float(task.get('path_error', 0.0)):.2f} m·s")
        st.write(task.get("message", "Idle"))

        df = pd.DataFrame(list(engine.telemetry_buffer))
        if not df.empty:
            st.line_chart(df.set_index("t_s")[["x_m", "y_m", "z_m"]])
        else:
            st.caption("Telemetry buffer is empty.")

    with tab_eval:
        eval_fragment()

    with tab_tests:
        st.subheader("Self-tests")
        if st.button("Run unit checks"):
            st.session_state.self_test_df = run_self_tests(engine.get_params_copy())
        if "self_test_df" in st.session_state:
            st.dataframe(st.session_state.self_test_df, use_container_width=True)
        else:
            st.info("Press the button to run mapping, failsafe, geofence, gesture, and video-backend checks.")

        st.subheader("Known limits")
        st.markdown(
            """
- This is a simulator only. Do not connect it to a real drone.
- Demo mode uses synthetic landmarks, not real hand camera data.
- Camera mode requires local MediaPipe Tasks model files; without them, landmark tracking is disabled.
- Thresholds are starting points and must be tuned on real recordings.
- Browser keyboard capture is unreliable, so manual control is provided as slider/joystick emulation.
- Video playback depends on local codecs. `imageio-ffmpeg` is preferred for browser-friendly MP4 files.
- Failsafes are intentionally not removable from this build.
            """
        )

    with tab_priv:
        st.subheader("Privacy")
        st.markdown(
            """
- No network upload is performed by this app.
- Camera access is local and only starts when you select Camera mode.
- Persistent CSV recording starts only when you press Record CSV.
- Generated videos are stored in the local `gesturefly_media` folder.
- Delete all data clears in-memory telemetry, event logs, errors, recorded CSV, evaluation results, and generated media files when possible.
- This app does not identify people.
            """
        )

        if snap.get("record_path") and os.path.exists(str(snap.get("record_path"))):
            with open(str(snap["record_path"]), "r", encoding="utf-8") as f:
                st.download_button("Download recorded CSV", f.read(), file_name=os.path.basename(str(snap["record_path"])), mime="text/csv")
        else:
            st.caption("No recorded CSV exists yet. Press Record CSV to create one.")

        mem_df = pd.DataFrame(list(engine.telemetry_buffer))
        if not mem_df.empty:
            buf = io.StringIO()
            mem_df.to_csv(buf, index=False)
            st.download_button("Export in-memory telemetry CSV", buf.getvalue(), file_name="gesturefly_in_memory.csv", mime="text/csv")

        st.subheader("Current errors")
        if snap.get("errors"):
            st.error("\n".join(snap["errors"]))
        else:
            st.success("No runtime errors captured.")


if __name__ == "__main__":
    main()