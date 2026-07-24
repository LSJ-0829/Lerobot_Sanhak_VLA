#!/usr/bin/env python
"""Approach a color destination WITHOUT rotating — pure translation only:
strafe sideways + drive straight. No alignment step, no rotation.

This is the standalone RED-handle approach (the washer has ONE red blob; there is
no green marker and no stereo/yaw alignment). It centers on the red blob by strafing
instead of turning, so the heading never changes. Same calibrated red params as the
old approach_*.py scripts (see red_approach.json), just without the rotate step.

LeKiwi 3-omniwheel base. Detects the largest color blob in the FRONT camera (video1):
  1) CENTER: STRAFE left/right (parallel move) until the blob is horizontally
             centered. No rotation — heading stays fixed.
  2) DRIVE : drive straight along the forward (image-vertical) axis until the blob
             reaches the target vertical position (cy), then stop.

While driving it keeps re-checking horizontal centering (by strafing), so it stays
square-on and parallel to the destination the whole way in.

Params default to the calibrated red-handle values in red_approach.json; override
with flags. --color defaults to red (green/blue/black presets exist but are unused
in the washer task). Use --detect-only to tune without motion.

  # tune detection only (NO motion): saves annotated frame to /tmp/dest_debug.jpg
  uv run python move_to_destination.py --detect-only
  # full run (robot will strafe and drive — keep the area clear!):
  uv run python move_to_destination.py
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

# HSV presets (OpenCV H:0-179). Red wraps around 0/180 so it needs two ranges.
HSV_PRESETS = {
    "red":   [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (179, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue":  [((95, 120, 60), (130, 255, 255))],
    "black": [((0, 0, 0), (180, 255, 50))],
}


def load_red_params(path):
    """Pull calibrated targets/HSV from red_approach.json if present (red only)."""
    try:
        with open(path) as f:
            c = json.load(f)
    except FileNotFoundError:
        return None
    hsv = c.get("HSV", {})
    ranges = None
    if hsv:
        ranges = [
            (tuple(hsv["lower_red1"]), tuple(hsv["upper_red1"])),
            (tuple(hsv["lower_red2"]), tuple(hsv["upper_red2"])),
        ]
    return {
        "target_y": c.get("TARGET_Y", 20),
        "center_x": c.get("CENTER_X", 320),
        "dead_zone_x": c.get("DEAD_ZONE_X", 50),
        "detect_area_min": c.get("DETECT_AREA_MIN", 300),
        "ranges": ranges,
    }


def detect(frame, ranges, min_area):
    """Return (cx, cy, area) of the largest color blob, or (None, None, None)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, None, None
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--cam", default="/dev/video1", help="front camera")
    p.add_argument("--color", choices=list(HSV_PRESETS), default="red")
    p.add_argument("--config", default="/home/comnet02/lerobot/red_approach.json")
    p.add_argument("--target-y", type=int, default=None,
                   help="destination blob cy (vertical arrival position). smaller = farther/higher in frame")
    p.add_argument("--center-x", type=int, default=None, help="frame center x for facing")
    p.add_argument("--dead-zone-x", type=int, default=None, help="cx tolerance before rotating")
    p.add_argument("--y-tol", type=int, default=4, help="cy tolerance for 'arrived'")
    p.add_argument("--wheel-speed", type=int, default=600, help="forward/back speed")
    p.add_argument("--strafe-speed", type=int, default=400, help="sideways (parallel) centering speed")
    p.add_argument("--strafe-sign", type=int, default=1, choices=(-1, 1),
                   help="wheel sign that strafes RIGHT (flip if centering goes the wrong way)")
    p.add_argument("--min-area", type=int, default=None, help="min blob area to accept")
    p.add_argument("--step", type=float, default=0.1, help="pulse duration per loop (s)")
    p.add_argument("--detect-only", action="store_true", help="no motion; save debug frame")
    p.add_argument("--save-frames", default=None,
                   help="접근 중 읽은 front 프레임을 이 폴더에 <타임스탬프>.jpg 로 저장"
                        "(연속녹화가 이 카메라를 놓은 구간의 공백을 메운다)")
    args = p.parse_args()

    save_dir = args.save_frames
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Resolve params: calibrated file (red) -> preset -> flags override.
    cal = load_red_params(args.config) if args.color == "red" else None
    ranges = (cal["ranges"] if cal and cal["ranges"] else HSV_PRESETS[args.color])
    # 중심 20 ± y_tol 4 → 정지 범위 cy 16~24. --target-y 로 덮어쓰지 않으면 20 를 쓴다.
    target_y = args.target_y if args.target_y is not None else 20
    # 가로 목표점 = '카메라 프레임 정중앙'(빨간 손잡이를 화면 한가운데로 맞춘다).
    # --center-x 로 덮어쓸 수 있고, None 이면 카메라 오픈 후 실제 프레임 폭/2 로 정한다.
    center_x = args.center_x
    dead_x = args.dead_zone_x if args.dead_zone_x is not None else (cal["dead_zone_x"] if cal else 50)
    min_area = args.min_area if args.min_area is not None else (cal["detect_area_min"] if cal else 300)

    # Jetson 기본 백엔드(GStreamer)는 해상도 지정을 무시하고 160x120(YUYV 기본)으로 떨어진다.
    # V4L2 백엔드 + MJPG 를 강제해야 640x480 이 나온다(캘리브레이션도 640x480 기준).
    cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 워밍업하며 실제 프레임 폭을 확인 → 가로 목표점을 프레임 정중앙으로 설정.
    frame_w = 640
    for _ in range(10):
        ret, f = cap.read()
        if ret and f is not None:
            frame_w = f.shape[1]
        time.sleep(0.03)
    if center_x is None:
        center_x = frame_w // 2
    print(f"카메라 프레임 폭={frame_w} → 가로 중앙 목표 center_x={center_x}")

    bus = None
    if not args.detect_only:
        bus = FeetechMotorsBus(
            port=args.port,
            motors={
                "wheel_right": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "wheel_back":  Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "wheel_left":  Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
            },
        )
        bus.connect()

    def set_wheels(left, right, back):
        if bus is None:
            return
        bus.sync_write("Goal_Velocity", {
            "wheel_left": int(left), "wheel_right": int(right), "wheel_back": int(back),
        })

    # Base primitives (sign conventions match test_strafe.py). No rotation is used
    # here — heading is fixed beforehand — so lateral error is corrected by strafing.
    def forward(s):   set_wheels(-s, s, 0)
    def backward(s):  set_wheels(s, -s, 0)
    def strafe(direction):  # +1 = right, -1 = left (verified live: this base is
        s = direction * args.strafe_sign * args.strafe_speed   # opposite of test_strafe.py's label)
        set_wheels(-0.5 * s, -0.5 * s, s)

    print(f"목표: {args.color} blob를 정면으로 확보 후 세로(전후)로 접근 "
          f"(center_x={center_x}, target_y={target_y}, y_tol={args.y_tol})")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # 연속녹화 공백 메우기: 접근 중 읽은 프레임을 그대로 저장(타임스탬프 파일명).
            if save_dir:
                cv2.imwrite(os.path.join(save_dir, f"{time.time():017.6f}.jpg"), frame)

            cx, cy, area = detect(frame, ranges, min_area)

            if args.detect_only:
                dbg = frame.copy()
                h, w = frame.shape[:2]
                cv2.line(dbg, (center_x, 0), (center_x, h), (255, 255, 255), 1)
                cv2.line(dbg, (0, target_y), (w, target_y), (0, 255, 255), 1)
                if cx is not None:
                    cv2.circle(dbg, (cx, cy), 8, (0, 0, 255), -1)
                    cv2.putText(dbg, f"cx={cx} cy={cy} area={area:.0f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"cx={cx} cy={cy} area={area:.0f}")
                else:
                    print(f"no {args.color} blob")
                cv2.imwrite("/tmp/dest_debug.jpg", dbg)
                time.sleep(0.2)
                continue

            if cx is None:
                print("목적지 안 보임 → 정지")
                set_wheels(0, 0, 0)
                time.sleep(0.1)
                continue

            # 1) CENTER: strafe sideways (parallel move, no rotation) until centered.
            if abs(cx - center_x) > dead_x:
                if cx < center_x:
                    print(f"좌측 평행이동 (cx={cx} < {center_x})")
                    strafe(-1)   # blob left of center → strafe left
                else:
                    print(f"우측 평행이동 (cx={cx} > {center_x})")
                    strafe(+1)   # blob right of center → strafe right
                time.sleep(args.step)
                set_wheels(0, 0, 0)
                continue

            # 2) DRIVE: move vertically (forward/back) to the destination cy.
            cy_diff = cy - target_y
            if abs(cy_diff) <= args.y_tol:
                print(f"도착! cy={cy} (target={target_y})")
                set_wheels(0, 0, 0)
                break
            if cy_diff > 0:
                print(f"전진 (cy={cy} > {target_y})")
                forward(args.wheel_speed)
            else:
                print(f"후진 (cy={cy} < {target_y})")
                backward(args.wheel_speed)
            time.sleep(args.step)
            set_wheels(0, 0, 0)

    finally:
        set_wheels(0, 0, 0)  # never leave a stale Goal_Velocity spinning the wheels
        if bus is not None:
            bus.disconnect()
        cap.release()
        print("완료!")


if __name__ == "__main__":
    main()
