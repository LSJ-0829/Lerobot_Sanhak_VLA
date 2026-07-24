# !/usr/bin/env python
"""랩탑 headless VLA grab — Option B 핸드오프에서 Jetson 이 SSH 로 호출하는 grab 전용.

GUI/SPACE 없이 자동으로:
  grabready 복귀 → VLA 집기 → 상공 게이트 streak 자동 정지 → 손목 확정 → (확정 시)압착
  → arm 은 수건 문 채 유지하고 종료.

종료코드로 결과를 Jetson 에 알린다:
  0 = 잡음 확정(grabbed+confirm),  1 = 미확정/실패,  2 = 오류.

전제: Jetson 이 이 호출 '전에' lekiwi_host 를 띄워둔다(핸드오프). 이 스크립트는 host 를
      끄지 않는다 — Jetson 이 호출 종료 후 host 를 kill(토크 유지)하고 이어받는다.

Jetson 에서:
  ssh andy2@192.168.55.100 "cd ~/lerobot && conda run -n lerobot \
      python examples/lekiwi/grab_vla_headless.py"
"""

import os
import sys
import time

os.environ.pop("SESSION_MANAGER", None)

import cv2
import numpy as np
import torch

from grasp_clip import GraspGate, open_v4l2_camera
from lekiwi_pose import move_to_pose, pose_to_action
from lerobot.common.control_utils import predict_action
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

REPO = os.path.expanduser("~/lerobot")
REMOTE_IP = os.environ.get("REMOTE_IP") or ("192.168.0.19" if os.environ.get("WIRELESS") else "192.168.55.1")
CKPT = os.environ.get("LEKIWI_CHECKPOINT", os.path.join(REPO, "models/smolvla_lekiwi_spin_cycle"))
OVERHEAD_CAM = os.environ.get("OVERHEAD_CAM", "/dev/video32")
TASK = "grabbing clothes from washer"
FPS = int(os.environ.get("FPS", "30"))
RENAME = {"front": "camera1", "wrist": "camera2"}
BASE_KEYS = ("x.vel", "y.vel", "theta.vel")
READY_POSE = os.environ.get("READY_POSE", "laundry_grabready")
HOME_DURATION = float(os.environ.get("HOME_DURATION", "3.0"))
GRASP_THRESHOLD = float(os.environ.get("GRASP_THRESHOLD", "0.55"))
GRASP_HOLD = int(os.environ.get("GRASP_HOLD", "3"))
GRASP_CHECK_EVERY = int(os.environ.get("GRASP_CHECK_EVERY", "3"))
WRIST_THRESHOLD = float(os.environ.get("WRIST_THRESHOLD", "0.5"))
GRIP_SQUEEZE = float(os.environ.get("GRIP_SQUEEZE", "0"))
GRIP_SQUEEZE_SEC = float(os.environ.get("GRIP_SQUEEZE_SEC", "1.2"))
GRAB_TIMEOUT = float(os.environ.get("GRAB_TIMEOUT", "40"))


def main():
    print(f"[headless] ckpt={CKPT} host={REMOTE_IP} overhead={OVERHEAD_CAM}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(CKPT); policy.eval()
    device = torch.device(policy.config.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=REMOTE_IP, id="lekiwi", connect_timeout_s=20))
    renamed = {RENAME.get(k, k): v for k, v in dict(robot.observation_features).items()}
    ds_features = {**hw_to_dataset_features(renamed, OBS_STR),
                   **hw_to_dataset_features(robot.action_features, ACTION)}
    overhead_gate = GraspGate(os.path.join(REPO, "grasp_probe_overhead.pt"), "grabbed",
                              GRASP_THRESHOLD, GRASP_HOLD, device=str(device))
    wrist_gate = GraspGate(os.path.join(REPO, "grasp_probe_wrist.pt"), "grabbed",
                           WRIST_THRESHOLD, 1, device="cpu")
    ohcap = open_v4l2_camera(OVERHEAD_CAM)
    ready = pose_to_action(READY_POSE)

    robot.connect(); policy.reset()
    print("[headless] 연결 완료. grabready 복귀 후 VLA 시작.", flush=True)
    move_to_pose(ready, robot=robot, duration=HOME_DURATION, fps=FPS)
    policy.reset(); overhead_gate.reset()

    stop_obs = None
    frame_i = 0
    deadline = time.time() + GRAB_TIMEOUT
    while time.time() < deadline:
        t0 = time.perf_counter()
        obs = robot.get_observation()
        ok, oh = ohcap.read(); oh = oh if ok else None
        obs_frame = build_dataset_frame(ds_features, {RENAME.get(k, k): v for k, v in obs.items()}, prefix=OBS_STR)
        at = predict_action(observation=obs_frame, policy=policy, device=device,
                            preprocessor=preprocessor, postprocessor=postprocessor,
                            use_amp=device.type == "cuda", task=TASK, robot_type=robot.name)
        action = make_robot_action(at, ds_features)
        for k in BASE_KEYS:
            action[k] = 0.0
        robot.send_action(action)
        frame_i += 1
        if oh is not None and frame_i % GRASP_CHECK_EVERY == 0:
            grabbed, prob, _ = overhead_gate.update(cv2.cvtColor(oh, cv2.COLOR_BGR2RGB))
            if grabbed:
                print(f"[headless] ▶ 상공 GRABBED (p={prob:.2f}) → 정지", flush=True)
                stop_obs = obs
                break
        time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    rc = 1
    if stop_obs is None:
        print("[headless] ⚠️ 타임아웃 — 상공 게이트 미발화", flush=True)
    else:
        confirmed = True
        wf = stop_obs.get("wrist")
        if isinstance(wf, np.ndarray):
            wtop, _, wsc = wrist_gate.predict(wf)
            confirmed = (wtop == "grabbed" and wsc.get("grabbed", 0.0) >= WRIST_THRESHOLD)
            print(f"[headless] 손목 확정: {wtop}(p={wsc.get('grabbed',0.0):.2f}) → "
                  + ("✅잡음" if confirmed else "✗헛집음"), flush=True)
        if confirmed:
            if GRIP_SQUEEZE_SEC > 0:
                arm = {k: float(v) for k, v in stop_obs.items()
                       if isinstance(v, (int, float)) and k.startswith("arm_") and k.endswith(".pos")}
                print(f"[headless] 압착 {GRIP_SQUEEZE_SEC}s", flush=True)
                for _ in range(int(GRIP_SQUEEZE_SEC * FPS)):
                    g = dict(arm); g["arm_gripper.pos"] = GRIP_SQUEEZE
                    for bk in BASE_KEYS:
                        g[bk] = 0.0
                    robot.send_action(g); time.sleep(1.0 / FPS)
            rc = 0  # 성공: 수건 문 채 유지(host 는 Jetson 이 kill → 토크 유지)

    # host 는 끄지 않는다(Jetson 이 kill). disconnect 는 소켓만 닫음 → 서보는 마지막 상태 유지.
    try:
        robot.disconnect()
    except Exception:
        pass
    ohcap.release()
    print(f"[headless] 종료 rc={rc}", flush=True)
    return rc


if __name__ == "__main__":
    try:
        with torch.inference_mode():
            sys.exit(main())
    except Exception as e:
        print(f"[headless] 오류: {e}", flush=True)
        sys.exit(2)
