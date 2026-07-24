# !/usr/bin/env python
"""랩탑 headless VLA grab — Option B 핸드오프에서 Jetson 이 호출하는 grab 전용.

GUI/SPACE 없이 자동으로:
  grabready 복귀 → VLA 집기 → 상공 게이트 streak 자동 정지 → 손목 확정 → (확정 시)압착
  → arm 은 수건 문 채 유지하고 종료.

■ 두 가지 실행 방식:
  1) one-shot(기존): Jetson 이 매 집기마다 SSH 로 이 스크립트를 새로 띄운다 →
     매번 SmolVLA(≈865MB)+CLIP 을 새로 로드해 집기 순간 스톨이 생긴다.
       ssh andy2@192.168.55.100 "cd ~/lerobot && \
           ~/miniforge3/envs/lerobot/bin/python examples/lekiwi/grab_vla_headless.py"
  2) serve(권장): 모델·게이트를 '시작 전에' 한 번만 로드하고 상주하며, Jetson 은
     TCP 로 GRASP 신호만 보낸다 → 집기 순간 로딩 스톨이 사라진다.
       ~/miniforge3/envs/lerobot/bin/python examples/lekiwi/grab_vla_headless.py --serve
     Jetson(grab_place_vla.py)의 preload 에서 미리 띄워 워밍업한다.

종료코드/응답으로 결과를 Jetson 에 알린다:
  0 = 잡음 확정(grabbed+confirm),  1 = 미확정/실패,  2 = 오류.

전제: Jetson 이 집기 '전에' lekiwi_host 를 띄워둔다(핸드오프). 이 스크립트는 host 를
      끄지 않는다 — Jetson 이 호출 종료 후 host 를 kill(토크 유지)하고 이어받는다.
      serve 모드에서도 로봇 연결은 GRASP 마다 열고 닫는다(모델만 상주).
"""

import os
import socket
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

# serve 모드 TCP 바인딩(Jetson 이 여기로 GRASP/PING 을 보낸다).
VLA_HOST = os.environ.get("VLA_HOST", "0.0.0.0")
VLA_PORT = int(os.environ.get("VLA_PORT", "5577"))


def load_stack():
    """무거운 것(SmolVLA 정책+프로세서, 상공/손목 CLIP 게이트)을 '한 번만' 로드해 반환한다.

    로봇 연결·카메라·ds_features 는 집기(run_one_grasp)마다 새로 잡으므로 여기 없다 →
    모델만 상주시키면 되고(가장 큰 지연), 로봇/카메라 상태는 매번 깨끗하게 시작한다.
    """
    print(f"[headless] 모델 로드 ckpt={CKPT} host={REMOTE_IP} overhead={OVERHEAD_CAM}", flush=True)
    policy = SmolVLAPolicy.from_pretrained(CKPT); policy.eval()
    device = torch.device(policy.config.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=CKPT,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    overhead_gate = GraspGate(os.path.join(REPO, "grasp_probe_overhead.pt"), "grabbed",
                              GRASP_THRESHOLD, GRASP_HOLD, device=str(device))
    wrist_gate = GraspGate(os.path.join(REPO, "grasp_probe_wrist.pt"), "grabbed",
                           WRIST_THRESHOLD, 1, device="cpu")
    print("[headless] 모델·게이트 로드 완료(상주).", flush=True)
    return {"policy": policy, "device": device, "pre": preprocessor, "post": postprocessor,
            "overhead_gate": overhead_gate, "wrist_gate": wrist_gate}


def run_one_grasp(stack):
    """상주 stack 으로 집기 1회 수행. 로봇 연결·상공캠은 여기서 열고 닫는다. rc 반환(0=잡음)."""
    policy, device = stack["policy"], stack["device"]
    preprocessor, postprocessor = stack["pre"], stack["post"]
    overhead_gate, wrist_gate = stack["overhead_gate"], stack["wrist_gate"]

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=REMOTE_IP, id="lekiwi", connect_timeout_s=20))
    renamed = {RENAME.get(k, k): v for k, v in dict(robot.observation_features).items()}
    ds_features = {**hw_to_dataset_features(renamed, OBS_STR),
                   **hw_to_dataset_features(robot.action_features, ACTION)}
    ready = pose_to_action(READY_POSE)
    ohcap = open_v4l2_camera(OVERHEAD_CAM)

    robot.connect()
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
    print(f"[headless] 집기 종료 rc={rc}", flush=True)
    return rc


def main():
    """one-shot: 모델 로드 → 집기 1회 → 종료코드 반환(기존 호환)."""
    stack = load_stack()
    return run_one_grasp(stack)


def serve():
    """상주 서버: 모델을 미리 로드하고 TCP 로 명령을 받는다.

    프로토콜(개행 종결 텍스트):
      PING  → READY          (워밍업 확인)
      GRASP → RC <0|1|2>     (집기 1회 실행. lekiwi_host 가 떠 있어야 함)
      BYE / SHUTDOWN → 서버 종료
    """
    stack = load_stack()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((VLA_HOST, VLA_PORT))
    srv.listen(1)
    print(f"[headless] serve 준비 완료 — {VLA_HOST}:{VLA_PORT} 에서 GRASP 대기", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            try:
                data = conn.recv(64).decode(errors="ignore").strip().upper()
                print(f"[headless] 명령 수신: {data!r} from {addr[0]}", flush=True)
                if data == "PING":
                    conn.sendall(b"READY\n")
                elif data == "GRASP":
                    try:
                        with torch.inference_mode():
                            rc = run_one_grasp(stack)
                    except Exception as e:
                        print(f"[headless] 집기 오류: {e}", flush=True); rc = 2
                    conn.sendall(f"RC {rc}\n".encode())
                elif data in ("BYE", "SHUTDOWN"):
                    conn.sendall(b"BYE\n"); break
                else:
                    conn.sendall(b"ERR unknown\n")
            finally:
                conn.close()
    finally:
        srv.close()
        print("[headless] serve 종료", flush=True)
    return 0


if __name__ == "__main__":
    try:
        if "--serve" in sys.argv:
            sys.exit(serve())
        with torch.inference_mode():
            sys.exit(main())
    except Exception as e:
        print(f"[headless] 오류: {e}", flush=True)
        sys.exit(2)
