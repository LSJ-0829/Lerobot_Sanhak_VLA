"""[3-VLA] VLA 집기 판(grab_place.py 의 drop-in 교체본).

grab_place.py 와 인터페이스(grab_and_return, load_checker)를 똑같이 맞춘 VLA 버전.
집기 동작만 rule-based 모션(laundry_grab)에서 **VLA** 로 바꾼다. 나머지(판별·복귀
주행·던지기)는 grab_place.py 와 동일하다.

새 동작 흐름 (VLA 로 훈련한 것과 동일한 bookend):
  집기 루프(최대 max_attempts 회):
    1) 준비자세(ready_pose=laundry_grabready)로 이동  ← VLA 훈련 시작 자세.
    2) **VLA 집기** (run_vla_grasp): 뻗기→잡기→준비자세 복귀.  ← 지금은 빈 스텁.
       계약: 끝나면 arm 은 다시 ready_pose(grabready) 에서 수건을 문 상태,
             그리고 /dev/ttyACM0(직접 버스)는 free (이후 rule-based 단계가 씀).
    3) 판단 자세(judge_pose)로 이동(그립 유지) → 손목캠 한 프레임 → grasp probe
       로 grabbed/failed 판별.  ← grab_place.py 와 동일.
    4) success_class 이고 신뢰도 >= success_conf 면 성공, 아니면 재시도.
  성공 시 복귀 주행 → 던지기.  ← grab_place.py 와 100% 동일.

grab_place.py 와의 유일한 차이:
  - [1] play_motion('laundry_grab') + [1b] compliant_grip(1차 압착)  →  삭제.
  - 그 자리에 [1] set_pose(ready_pose) + [1b] run_vla_grasp() 스텁.
  - 나머지([2]~[5])는 그대로. gripper 순응/과부하 완화는 '복귀 주행 전 firm-hold'
    ([3b]) 에만 남긴다(수건을 문 채 긴 주행 동안 안 놓게).

laundry_task2.py 에 갈아끼우기 (실제 반영은 나중에):
    - import grab_place as stage_grab
    + import grab_place_vla as stage_grab
  → grab_and_return / load_checker 시그니처가 같으므로 그 한 줄이면 끝.
  (judge_pose 는 여전히 laundry_default 로 넘어온다. VLA 는 grabready 에서 끝나므로
   판별을 grabready 에서 하고 싶으면 laundry_task2 의 judge_pose=... 만 바꾸면 된다.)

전제:
  - poses/laundry_grabready.json (준비자세=훈련 시작 자세, 필수)
  - poses/<judge_pose>.json, grasp_probe.pt (판별용, 필수)
  - motions/<throw_motion>.json (선택 — 없으면 던지기 단계 건너뜀)
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    MOTIONS_DIR, POSES_DIR, REPO,
    compliant_grip, disable_gripper_overload_unload, drive_straight,
    ensure_gripper_max_torque, ensure_gripper_pgain, grab_frame, play_motion,
    relax_gripper_overload, release_arm, rotate, set_pose, stop_wheels,
)
# 판별기 로더는 grab_place 와 동일한 것을 그대로 재사용(중복 구현 방지).
# laundry_task2.preload() 가 stage_grab.load_checker 를 호출하므로 재노출한다.
from grab_place import load_checker  # noqa: F401  (drop-in 재노출)


# ── Option B 핸드오프 설정 ──────────────────────────────────────────
# Jetson 에서 랩탑 VLA(headless grab)를 SSH 로 호출한다. 랩탑 IP/유저·실행 커맨드는 env 로 조절.
LAPTOP_SSH = os.environ.get("LAPTOP_SSH", "andy2@192.168.55.100")
HEADLESS_GRAB_CMD = os.environ.get(
    "HEADLESS_GRAB_CMD",
    "cd ~/lerobot && ~/miniforge3/envs/lerobot/bin/python examples/lekiwi/grab_vla_headless.py")
HOST_PY = os.environ.get("HOST_PY", "~/miniforge3/envs/lerobot/bin/python")

# ── VLA 상주 서버(권장) 설정 ────────────────────────────────────────
# 랩탑에서 grab_vla_headless.py --serve 를 미리 띄워 SmolVLA 를 상주시키면, 집기 순간
# 매번 모델을 로드하던 스톨이 사라진다. Jetson 은 TCP 로 GRASP/PING 만 보낸다.
# warmup_vla_server() 를 laundry_task3.preload() 에서 호출해 접근/문열기 동안 모델을
# 미리 로드시킨다. 서버가 없거나 통신이 안 되면 자동으로 one-shot SSH 방식으로 폴백한다.
LAPTOP_HOST = os.environ.get("LAPTOP_HOST", LAPTOP_SSH.split("@")[-1])  # TCP 대상 IP
VLA_PORT = int(os.environ.get("VLA_PORT", "5577"))
SERVE_CMD = os.environ.get(
    "SERVE_CMD",
    "cd ~/lerobot && ~/miniforge3/envs/lerobot/bin/python "
    "examples/lekiwi/grab_vla_headless.py --serve")


def _vla_send(cmd, timeout=5.0):
    """상주 서버에 한 줄 명령을 보내고 응답 첫 줄을 반환. 실패 시 None."""
    try:
        with socket.create_connection((LAPTOP_HOST, VLA_PORT), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall((cmd + "\n").encode())
            return s.recv(64).decode(errors="ignore").strip()
    except Exception:
        return None


def _vla_server_up():
    return _vla_send("PING", timeout=3.0) == "READY"


def warmup_vla_server(timeout=180.0):
    """VLA 상주 서버가 없으면 SSH 로 띄우고 모델 로드 완료(READY)까지 기다린다.

    laundry_task3.preload() 에서 호출 → 로봇이 접근/문 여는 동안 모델이 병렬로 로드된다.
    반환 True = 서버 상주(집기 스톨 없음). 실패해도 예외 없이 False (집기 때 one-shot 폴백).
    """
    if _vla_server_up():
        print("  [VLA] 상주 서버 이미 READY → 재사용")
        return True
    print(f"  [VLA] 상주 서버 기동(SSH {LAPTOP_SSH}) — SmolVLA 로드 대기...")
    try:
        # SERVE_CMD 는 'cd ... && python ...' 형태라 반드시 셸로 실행해야 한다.
        # nohup 에 직접 넘기면 nohup 이 'cd' 를 실행파일로 여겨 실패하므로 bash -c 로 감싼다.
        subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
             LAPTOP_SSH, f"setsid nohup bash -c '{SERVE_CMD}' > /tmp/vla_serve.log 2>&1 < /dev/null &"],
            timeout=20)
    except Exception as e:
        print(f"  [VLA] 서버 기동 SSH 실패: {e} → 집기 때 one-shot 폴백")
        return False
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3.0)
        if _vla_server_up():
            print("  [VLA] ✅ 상주 서버 READY(모델 로드 완료)")
            return True
    print("  [VLA] ⚠️ 서버 워밍업 타임아웃 → 집기 때 one-shot 폴백")
    return False


def _host_running():
    return subprocess.run(["pgrep", "-f", "lekiwi_host --robot"],
                          capture_output=True).returncode == 0


def _start_host():
    """Jetson 에 lekiwi_host 를 백그라운드로 띄운다(calibration 프롬프트에 yes '' 자동응답)."""
    script = (
        "cd ~/lerobot\n"
        "cat > /tmp/run_host.sh <<'INNER'\n#!/bin/bash\ncd ~/lerobot\n"
        f"yes '' | {HOST_PY} -m lerobot.robots.lekiwi.lekiwi_host "
        "--robot.id=my_awesome_kiwi --host.connection_time_s=7200\nINNER\n"
        "chmod +x /tmp/run_host.sh\n"
        "setsid /tmp/run_host.sh > /tmp/lekiwi_host.log 2>&1 < /dev/null &\n")
    subprocess.run(["bash", "-s"], input=script, text=True)


def _kill_host():
    """SIGKILL 로 host 를 즉시 종료 → cleanup 이 안 돌아 서보 토크가 유지된다(수건 안 놓침)."""
    subprocess.run(["pkill", "-9", "-f", "lekiwi_host --robot"])


def run_vla_grasp(port="/dev/ttyACM0", ready_pose="laundry_grabready",
                  camera=None, checker=None, vla_grasp=None):
    """VLA 집기 **핸드오프**(Option B): Jetson 이 host 를 띄우고 랩탑 headless grab 를
    SSH 로 호출해 VLA 로 집게 한 뒤, host 를 kill(토크 유지)하고 버스를 반환한다.

    [계약] 리턴 시: 팔은 수건 문 상태(grabready 근처) + /dev/ttyACM0 free.
      반환 True = 잡음 확정(랩탑 headless 가 상공 게이트+손목 확정+압착까지 끝냄).
    """
    if vla_grasp is not None:  # 콜백 주입(테스트용)
        return bool(vla_grasp(port=port, ready_pose=ready_pose, camera=camera, checker=checker))

    # 1) 준비자세(직접버스). host 뜨기 전이라 버스 free.
    print(f"  [VLA] 준비자세 이동(직접버스): {ready_pose}")
    if not set_pose(ready_pose, hold=False):
        print("  [VLA] 준비자세 실패"); return False

    # 2) host 기동(백그라운드). 이미 떠 있으면 재사용.
    if _host_running():
        print("  [VLA] host 이미 UP → 재사용")
    else:
        print("  [VLA] host 기동...")
        _start_host()
        up = any((time.sleep(1.0) or _host_running()) for _ in range(15))
        if not up:
            print("  [VLA] host 기동 실패"); return False
        time.sleep(3.0)  # ZMQ 바인딩 여유

    # 3) 랩탑 집기 호출. 상주 서버가 있으면 TCP GRASP(모델 상주 → 스톨 없음),
    #    없으면 one-shot SSH 로 폴백(매번 모델 로드). 종료코드 0 = 잡음 확정.
    rc = None
    if _vla_server_up():
        print(f"  [VLA] 상주 서버에 GRASP 전송: {LAPTOP_HOST}:{VLA_PORT}")
        reply = _vla_send("GRASP", timeout=200.0)
        if reply and reply.startswith("RC"):
            try:
                rc = int(reply.split()[1])
            except (IndexError, ValueError):
                rc = None
        if rc is None:
            print(f"  [VLA] 서버 응답 이상({reply!r}) → one-shot 폴백")
    if rc is None:
        print(f"  [VLA] one-shot grab 호출(SSH): {LAPTOP_SSH}")
        try:
            rc = subprocess.run(
                ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=10",
                 LAPTOP_SSH, HEADLESS_GRAB_CMD], timeout=180).returncode
        except Exception as e:
            print(f"  [VLA] 랩탑 grab 호출 실패: {e}"); rc = 2

    # 4) host kill(토크 유지 → 수건 안 놓침)
    print("  [VLA] host kill(토크 유지)")
    _kill_host(); time.sleep(1.5)

    if rc == 0:
        print("  [VLA] ✅ 잡음 확정 — 팔 수건 문 채, 버스 free")
        return True
    print(f"  [VLA] ✗ grab 실패(rc={rc})")
    return False


def grab_and_return(port="/dev/ttyACM0", camera="/dev/video0",
                    grab_motion="laundry_grab", judge_pose="laundry_default",
                    grasp_probe=None, success_class="grabbed", success_conf=0.70,
                    max_attempts=3,
                    backup_sec=5.0, rotate_sec=17.3, rotate_speed=300,
                    forward_sec=5.0, drive_speed=500, clockwise=True,
                    throw_motion="laundry_throw", keep_held=False, checker=None,
                    retreat_delay=1.0, retreat_sec=4.0,
                    drive=True, compliant=True, grip_torque=500, grip_hold_torque=300,
                    grip_squeeze_sec=6.0, grip_close=None, grip_pgain=32,
                    grip_max_torque=None, relax_overload=True, disable_overload=False,
                    overload_torque=80, protective_torque=80, protection_current=500,
                    ready_pose="laundry_grabready", vla_grasp=None):
    """VLA 집기→판별→(성공 시)복귀 주행→던지기. 성공하면 True.

    grab_place.grab_and_return 과 인터페이스가 같은 **drop-in** 이다. 차이는 집기
    동작을 rule-based 모션 대신 VLA(run_vla_grasp)로 한다는 것뿐. 추가 인자:
      ready_pose : VLA 훈련 시작 자세(=집기 시작/종료 자세). 기본 laundry_grabready.
      vla_grasp  : 실제 VLA 집기 콜백(주입용). None 이면 스텁(아무 동작 없이 통과).
    grab_motion 인자는 시그니처 호환을 위해 남겨두지만 이 판에서는 쓰지 않는다.
    """
    grasp_probe = Path(grasp_probe or (REPO / "grasp_probe.pt"))

    # 필수 준비물 점검(grab 모션 대신 ready_pose 를 점검. throw 는 선택).
    ready_path = POSES_DIR / f"{ready_pose}.json"
    judge_path = POSES_DIR / f"{judge_pose}.json"
    missing = [str(p) for p in [ready_path, judge_path] if not p.exists()]
    if missing:
        print("다음 파일이 없어 실행할 수 없습니다:")
        for m in missing:
            print(f"  - {m}")
        return False

    # Option B: 잡음 판별은 랩탑 headless(상공 게이트+손목 확정)가 한다 → Jetson 판별기 불필요.
    #   (checker/grasp_probe 인자는 drop-in 호환용으로 남겨두되 여기선 쓰지 않는다.)

    # 순응제어 준비: VLA 집기 '전'에 6번 EEPROM(과부하 완화/P게인/절대상한)을 세팅한다.
    #   VLA 는 집기 중 gripper 를 제어하지만, 복귀 주행 동안의 firm-hold([3b])는
    #   여전히 rule-based 이므로, 세게 물어도 안 끊기게 EEPROM 을 미리 완화해 둔다.
    #   (EEPROM 이라 host up/down 을 넘어 유지된다. 옷 물기 전이라 토크 꺼도 안전.)
    if compliant:
        print(f"[0] 6번(gripper) 순응제어 준비: (복귀 주행 firm-hold 용)")
        if grip_pgain is not None:
            print(f"[0-P] 6번 P_Coefficient→{grip_pgain} (위치오차당 토크↑, 그립 강화)")
            ensure_gripper_pgain(port, grip_pgain)
        if grip_max_torque is not None:
            print(f"[0a] (opt-in) Max_Torque_Limit→{grip_max_torque} (EEPROM 영구 변경)")
            ensure_gripper_max_torque(port, grip_max_torque)
        if relax_overload:
            print(f"[0b] 6번 과부하 보호 완화: Overload_Torque→{overload_torque}, "
                  f"Protective_Torque→{protective_torque}, Protection_Current→{protection_current}")
            relax_gripper_overload(port, overload_torque, protective_torque, protection_current)
        if disable_overload:
            print("[0c] 6번 과부하 차단(unload) 메커니즘 제거(과열·과전류 보호는 유지)")
            disable_gripper_overload_unload(port)

    # ---- 집기 루프 (VLA 핸드오프) ----
    success = False
    for attempt in range(1, max_attempts + 1):
        print(f"\n===== VLA 집기 시도 {attempt}/{max_attempts} =====")

        # [1] VLA 집기 핸드오프(랩탑): host 기동 → 랩탑 headless grab(준비자세→VLA→
        #     상공 정지→손목 확정→압착) → host kill(토크 유지). 성공 시 팔은 수건 문 채,
        #     /dev/ttyACM0 free 로 돌아온다.
        print("[1] VLA 집기 핸드오프(랩탑 headless grab)")
        if not run_vla_grasp(port=port, ready_pose=ready_pose, camera=camera,
                             checker=checker, vla_grasp=vla_grasp):
            print("  ✗ VLA 집기 미확인 → 재시도")
            continue

        # [2] 이송(판단) 자세로 이동(그립 유지). 랩탑이 이미 판별했으니 Jetson 재판별은 없음.
        print(f"[2] 이송 자세로 이동: '{judge_pose}' (그립 유지"
              + (", 6번 제외)" if compliant else ")"))
        if not set_pose(judge_pose, hold=True, skip="gripper" if compliant else None):
            print("  자세 이동 실패 → 중단"); return False
        success = True
        break

    if not success:
        print(f"\n⚠️ {max_attempts}회 시도했지만 집기를 확인하지 못했습니다. 사람이 확인/개입하세요.")
        return False

    # 안전(테이블 위 등): 주행 없이 집기+판별만 검증. 옷을 문 채 정지한다.
    if not drive:
        if compliant:
            print("\n[4] --no-drive: 순응 firm-hold(6번) 적용하고 옷을 문 채 정지.")
            compliant_grip(port, close_pos=grip_close, torque_limit=grip_torque,
                           hold_torque=grip_hold_torque, squeeze_s=grip_squeeze_sec)
        print("\n[4] --no-drive: 복귀 주행/던지기 생략. 옷을 문 채 판단 자세로 정지.")
        print("✅ VLA 집기 → 판별 성공(주행 생략).")
        return True

    # [3b] 복귀 주행 '전' firm-hold: 판단 자세 이동 때 얇은 옷 그립이 느슨해질 수
    #      있으니 다시 완전히 닫아 긴 주행 동안 상한 힘으로 꽉 물게 한다.
    if compliant:
        print("\n[3b] 복귀 주행 전 순응 firm-hold(6번)")
        compliant_grip(port, close_pos=grip_close, torque_limit=grip_torque,
                       hold_torque=grip_hold_torque, squeeze_s=grip_squeeze_sec)

    # ---- 복귀 주행 (grab_place.py 와 동일) ----
    print("\n[4] 복귀 주행 (팔은 판단 자세로 옷을 문 채 유지)")
    print(f"  후진 {backup_sec}s → 180도 회전 → 직진 {forward_sec}s")
    drive_straight(port, -1, backup_sec, drive_speed)   # 후진
    rotate(port, rotate_sec, rotate_speed, clockwise=clockwise)  # 180도(시간 근사)
    drive_straight(port, +1, forward_sec, drive_speed)  # 직진

    # ---- 던지기 (grab_place.py 와 동일) ----
    throw_path = MOTIONS_DIR / f"{throw_motion}.json"
    if throw_path.exists():
        print(f"\n[5] 던지기 모션 재생: '{throw_motion}'")
        if not play_motion(throw_motion, 1, hold=keep_held):
            print("  던지기 모션 실패 → 중단(옷 유지)"); return False
        if not keep_held:
            release_arm(port)
        if retreat_sec > 0:
            print(f"\n[6] 그리퍼 열림 → {retreat_delay}s 대기 → 후진 {retreat_sec}s")
            time.sleep(retreat_delay)
            drive_straight(port, -1, retreat_sec, drive_speed)
        print("\n✅ VLA 집기 → 복귀 → 던지기 완료.")
    else:
        print(f"\n[5] 던지기 모션 없음({throw_path.name}) → 복귀 지점에서 옷을 문 채 정지/유지.")
        print("    이동이 검증되면 던지기 모션을 녹화한 뒤 --throw-motion 으로 연결하세요.")
        print("\n✅ VLA 집기 → 복귀 주행 완료(던지기 대기).")
    return True


def main():
    ap = argparse.ArgumentParser(description="VLA 집기→판별→복귀 주행→던지기 (grab_place 의 VLA 판).")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--camera", default="/dev/video0", help="grasp 판별용 손목캠(video0=wrist)")
    ap.add_argument("--ready-pose", default="laundry_grabready",
                    help="VLA 시작/종료 자세(훈련 시작 자세). poses/<이름>.json")
    ap.add_argument("--judge-pose", default="laundry_default", help="판단(촬영) 자세")
    ap.add_argument("--grasp-probe", default=str(REPO / "grasp_probe.pt"))
    ap.add_argument("--success-class", default="grabbed")
    ap.add_argument("--success-conf", type=float, default=0.70)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--backup-sec", type=float, default=5.0, help="집은 뒤 후진 시간(초)")
    ap.add_argument("--rotate-sec", type=float, default=17.3,
                    help="180도 회전 시간(초). 실측: 시계방향 speed 300 에서 17.3초 ≈ 180도")
    ap.add_argument("--rotate-speed", type=int, default=300, help="회전 속도")
    ap.add_argument("--forward-sec", type=float, default=5.0, help="회전 후 직진 시간(초)")
    ap.add_argument("--drive-speed", type=int, default=500, help="전/후진 속도")
    ap.add_argument("--retreat-delay", type=float, default=1.0,
                    help="던지기(그리퍼 열림) 후 후진 시작까지 대기(초)")
    ap.add_argument("--retreat-sec", type=float, default=4.0,
                    help="던지기 후 후진 시간(초). 0이면 후진 안 함")
    ap.add_argument("--ccw", action="store_true", help="반시계로 회전(기본 시계방향)")
    ap.add_argument("--throw-motion", default="laundry_throw",
                    help="복귀 후 던지기 동작(motions/<이름>.json). 없으면 건너뜀")
    ap.add_argument("--keep-held", action="store_true", help="끝나도 토크 유지(디버그)")
    ap.add_argument("--no-drive", action="store_true",
                    help="복귀 주행/던지기 생략(테이블 위 등 안전 테스트). 집기+판별만 검증")
    ap.add_argument("--no-compliant", action="store_true",
                    help="6번 순응제어(복귀 주행 firm-hold) 끄기")
    ap.add_argument("--grip-torque", type=int, default=500,
                    help="firm-hold: 1차 압착 토크 상한(≤ Max_Torque_Limit=500)")
    ap.add_argument("--grip-hold-torque", type=int, default=300,
                    help="firm-hold: 유지 토크 상한(현재위치 유지, 작을수록 과부하 안전)")
    ap.add_argument("--grip-squeeze-sec", type=float, default=6.0,
                    help="firm-hold: 1차 최고출력 압착 시간(초)")
    ap.add_argument("--grip-max-torque", type=int, default=None,
                    help="[EEPROM 영구변경] 6번 Max_Torque_Limit 을 이 값으로 올림(기본 안 건드림)")
    ap.add_argument("--grip-close", type=int, default=None,
                    help="순응 닫힘 목표(steps). 기본=calib range_min≈완전히 닫힘")
    ap.add_argument("--grip-pgain", type=int, default=32,
                    help="[EEPROM] 6번 P_Coefficient(위치 P게인). 기본 32")
    ap.add_argument("--no-relax-overload", action="store_true",
                    help="[EEPROM] 6번 과부하 문턱 완화를 끄기(기본 ON)")
    ap.add_argument("--disable-overload", action="store_true",
                    help="[EEPROM 영구변경] 6번 과부하 차단(unload) 비트 제거. 기본 OFF")
    ap.add_argument("--overload-torque", type=int, default=80,
                    help="과부하 트리거 문턱 Overload_Torque(%%, 기본 25→80)")
    ap.add_argument("--protective-torque", type=int, default=80,
                    help="과부하 후 남는 토크 Protective_Torque(%%, 기본 20→80)")
    ap.add_argument("--protection-current", type=int, default=500,
                    help="과전류 보호 문턱 Protection_Current(기본 250→500)")
    ap.add_argument("--yes", action="store_true", help="시작 확인 없이 바로 실행")
    args = ap.parse_args()

    print(f"[계획] VLA 집기({args.ready_pose} bookend) → 판단({args.judge_pose}) → "
          f"grabbed≥{args.success_conf:.0%}면 후진 {args.backup_sec}s·180도·직진 "
          f"{args.forward_sec}s → 던지기.")
    print("  주의: run_vla_grasp 는 아직 스텁입니다(실제 VLA 미연결).")
    if not args.yes:
        if input("시작하려면 Enter, 취소하려면 그 외 입력 후 Enter: ").strip() != "":
            print("취소되었습니다.")
            return 0

    try:
        ok = grab_and_return(
            port=args.port, camera=args.camera, judge_pose=args.judge_pose,
            grasp_probe=args.grasp_probe, success_class=args.success_class,
            success_conf=args.success_conf, max_attempts=args.max_attempts,
            backup_sec=args.backup_sec, rotate_sec=args.rotate_sec,
            rotate_speed=args.rotate_speed, forward_sec=args.forward_sec,
            drive_speed=args.drive_speed, clockwise=not args.ccw,
            throw_motion=args.throw_motion, keep_held=args.keep_held,
            retreat_delay=args.retreat_delay, retreat_sec=args.retreat_sec,
            drive=not args.no_drive, compliant=not args.no_compliant,
            grip_torque=args.grip_torque, grip_hold_torque=args.grip_hold_torque,
            grip_squeeze_sec=args.grip_squeeze_sec, grip_close=args.grip_close,
            grip_pgain=args.grip_pgain, grip_max_torque=args.grip_max_torque,
            relax_overload=not args.no_relax_overload,
            disable_overload=args.disable_overload,
            overload_torque=args.overload_torque, protective_torque=args.protective_torque,
            protection_current=args.protection_current,
            ready_pose=args.ready_pose,
        )
        return 0 if ok else 1
    except KeyboardInterrupt:
        print("\n[중단] 바퀴 정지.")
        try:
            stop_wheels(args.port)
        except Exception:
            pass
        return 130


if __name__ == "__main__":
    sys.exit(main() or 0)
