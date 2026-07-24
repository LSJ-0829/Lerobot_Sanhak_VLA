"""record_motion.py 로 녹화한 동작을 재생(복사)하는 스크립트.

사용법:
    uv run python play_motion.py [동작이름] [반복횟수]

  - 동작이름을 주지 않으면 'motion'을 사용한다.
  - 반복횟수를 주지 않으면 1회 재생한다.
  읽는 위치: motions/<동작이름>.json

동작 방식:
    1) 팔을 POSITION 모드로 초기화(set_pose.py 와 동일한 시퀀스)한다.
    2) 안전을 위해 먼저 '첫 프레임' 위치로 천천히 이동한 뒤 Enter로 확인받는다.
       (현재 자세와 녹화 시작 자세가 크게 다르면 급하게 튀는 것을 막기 위함)
    3) 확인되면 녹화된 프레임을 간격(interval)마다 Goal_Position 으로 연달아 써서
       동작을 그대로 재현한다. 각 프레임은 절대 목표 위치이므로 POSITION 모드의
       모터가 알아서 그 위치로 이동한다.
    4) 재생 중 'q' 를 누르면 현재 위치에서 즉시 멈춘다.

[주의] 재생 간격(interval)과 이동 속도(MOVE_VELOCITY)가 맞아야 부드럽다.
    간격 안에 다음 목표까지 이동할 수 있을 만큼 속도가 충분해야 한다. 반대로
    속도가 너무 낮으면 목표에 못 미친 채 다음 프레임으로 넘어가 동작이 뭉개진다.
"""

import json
import select
import sys
import termios
import time
import tty
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

PORT = "/dev/ttyACM0"
MOTIONS_DIR = Path("/home/comnet02/lerobot/motions")
CALIB_PATH = Path(
    "/home/comnet02/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_awesome_kiwi.json"
)

# === 재생 속도/가속 (POSITION 모드에서 유효) ===
MOVE_VELOCITY = 800     # 목표까지 가는 최고 속도(steps/s). 간격 안에 도달할 만큼 넉넉히.
MOVE_ACCELERATION = 40  # 가감속(값 * 100 steps/s^2)

# 첫 프레임으로 '안전하게' 처음 이동할 때만 쓰는 느린 속도/가속
FIRST_MOVE_VELOCITY = 300
FIRST_MOVE_ACCELERATION = 20
FIRST_TOLERANCE = 20
FIRST_TIMEOUT = 30.0

# 정체(stall) 감지: 목표 오차가 tolerance 안에 못 들어와도 팔이 '더는 안 움직이면'
# 첫 프레임 도달로 보고 즉시 넘어간다(set_pose.py 와 동일한 로직).
FIRST_STALL_EPS = 4
FIRST_STALL_FRAMES = 3
FIRST_STALL_GRACE_S = 0.5

MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.DEGREES),
}
ARM_MOTOR_NAMES = list(MOTORS.keys())

# === 그리퍼 stall-freeze (--hold 재생 시 과부하 예방) ===
# 그리퍼가 옷에 막혀 '연속 GRIP_STALL_FRAMES 프레임' 동안 위치 변화가
# GRIP_STALL_EPS 이하이고, 아직 목표에서 GRIP_STALL_TOL 넘게 떨어져 있으면
# (= 도달 불가) 현재 위치로 고정해 지속 최대토크를 멈춘다.
GRIPPER_NAME = "gripper"
GRIP_STALL_EPS = 6
GRIP_STALL_TOL = 15
GRIP_STALL_FRAMES = 2


def _hold_here(bus, names):
    """현재 위치를 목표로 다시 써서 그 자리에 즉시 정지/고정.

    모터별로 방어한다: 그리퍼처럼 옷/한계에 눌려 과부하(Overload) 래치가 걸린
    모터는 Present_Position 읽기 자체가 예외를 던진다. 그 경우 그 모터는 건너뛰고
    (이미 과부하로 토크가 빠진 상태) 나머지 모터만 고정한다 — 한 모터 때문에
    전체 재생/파이프라인이 죽지 않게."""
    for name in names:
        try:
            cur = bus.read("Present_Position", name, normalize=False, num_retry=3)
            bus.write("Goal_Position", name, cur, normalize=False)
        except Exception as e:
            print(f"\n  [hold] '{name}' 고정 건너뜀(과부하/통신 오류): "
                  f"{str(e).splitlines()[0]}")


def _load_limits():
    limits = {}
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        for name in ARM_MOTOR_NAMES:
            entry = calib.get(f"arm_{name}")
            if entry is not None:
                limits[name] = (entry["range_min"], entry["range_max"])
    else:
        print(f"경고: 캘리브레이션 파일이 없어 범위 제한을 적용하지 않습니다: {CALIB_PATH}")
    return limits


def _clamp_frame(frame, limits):
    """한 프레임의 각 값을 calibration 범위 안으로 클램프."""
    out = {}
    for name in ARM_MOTOR_NAMES:
        if name not in frame:
            continue
        v = int(frame[name])
        if name in limits:
            lo, hi = limits[name]
            v = max(lo, min(hi, v))
        out[name] = v
    return out


def _move_to_first(bus, goals):
    """첫 프레임 위치로 천천히 이동(도달 판정 + q 정지)."""
    for name in goals:
        bus.write("Acceleration", name, FIRST_MOVE_ACCELERATION, normalize=False)
        bus.write("Goal_Velocity", name, FIRST_MOVE_VELOCITY, normalize=False)
    for name, goal in goals.items():
        bus.write("Goal_Position", name, goal, normalize=False)

    is_tty = sys.stdin.isatty()
    old_attr = None
    if is_tty:
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    prev = None      # 직전 프레임의 각 모터 위치
    stall = 0        # 연속 '안 움직임' 횟수
    moved = False    # 시작 이후 한 번이라도 실제로 움직였는지
    try:
        start = time.monotonic()
        while time.monotonic() - start < FIRST_TIMEOUT:
            if is_tty:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r and sys.stdin.read(1).lower() == "q":
                    print("\n[q] 정지.")
                    _hold_here(bus, goals)
                    return
            done = True
            cur_all = {}
            for name, goal in goals.items():
                cur = bus.read("Present_Position", name, normalize=False, num_retry=3)
                cur_all[name] = cur
                if abs(goal - cur) > FIRST_TOLERANCE:
                    done = False
            if done:
                return

            # 정체 감지: 움직이기 시작했거나 grace 경과 후 연속 정지면 도달로 간주
            if prev is not None:
                max_move = max(abs(cur_all[n] - prev[n]) for n in cur_all)
                if max_move > FIRST_STALL_EPS:
                    moved = True
                    stall = 0
                else:
                    stall += 1
                grace_passed = (time.monotonic() - start) > FIRST_STALL_GRACE_S
                if (moved or grace_passed) and stall >= FIRST_STALL_FRAMES:
                    _hold_here(bus, goals)
                    return
            prev = cur_all

            time.sleep(0.05)
        print("\n첫 프레임 이동 시간 초과(계속 진행).")
    finally:
        if is_tty and old_attr is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_attr)


def _play(bus, frames, interval, freeze_grip=False):
    """녹화 프레임을 간격마다 연달아 써서 동작을 재현. q로 즉시 정지.

    freeze_grip=True(잡기/유지 재생, --hold): 그리퍼가 옷에 막혀 더 못 닫히면
    그 뒤로는 녹화된 '더 닫힌' 목표 대신 그 순간의 현재 위치를 목표로 고정한다.
    도달 불가한 목표를 매 프레임 다시 쓰지 않으므로 모터가 지속적으로 최대 토크를
    내지 않아 과부하(Overload) 래치를 예방한다(물고 있는 힘은 유지). 던지기처럼
    hold 없이 재생하는 모션은 이 로직을 끄고 녹화 그대로 재생(정상적으로 열림)."""
    is_tty = sys.stdin.isatty()
    old_attr = None
    if is_tty:
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        tty.setcbreak(fd)
    grip_frozen = False   # 그리퍼가 막혀 현재 위치로 고정됐는지
    grip_hold = None      # 고정 목표(현재 위치)
    grip_prev = None      # 직전 그리퍼 위치
    grip_stall = 0        # 그리퍼 연속 '안 움직임' 횟수
    try:
        next_t = time.monotonic()
        for i, frame in enumerate(frames):
            if is_tty:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r and sys.stdin.read(1).lower() == "q":
                    print("\n[q] 재생 정지. 현재 위치에서 멈춥니다.")
                    _hold_here(bus, ARM_MOTOR_NAMES)
                    return False

            # 이번 프레임에 실제로 쓸 목표. 그리퍼 freeze 가 걸리면 그리퍼만 대체한다.
            goals = dict(frame)
            if freeze_grip and GRIPPER_NAME in goals:
                if grip_frozen:
                    goals[GRIPPER_NAME] = grip_hold
                else:
                    try:
                        cur = bus.read("Present_Position", GRIPPER_NAME,
                                       normalize=False, num_retry=2)
                        if (grip_prev is not None
                                and abs(cur - grip_prev) <= GRIP_STALL_EPS
                                and abs(goals[GRIPPER_NAME] - cur) > GRIP_STALL_TOL):
                            grip_stall += 1
                        else:
                            grip_stall = 0
                        grip_prev = cur
                        if grip_stall >= GRIP_STALL_FRAMES:
                            grip_frozen = True
                            grip_hold = cur
                            goals[GRIPPER_NAME] = cur
                            print(f"\n  [grip] 그리퍼가 더 안 닫혀 현재 위치({cur})로 고정 "
                                  f"— 도달 불가 목표를 밀어붙이지 않아 과부하 예방.")
                    except Exception:
                        pass  # 그리퍼 읽기 실패는 무시하고 녹화 프레임대로 진행

            # 프레임 쓰기: 통신 실패 시 최대 6회 재시도, 그래도 실패하면 이 프레임은
            # 건너뛰고 다음 프레임으로 진행(전체 재생이 죽지 않게).
            wrote = False
            last_err = None
            for attempt in range(1, 7):
                try:
                    for name, goal in goals.items():
                        bus.write("Goal_Position", name, goal, normalize=False)
                    wrote = True
                    break
                except Exception as e:
                    last_err = e
            if not wrote:
                print(f"\n  ⚠️ 프레임 {i + 1} 쓰기 6회 실패 → 건너뜀 ({last_err})")
            print(f"  재생 프레임 {i + 1}/{len(frames)}"
                  f"{'' if wrote else ' [SKIP]'}", end="\r", flush=True)

            next_t += interval
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()
        print()
        return True
    finally:
        if is_tty and old_attr is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_attr)


def main():
    argv = sys.argv[1:]
    auto = ("--yes" in argv) or ("-y" in argv)
    # --hold: 재생이 끝나도 팔 토크를 끄지 않는다(그 자리에서 잡은 것을 계속 고정).
    #   바구니를 든 채로 베이스가 후진하는 복귀 단계에서 쓴다(washer_task.py).
    hold = "--hold" in argv
    # --no-grip-freeze: --hold 라도 그리퍼 freeze(막히면 미는 걸 멈추는 소프트웨어
    #   과부하 예방)를 끈다. 하드웨어 과부하 보호를 끄고 순응제어로 계속 물릴 때,
    #   이 freeze 가 오히려 무는 힘을 멈춰 방해하므로 집기 모션에서는 꺼서 계속 밀게 한다.
    no_grip_freeze = "--no-grip-freeze" in argv
    argv = [a for a in argv if a not in ("--yes", "-y", "--hold", "--no-grip-freeze")]
    grip_freeze = hold and not no_grip_freeze
    motion_name = argv[0] if len(argv) > 0 else "motion"
    repeats = int(argv[1]) if len(argv) > 1 else 1
    path = MOTIONS_DIR / f"{motion_name}.json"
    if not path.exists():
        print(f"동작 파일을 찾을 수 없습니다: {path}")
        return

    with open(path) as f:
        data = json.load(f)
    interval = float(data.get("interval", 0.5))
    raw_frames = data.get("frames", [])
    if not raw_frames:
        print("프레임이 비어 있습니다.")
        return

    limits = _load_limits()
    frames = [_clamp_frame(fr, limits) for fr in raw_frames]

    print(f"\n동작 '{motion_name}' : 프레임 {len(frames)}개, 간격 {interval}초, "
          f"총 길이 약 {len(frames) * interval:.1f}초, 반복 {repeats}회")

    bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
    bus.connect()
    try:
        # === POSITION 모드 초기화(set_pose.py 와 동일) ===
        bus.disable_torque(ARM_MOTOR_NAMES)
        for name in ARM_MOTOR_NAMES:
            bus.write("Operating_Mode", name, 0, normalize=False)   # 0 = POSITION
        bus.enable_torque(ARM_MOTOR_NAMES)

        # 1) 안전하게 첫 프레임으로 천천히 이동
        print("\n첫 프레임 위치로 천천히 이동합니다... (정지하려면 q)")
        _move_to_first(bus, frames[0])

        # --yes 면 확인 없이 재생(자동 루프용).
        if not auto:
            answer = input("\n재생하려면 Enter, 취소하려면 그 외 입력 후 Enter: ")
            if answer.strip() != "":
                print("취소되었습니다.")
                return

        # 2) 재생 속도/가속 설정 후 스트리밍 재생
        for name in ARM_MOTOR_NAMES:
            bus.write("Acceleration", name, MOVE_ACCELERATION, normalize=False)
            bus.write("Goal_Velocity", name, MOVE_VELOCITY, normalize=False)

        for rep in range(repeats):
            print(f"\n[{rep + 1}/{repeats}] 재생 중... (정지하려면 q)")
            if not _play(bus, frames, interval, freeze_grip=grip_freeze):
                break  # q로 중단됨
        print("완료!")

        if hold:
            # 마지막 프레임 목표가 물체/한계에 막혀 도달 불가하면, 토크를 켠 채로 두면
            # 모터가 계속 최대 토크로 밀어붙여 과부하(Overload) 보호가 걸린다(특히 그리퍼).
            # 현재 위치를 목표로 재설정해 '그 자리에서 버티기'만 하게 하여 지속 과부하를 막는다.
            # 단, --no-grip-freeze 면 그리퍼는 고정하지 않고 마지막 목표(닫힘)를 계속
            # 밀게 둔다(순응제어로 계속 물려야 하므로). 나머지 팔만 현재 위치로 고정.
            hold_names = ARM_MOTOR_NAMES if grip_freeze else \
                [n for n in ARM_MOTOR_NAMES if n != GRIPPER_NAME]
            _hold_here(bus, hold_names)
            print("[--hold] 현재 위치로 고정(과부하 방지)."
                  + ("" if grip_freeze else " (그리퍼 제외: 계속 물림)"))
    finally:
        # --hold 면 토크를 켠 채로 포트만 닫는다(팔이 잡은 것을 놓지 않음).
        bus.disconnect(disable_torque=not hold)
        if hold:
            print("[--hold] 팔 토크 유지: 잡은 상태 그대로 고정됩니다.")


if __name__ == "__main__":
    main()
