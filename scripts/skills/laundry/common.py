"""laundry_task 단계들이 공유하는 하드웨어/실행 헬퍼 모음.

여기 있는 함수들은 laundry/ 의 각 단계 모듈(approach/open_door/grab_place)과
오케스트레이터(laundry_task.py)가 함께 쓴다. 이전에 washer_task/clear_washer/
grab_and_place 에 흩어져 중복돼 있던 코드를 한곳으로 합친 것이다.

설계 메모:
  - 검증된 스크립트(set_pose/play_motion/move_to_destination)는 여전히 subprocess 로
    호출한다. 매 호출마다 포트를 열고 닫아 '점유한 채 죽는' 위험을 줄이는 기존
    안전성을 유지하기 위함이다.
  - 반면 바퀴/팔 직접 제어(strafe/drive/release)와 카메라 프레임 취득은 짧고
    try/finally 로 버스를 반드시 닫으므로 함수로 직접 노출한다.
"""

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode

REPO = Path("/home/comnet02/lerobot")
POSES_DIR = REPO / "poses"
MOTIONS_DIR = REPO / "motions"
CALIB_PATH = Path(
    "/home/comnet02/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_awesome_kiwi.json"
)

# 검증된 단독 스크립트 경로(subprocess 로 호출)
SET_POSE = REPO / "scripts/arm/set_pose.py"
PLAY_MOTION = REPO / "scripts/arm/play_motion.py"
APPROACH_SCRIPT = REPO / "scripts/skills/laundry/move_to_destination.py"

# 바퀴 전용 버스(팔 모터는 건드리지 않는다).
WHEELS = {
    "wheel_right": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
    "wheel_back":  Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "wheel_left":  Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
}

# 팔 모터(1~6). 토크 해제(unhold) 용도로만 쓴다.
ARM = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.DEGREES),
}

# 그리퍼(6번) 전용 버스. 순응제어(compliant_grip)에서 6번만 건드릴 때 쓴다.
GRIPPER = {"gripper": Motor(6, "sts3215", MotorNormMode.DEGREES)}


def gripper_range():
    """calibration 에서 gripper(6번) (range_min, range_max) 반환. 없으면 (None, None).

    이 로봇 기준: 값이 작을수록 '닫힘'(range_min≈완전히 닫힘), 클수록 '열림'.
    """
    if not CALIB_PATH.exists():
        return (None, None)
    with open(CALIB_PATH) as f:
        entry = json.load(f).get("arm_gripper")
    if not entry:
        return (None, None)
    return (entry.get("range_min"), entry.get("range_max"))


# ---------------------------------------------------------------------------
# subprocess 실행 + 검증된 스크립트 래퍼
# ---------------------------------------------------------------------------
def run_script(script, *cli_args):
    """검증된 스크립트를 같은 파이썬으로 subprocess 실행. 성공하면 True."""
    cmd = [sys.executable, str(script), *[str(a) for a in cli_args]]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode == 0


def set_pose(name, quiet=True, hold=False, order=None, skip=None):
    """set_pose.py 로 poses/<name>.json 자세로 이동.

    hold=True 면 도달 후 토크를 유지하고 시작 시 재초기화를 생략한다(잡은 것을
    놓지 않게). 집기 후 판단 자세로 이동할 때처럼 그립을 유지해야 할 때 쓴다.

    order 는 모터 이동 순서(예: "6;2" = 6번 먼저, 그 다음 2번, 나머지는 마지막).
    지정 안 하면 전 모터 동시 이동.

    skip 은 이번 이동에서 '건드리지 않을' 모터(예: "6" 또는 "gripper"). 순응제어로
    그리퍼를 물린 채 팔만 옮길 때 6번을 제외해 무는 힘이 풀리지 않게 한다.
    """
    extra = []
    if quiet:
        extra.append("--quiet")
    if hold:
        extra.append("--hold")
    if order:
        extra += ["--order", order]
    if skip:
        extra += ["--skip", skip]
    return run_script(SET_POSE, name, "--yes", *extra)


def play_motion(name, repeats=1, hold=False, grip_freeze=True):
    """play_motion.py 로 motions/<name>.json 동작을 repeats 회 재생.

    hold=True 면 재생 후 토크를 유지한다(잡은 것을 놓지 않게).

    grip_freeze=False 면 --hold 라도 그리퍼 freeze(막히면 미는 걸 멈추는 소프트웨어
    과부하 예방)를 끈다. 하드웨어 과부하 보호를 끄고 순응제어로 계속 물릴 때, 이
    freeze 가 무는 힘을 멈춰 방해하므로 집기 모션에서는 꺼서 계속 밀게 한다.
    """
    extra = ["--hold"] if hold else []
    if not grip_freeze:
        extra.append("--no-grip-freeze")
    return run_script(PLAY_MOTION, name, repeats, "--yes", *extra)


# ---------------------------------------------------------------------------
# 바퀴/팔 직접 제어 (짧고, try/finally 로 버스를 반드시 닫는다)
# ---------------------------------------------------------------------------
def _drive_wheels(port, left, right, back, seconds):
    """바퀴 3개에 Goal_Velocity 를 주고 seconds 초 뒤 0 으로 정지."""
    bus = FeetechMotorsBus(port=port, motors=WHEELS)
    bus.connect()
    try:
        bus.sync_write("Goal_Velocity", {
            "wheel_left": int(left), "wheel_right": int(right), "wheel_back": int(back),
        })
        time.sleep(seconds)
    finally:
        # 바퀴 정지. disable_torque 는 바퀴에만 적용되어 팔 토크는 유지된다.
        bus.sync_write("Goal_Velocity", {"wheel_left": 0, "wheel_right": 0, "wheel_back": 0})
        bus.disconnect()


def strafe(port, direction, seconds, speed, strafe_sign=1):
    """회전 없이 좌(-1)/우(+1)로 평행이동을 seconds 초 동안.

    부호 규약은 move_to_destination.py 와 동일(실기 검증):
      s = direction * strafe_sign * speed → set_wheels(-0.5*s, -0.5*s, s)
    """
    s = direction * strafe_sign * speed
    label = "오른쪽 strafe" if direction > 0 else "왼쪽 strafe"
    print(f"  {label} {seconds:.1f}s (speed={speed})")
    _drive_wheels(port, -0.5 * s, -0.5 * s, s, seconds)


def drive_straight(port, direction, seconds, speed):
    """회전 없이 전(+1)/후(-1) 직진을 seconds 초 동안.

    부호 규약은 move_to_destination.py 와 동일(실기 검증):
      forward = set_wheels(-s, s, 0), backward 는 s 부호 반전.
    """
    s = speed if direction > 0 else -speed
    label = "전진" if direction > 0 else "후진"
    print(f"  {label} {seconds:.1f}s (speed={speed})")
    _drive_wheels(port, -s, s, 0, seconds)


def rotate(port, seconds, speed, clockwise=True):
    """제자리 회전을 seconds 초 동안. 세 옴니휠을 같은 속도로 돌린다.

    규약은 scripts/drive/rotate_right.py 와 동일:
      시계방향(우회전) = 세 바퀴 모두 -speed. 반시계 = +speed.
    각도는 오도메트리가 없어 seconds 로 근사한다(실기에서 180도가 되게 조정).
    """
    v = speed if clockwise else -speed
    label = "시계방향(우)" if clockwise else "반시계(좌)"
    print(f"  회전 {label} {seconds:.1f}s (speed={speed})")
    _drive_wheels(port, -v, -v, -v, seconds)  # left, right, back 모두 -v


def stop_wheels(port):
    """바퀴 즉시 정지(안전용). 팔 토크는 건드리지 않는다."""
    _drive_wheels(port, 0, 0, 0, 0.0)


def pose_value(pose_name, motor="shoulder_pan"):
    """poses/<pose_name>.json 에서 특정 모터의 raw 값을 읽어 반환(없으면 None)."""
    path = POSES_DIR / f"{pose_name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get(motor)


def backdrive_and_rotate_pan(port, target_pan, seconds=2.0, drive_speed=500, accel=10):
    """후진과 '동시에' shoulder_pan(1번) 을 target_pan 까지 회전시킨다.

    직전 set_pose(hold=True) 로 팔 토크가 켜져 있고 POSITION 모드라고 가정한다.
    한 버스(바퀴 7·8·9 + 팔 1~6)로 다음을 동시에 한다:
      - 1번(shoulder_pan): (|target-현재|/seconds) steps/s 를 Goal_Velocity 로 줘서
        seconds 초쯤에 target_pan 에 도달(POSITION 모드에서 Goal_Velocity 는 목표까지
        가는 '최고 속도'). 나머지 2~6번은 현재 위치로 고정(=laundry_opencatch 유지).
      - 바퀴: seconds 초 동안 후진, 종료 시 반드시 0 으로 멈춘다(잔여 속도 방지).
    종료 후 팔 토크는 유지(회전한 자세를 홀드).
    """
    motors = {**WHEELS, **ARM}
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()
    try:
        # 팔 토크 ON (POSITION 모드는 직전 set_pose 에서 설정됨). 모터별로 켠다.
        for name in ARM:
            try:
                bus.enable_torque(name)
            except Exception as e:
                print(f"  ⚠️ {name} enable_torque 실패 → 건너뜀: {str(e).splitlines()[0]}")

        # 2~6번은 현재 위치로 고정, 1번은 시작값만 기록.
        start_pan = None
        for name in ARM:
            try:
                cur = bus.read("Present_Position", name, normalize=False, num_retry=3)
            except Exception as e:
                print(f"  ⚠️ {name} 위치 읽기 실패({e}) → 고정 생략")
                continue
            if name == "shoulder_pan":
                start_pan = cur
            else:
                bus.write("Goal_Position", name, cur, normalize=False)  # 그 자리 고정

        # 1번: nothing 값까지 seconds 초에 도달하도록 속도 계산 후 목표 지정.
        if start_pan is None:
            print("  ⚠️ shoulder_pan 현재값을 못 읽어 1번 회전을 생략합니다.")
        else:
            delta = int(target_pan) - int(start_pan)
            vel = max(1, round(abs(delta) / seconds))  # steps/s
            print(f"  1번(shoulder_pan) {start_pan}→{target_pan} (Δ{delta:+d}) "
                  f"— {seconds:.1f}s 목표, 속도 {vel} steps/s")
            bus.write("Acceleration", "shoulder_pan", accel, normalize=False)
            bus.write("Goal_Velocity", "shoulder_pan", vel, normalize=False)
            bus.write("Goal_Position", "shoulder_pan", int(target_pan), normalize=False)

        # 바퀴 후진 시작(직진 규약과 동일: 후진 = wheel_left +speed, wheel_right -speed).
        print(f"  후진 {seconds:.1f}s (speed={drive_speed}) + 1번 회전 동시 진행")
        bus.sync_write("Goal_Velocity", {
            "wheel_left": int(drive_speed), "wheel_right": -int(drive_speed), "wheel_back": 0,
        })
        time.sleep(seconds)
    finally:
        # 바퀴는 반드시 0 으로 정지(잔여 속도 방지). 팔 토크는 유지한다.
        try:
            bus.sync_write("Goal_Velocity",
                           {"wheel_left": 0, "wheel_right": 0, "wheel_back": 0})
        except Exception:
            pass
        bus.disconnect(disable_torque=False)


def ensure_gripper_max_torque(port, value=800):
    """6번(gripper)의 Max_Torque_Limit(EEPROM 절대 토크 상한)을 value 로 보장한다.

    기본 500 이라 teleop 포함 모든 그립이 최대 500 힘으로 제한됐다. 이걸 올리면
    6번만 더 세게 물 수 있다(1~5번은 건드리지 않는다).

    EEPROM 레지스터라 쓰기 전 토크를 꺼 Lock(0) 을 풀어야 한다. 따라서 반드시
    '옷을 물기 전'(그립 유지가 필요 없을 때)에 호출해야 한다 — 무는 도중 부르면
    토크가 꺼져 놓친다. EEPROM 마모를 줄이려 현재값과 다를 때만 쓴다.
    런타임 상한(Torque_Limit, SRAM)도 함께 올려 둔다. 반환: 최종 Max_Torque_Limit.
    """
    bus = FeetechMotorsBus(port=port, motors=GRIPPER)
    bus.connect()
    try:
        try:
            cur = bus.read("Max_Torque_Limit", "gripper", normalize=False, num_retry=3)
        except Exception as e:
            print(f"  ⚠️ Max_Torque_Limit 읽기 실패({e}) → 설정 생략")
            return None
        if int(cur) == int(value):
            print(f"  [max torque] gripper Max_Torque_Limit 이미 {cur} (변경 없음)")
            return int(cur)
        bus.disable_torque("gripper")   # Torque_Enable=0 + Lock=0 → EEPROM 쓰기 가능
        bus.write("Max_Torque_Limit", "gripper", int(value), normalize=False)
        bus.write("Torque_Limit", "gripper", int(value), normalize=False)
        new = bus.read("Max_Torque_Limit", "gripper", normalize=False, num_retry=3)
        print(f"  [max torque] gripper Max_Torque_Limit {cur} → {new} "
              f"(6번만 더 세게 물 수 있게 상한 상향)")
        return int(new)
    finally:
        # 토크는 꺼진 채로 둔다(설정 단계). 곧 이어지는 compliant_grip 이 다시 켠다.
        bus.disconnect(disable_torque=False)


def ensure_gripper_pgain(port, value=32):
    """6번(gripper)의 P_Coefficient(위치 P게인, EEPROM)를 value 로 보장한다.

    STS3215 는 POSITION 모드에서 목표까지의 위치오차에 P_Coefficient 를 곱해
    토크를 낸다. 계측 결과(2026-07-09) 6번이 P=16 이라, 수건처럼 두꺼운 걸 물면
    스톨 상태의 위치오차에도 토크를 약하게 내 load 가 34%(Torque_Limit 캡 50%에
    한참 못 미침)에서 평형 → 그립이 약했다. P=32(공장기본)로 올리면 같은 오차에
    토크를 2배로 내 캡(Torque_Limit)까지 꽉 채워 물고, 그래도 Overload_Torque=80
    아래라 안 끊긴다. 즉 P 를 올려야 캡이 실제 병목이 되어 그립이 세진다.

    EEPROM 레지스터라 쓰기 전 토크를 꺼 Lock(0) 을 풀어야 한다(옷 물기 전 호출).
    현재값과 다를 때만 써 EEPROM 마모를 줄인다. 1~5번은 건드리지 않는다.
    반환: 최종 P_Coefficient.
    """
    bus = FeetechMotorsBus(port=port, motors=GRIPPER)
    bus.connect()
    try:
        try:
            cur = bus.read("P_Coefficient", "gripper", normalize=False, num_retry=3)
        except Exception as e:
            print(f"  ⚠️ P_Coefficient 읽기 실패({e}) → 설정 생략")
            return None
        if int(cur) == int(value):
            print(f"  [P게인] gripper P_Coefficient 이미 {cur} (변경 없음)")
            return int(cur)
        bus.disable_torque("gripper")   # Lock=0 → EEPROM 쓰기 가능
        bus.write("P_Coefficient", "gripper", int(value), normalize=False)
        new = bus.read("P_Coefficient", "gripper", normalize=False, num_retry=3)
        print(f"  [P게인] gripper P_Coefficient {cur} → {new} (위치오차당 토크↑, 그립 강화)")
        return int(new)
    finally:
        # 토크는 꺼진 채로 둔다(설정 단계). 곧 이어지는 compliant_grip 이 다시 켠다.
        bus.disconnect(disable_torque=False)


def relax_gripper_overload(port, overload_torque=80, protective_torque=80,
                           protection_current=500):
    """6번(gripper) 과부하 보호를 덜 민감하게 풀어 모터가 더 밀며 물게 한다.

    STS3215 는 부하가 Overload_Torque(%) 를 Protection_Time 넘게 지속하면 출력을
    Protective_Torque(%) 로 낮춘다(기본 25%→20% : 물다가 힘이 빠지는 원인). 이걸
      - Overload_Torque  ↑ : 트리거 문턱을 올려 웬만해선 안 걸리게
      - Protective_Torque ↑ : 걸려도 남는 토크를 크게(그립 유지)
      - Protection_Current ↑ : 과전류 문턱 여유
    로 완화한다. 모두 EEPROM 이라 토크를 꺼 Lock(0) 을 푼 뒤 쓴다(옷 물기 전 호출).
    현재값과 다를 때만 써서 EEPROM 마모를 줄인다. 1~5번은 건드리지 않는다.

    [주의] 보호를 풀면 정지 상태로 오래 세게 물 때 6번이 발열/손상될 수 있다.
    긴 유지 시 Present_Temperature 를 확인하고, 필요하면 값을 되돌린다.
    """
    targets = {
        "Overload_Torque": int(overload_torque),
        "Protective_Torque": int(protective_torque),
        "Protection_Current": int(protection_current),
    }
    bus = FeetechMotorsBus(port=port, motors=GRIPPER)
    bus.connect()
    try:
        # 현재값 확인 → 하나라도 다르면 토크 끄고(Lock 해제) 쓴다.
        cur = {}
        for reg in targets:
            try:
                cur[reg] = bus.read(reg, "gripper", normalize=False, num_retry=3)
            except Exception as e:
                print(f"  ⚠️ {reg} 읽기 실패({e}) → 과부하 완화 생략")
                return
        if all(int(cur[r]) == targets[r] for r in targets):
            print(f"  [과부하 완화] 이미 목표값 {targets} (변경 없음)")
            return
        bus.disable_torque("gripper")  # Lock=0 → EEPROM 쓰기 가능
        for reg, val in targets.items():
            if int(cur[reg]) != val:
                bus.write(reg, "gripper", val, normalize=False)
                print(f"  [과부하 완화] {reg} {cur[reg]} → {val}")
    finally:
        # 토크는 꺼진 채로 둔다(설정 단계). 곧 이어지는 compliant_grip 이 다시 켠다.
        bus.disconnect(disable_torque=False)


OVERLOAD_BIT = 0x20  # Unloading_Condition 의 과부하(overload) 비트(32)

# 6번(gripper) 공장/원래 기본값(2026-07-09 변경 전 실측). restore_gripper_defaults 로 복구.
GRIPPER_DEFAULTS = {
    "Max_Torque_Limit": 500,
    "Torque_Limit": 500,          # SRAM(전원 시 Max_Torque_Limit 로 리셋)이지만 함께 되돌림
    "Overload_Torque": 25,
    "Protective_Torque": 20,
    "Protection_Current": 250,
    "Unloading_Condition": 44,     # 과열(4)+과전류(8)+과부하(32)
}


def restore_gripper_defaults(port, values=None):
    """6번(gripper) 의 힘/과부하 관련 레지스터를 원래 기본값으로 되돌린다.

    앞서 강한 그립을 위해 올렸던 EEPROM 값(Max_Torque_Limit, Overload_Torque,
    Protective_Torque, Protection_Current, Unloading_Condition)을 전부 복구한다.
    EEPROM 이라 토크를 꺼 Lock(0) 을 푼 뒤 쓰고, 현재값과 다를 때만 쓴다. 1~5번은
    건드리지 않는다. 순응제어(compliant_grip) 소프트웨어 로직은 이와 무관하게 유지된다.
    """
    targets = values or GRIPPER_DEFAULTS
    bus = FeetechMotorsBus(port=port, motors=GRIPPER)
    bus.connect()
    try:
        cur = {}
        for reg in targets:
            try:
                cur[reg] = int(bus.read(reg, "gripper", normalize=False, num_retry=3))
            except Exception as e:
                print(f"  ⚠️ {reg} 읽기 실패({e}) → 건너뜀")
        if all(cur.get(r) == targets[r] for r in targets):
            print("  [복구] 6번 레지스터 이미 기본값 (변경 없음)")
            return
        bus.disable_torque("gripper")  # Lock=0 → EEPROM 쓰기 가능
        for reg, val in targets.items():
            if reg in cur and cur[reg] != val:
                bus.write(reg, "gripper", int(val), normalize=False)
                print(f"  [복구] {reg} {cur[reg]} → {val}")
    finally:
        bus.disconnect(disable_torque=False)


def disable_gripper_overload_unload(port):
    """6번(gripper)의 '과부하 시 토크 차단(unload)' 메커니즘 자체를 끈다.

    STS3215 는 Unloading_Condition(비트마스크)에 걸린 조건이 발생하면 토크를
    아예 내려버린다(unload). 기본 44 = 과열(4)+과전류(8)+과부하(32). 여기서
    과부하 비트(32)만 지워(→ 12) 과열·과전류 보호는 남기고 '과부하로 인한 토크
    차단'만 제거한다. 그러면 옷을 세게 물어도 과부하로 힘이 빠지지 않는다.

    EEPROM 이라 토크를 꺼 Lock(0) 을 푼 뒤 쓴다(옷 물기 전 호출). 이미 꺼져 있으면
    건드리지 않는다(마모 방지). 1~5번은 손대지 않는다.

    [경고] 과부하 차단을 없애면 옷/한계에 걸린 채 세게 물면 6번이 계속 큰 토크를
    내 발열·손상 위험이 커진다(과열 보호는 남지만 그 전에 마모될 수 있음). 오래
    꽉 물지 말고, 유지 후 Present_Temperature 를 확인할 것.
    """
    bus = FeetechMotorsBus(port=port, motors=GRIPPER)
    bus.connect()
    try:
        try:
            uc = int(bus.read("Unloading_Condition", "gripper", normalize=False, num_retry=3))
        except Exception as e:
            print(f"  ⚠️ Unloading_Condition 읽기 실패({e}) → 과부하 차단 제거 생략")
            return
        new = uc & ~OVERLOAD_BIT
        if new == uc:
            print(f"  [과부하 차단 제거] 이미 과부하 비트 꺼짐(Unloading_Condition={uc})")
            return
        bus.disable_torque("gripper")  # Lock=0 → EEPROM 쓰기 가능
        bus.write("Unloading_Condition", "gripper", new, normalize=False)
        print(f"  [과부하 차단 제거] Unloading_Condition {uc}(0b{uc:08b}) → "
              f"{new}(0b{new:08b}) : 과부하 비트(32) 제거, 과열·과전류는 유지")
    finally:
        bus.disconnect(disable_torque=False)


def compliant_grip(port, close_pos=None, torque_limit=500, hold_torque=300,
                   speed=200, accel=8, squeeze_s=6.0, command_close=True, margin=0,
                   close_threshold=2150, extra_after_close=0.5):
    """6번(gripper)에만 2단계 순응 물기를 적용한다(세게 물되 과부하로 안 죽게).

    '완전히 닫힘 목표를 계속 밀기'만 하면 옷/한계에 막힌 채 지속적으로 큰 토크를
    내다가 과부하(Overload) 래치가 걸려 모터가 죽는다. 그래서 두 단계로 나눈다:

      1단계(압착): Torque_Limit=torque_limit(기본 500=Max) 로 close_pos(기본=range_min
        ≈완전히 닫힘)까지 밀며, '위치를 보면서' 얼마나 더 닫을지 정한다(값이 작을수록
        더 닫힘. range_min=2005 완전닫힘, 열림 3176):
          · 아직 close_threshold(기본 2150)보다 '덜 닫힘'(pos > threshold)이면 계속 민다
            (두꺼운 수건이 천천히 파고들도록). 단 squeeze_s(기본 6.0초)를 발열 안전
            상한으로 두고 그 안에서만.
          · close_threshold '이상 닫힘'(pos <= threshold)에 도달하면 extra_after_close
            (기본 0.5초)만 더 압착하고 멈춘다(얇은 옷/빈 그립을 너무 오래 밀어 발열·
            과부하 나는 것 방지).
        P=32+Overload_Torque=80 이면 압착 부하가 ~50%<80% 라 상한까지 밀어도 과부하
        래치가 안 걸린다(발열은 온도 확인).
      2단계(유지): 그 순간의 '현재 위치'를 목표로 다시 쓰고 Torque_Limit 을 hold_torque
        로 낮춘다. 도달 불가한 목표를 계속 밀지 않으므로(위치 오차≈0) 지속 토크가
        뚝 떨어져 래치가 안 걸린다. 문 힘은 유지된다.

    6번만 담은 버스를 열어 1~5번은 건드리지 않는다. 끝나도 토크를 켠 채 포트만 닫아
    무는 상태를 유지한다. set_pose/play_motion 은 Torque_Limit 을 안 건드려, 낮춘
    hold_torque 상한이 이후 판단자세 이동·복귀 주행 내내 SRAM 에 유지된다.

    command_close=False 면 Torque_Limit 만 torque_limit 로 설정하고 목표는 주지
    않는다(집기 모션 재생 '전' 힘만 제한해 두는 용도).
    margin: close_pos 를 range_min 에서 이만큼 열어둔 값으로 잡고 싶을 때(steps).
    hold_torque: 2단계 유지 토크 상한(작을수록 안전, 너무 작으면 미끄러짐).
    close_threshold: '충분히 닫힘'으로 볼 위치(이하면 곧 멈춤). squeeze_s: 발열 안전 상한.
    extra_after_close: threshold 도달 후 추가로 더 압착할 시간(초).
    """
    lo, hi = gripper_range()
    if close_pos is None:
        base = lo if lo is not None else 2005
        close_pos = base + int(margin)
    if lo is not None:
        close_pos = max(lo, min(hi, int(close_pos)))

    bus = FeetechMotorsBus(port=port, motors=GRIPPER)
    bus.connect()
    try:
        try:
            bus.enable_torque("gripper")
        except Exception as e:
            print(f"  ⚠️ gripper enable_torque 실패(과부하 추정) → 건너뜀: "
                  f"{str(e).splitlines()[0]}")
        if not command_close:
            bus.write("Torque_Limit", "gripper", int(torque_limit), normalize=False)
            print(f"  [순응] gripper Torque_Limit={torque_limit} 설정(힘 제한만, 목표 미지정)")
            return

        # 1단계(압착): close_pos(range_min≈완전닫힘) 를 목표로 밀며 위치를 폴링한다.
        #   pos > close_threshold(덜 닫힘) → squeeze_s 상한까지 계속 민다.
        #   pos <= close_threshold(충분히 닫힘) → extra_after_close 만 더 밀고 멈춘다.
        bus.write("Torque_Limit", "gripper", int(torque_limit), normalize=False)
        bus.write("Acceleration", "gripper", int(accel), normalize=False)
        bus.write("Goal_Velocity", "gripper", int(speed), normalize=False)
        bus.write("Goal_Position", "gripper", int(close_pos), normalize=False)
        deadline = time.monotonic() + float(squeeze_s)
        reached = False
        while time.monotonic() < deadline:
            time.sleep(0.1)
            try:
                p = int(bus.read("Present_Position", "gripper", normalize=False, num_retry=2))
            except Exception:
                continue  # 압착 중 읽기 일시 실패는 무시하고 계속 민다
            if p <= int(close_threshold):
                reached = True
                break
        if reached:
            # 2150 이상 닫혔으면 0.5초만 더 압착(얇은 옷/빈 그립 과압 방지).
            time.sleep(max(0.0, float(extra_after_close)))
            print(f"  [압착] {close_threshold} 이상 닫힘 → {extra_after_close}s 추가 후 유지")
        else:
            print(f"  [압착] {close_threshold} 미달(덜 닫힘) → {squeeze_s}s 상한까지 압착 후 유지")

        # 2단계(유지): 현재 위치를 목표로 재설정 + 토크 상한을 hold_torque 로 낮춤.
        #   도달 불가 목표(close_pos)를 계속 밀지 않아 지속 토크가 떨어져 래치 예방.
        try:
            pos = int(bus.read("Present_Position", "gripper", normalize=False, num_retry=3))
        except Exception as e:
            # 압착 중 과부하로 읽기 실패하면, 최소한 토크 상한만 낮춰 지속 부하를 줄인다.
            print(f"  ⚠️ 2단계 현재위치 읽기 실패({str(e).splitlines()[0]}) → 토크 상한만 하향")
            try:
                bus.write("Torque_Limit", "gripper", int(hold_torque), normalize=False)
            except Exception:
                pass
            return
        bus.write("Torque_Limit", "gripper", int(hold_torque), normalize=False)
        bus.write("Goal_Position", "gripper", pos, normalize=False)  # 그 자리 유지
        try:
            load = bus.read("Present_Load", "gripper", normalize=False, num_retry=3)
            print(f"  [2단계 순응 물기] 1단계 압착 {close_pos}(Torque {torque_limit}) → "
                  f"현재 {pos} 유지(Torque {hold_torque}), 부하 {load}")
        except Exception:
            print(f"  [2단계 순응 물기] 압착 {close_pos}(T{torque_limit}) → "
                  f"현재 {pos} 유지(T{hold_torque})")
    finally:
        # 토크 유지(무는 상태 그대로). 바퀴가 아니라 6번만 열었으므로 disable 안 함.
        bus.disconnect(disable_torque=False)


def release_arm(port):
    """팔 토크 해제(unhold). 잡고 있던 것을 놓고 손으로 움직일 수 있게 된다."""
    print("  팔 토크 해제 중...")
    bus = FeetechMotorsBus(port=port, motors=ARM)
    bus.connect()
    try:
        bus.disable_torque(list(ARM))
    finally:
        bus.disconnect(disable_torque=False)  # 이미 껐으므로 재토글 안 함
    print("  팔 토크 해제됨(모터 free).")


# ---------------------------------------------------------------------------
# 카메라
# ---------------------------------------------------------------------------
def grab_frame(cam_device, warmup=15, settle=0.5, open_retries=4, open_wait=0.6):
    """raw cv2 로 카메라 한 프레임을 잡아 PIL 이미지로 반환(없으면 None).

    lerobot OpenCVCamera 가 이 카메라 검증에 실패하므로 다른 vision 스크립트와
    동일하게 raw cv2 를 쓴다(BGR→RGB). 자세 이동 직후 카메라 열기가 일시적으로
    실패할 수 있어 open_retries 만큼 재시도한다.
    """
    # 이 장치를 연속녹화 중인 레코더가 있으면 촬영 동안 잠깐 놓게 한다(동시 오픈 충돌 회피).
    pause_device(cam_device)
    try:
        return _grab_frame_raw(cam_device, warmup, settle, open_retries, open_wait)
    finally:
        resume_device(cam_device)


def _grab_frame_raw(cam_device, warmup, settle, open_retries, open_wait):
    import cv2
    from PIL import Image

    cam = int(cam_device) if str(cam_device).isdigit() else cam_device
    cap = None
    for attempt in range(1, open_retries + 1):
        # V4L2 백엔드 + MJPG 를 강제해야 640x480 이 나온다. 안 하면 YUYV 160x120 으로 떨어진다.
        cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
        if cap.isOpened():
            break
        cap.release()
        cap = None
        print(f"  카메라 열기 실패({attempt}/{open_retries}): {cam_device} → {open_wait}초 후 재시도")
        time.sleep(open_wait)
    if cap is None:
        print(f"  카메라를 열 수 없습니다: {cam_device}")
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        for _ in range(warmup):  # 자동노출/화이트밸런스 안정화
            wret, wframe = cap.read()
            # 녹화 중이면 이 촬영 구간(레코더가 놓은 사이)도 프레임으로 남긴다.
            if wret and wframe is not None:
                save_pipeline_frame(cam_device, wframe)
            time.sleep(0.03)
        time.sleep(settle)
        ret, frame = cap.read()
        if ret and frame is not None:
            save_pipeline_frame(cam_device, frame)
    finally:
        cap.release()
    if not ret or frame is None:
        print("  프레임을 읽지 못했습니다.")
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# 카메라 연속 녹화(사진 프레임) — UVC 웹캠은 동시 오픈이 안 되므로, 파이프라인이
# 같은 장치를 쓰는 순간(접근=front, grab_frame=wrist)에는 레코더를 잠깐 놓았다가
# (장치 release) 다시 이어서 녹화한다. 저장은 10fps JPG 시퀀스(코덱 이슈 없음).
#   ffmpeg -framerate 10 -i front/%06d.jpg front.mp4  로 나중에 영상화.
# 일시정지 구간은 번호를 건너뛰지 않고(연속 증가) 프레임만 빠진다 → 영상이 튄다.
# ---------------------------------------------------------------------------
_RECORDERS = {}   # device(str) -> CamRecorder (grab_frame 이 촬영 순간 자동 pause 하려고)
_SAVE_DIRS = {}   # device(str) -> Path (파이프라인이 그 장치를 쓸 때 프레임을 여기 저장)


def _frame_name():
    """시간순 정렬되는 고정폭 타임스탬프 파일명. 여러 소스(레코더+파이프라인)를
    한 폴더에 섞어도 이름순 정렬 = 시간순이 되게 한다."""
    return f"{time.time():017.6f}.jpg"


def save_pipeline_frame(device, frame_bgr):
    """파이프라인이 이 장치로 읽은 프레임을 녹화 폴더에 저장(레코더가 놓은 구간 메움).

    해당 장치에 녹화 폴더가 등록돼 있을 때만 저장한다(아니면 no-op).
    """
    d = _SAVE_DIRS.get(str(device))
    if d is None:
        return
    try:
        import cv2
        cv2.imwrite(str(Path(d) / _frame_name()), frame_bgr)
    except Exception:
        pass


def frame_dir_for_device(device):
    """이 장치의 녹화 프레임 저장 폴더(없으면 None). subprocess 에 넘길 때 쓴다."""
    d = _SAVE_DIRS.get(str(device))
    return str(d) if d is not None else None


def pause_device(device):
    """해당 장치를 녹화 중인 레코더가 있으면 일시정지(장치 release)하고 완료까지 대기."""
    rec = _RECORDERS.get(str(device))
    if rec is not None:
        rec.pause()


def resume_device(device):
    """해당 장치 레코더가 있으면 녹화 재개."""
    rec = _RECORDERS.get(str(device))
    if rec is not None:
        rec.resume()


class CamRecorder(threading.Thread):
    """한 카메라를 10fps JPG 시퀀스로 연속 저장하는 백그라운드 스레드.

    pause()/resume() 로 장치를 놓았다/다시 잡는다(파이프라인이 그 장치를 쓸 때).
    저장 파일명은 연속 인덱스(000000.jpg…)라 pause 구간에도 번호가 안 끊긴다.
    """

    def __init__(self, device, out_dir, fps=10):
        super().__init__(daemon=True)
        self.device = str(device)
        self.out_dir = Path(out_dir)
        self.period = 1.0 / fps
        self._pause = threading.Event()     # set = 일시정지 요청
        self._released = threading.Event()  # set = 실제로 장치를 놓음(pause 완료)
        self._stop_evt = threading.Event()  # 이름 주의: Thread._stop 을 덮으면 안 됨
        self.saved = 0

    def _open(self):
        import cv2
        cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return cap

    def run(self):
        import cv2
        self.out_dir.mkdir(parents=True, exist_ok=True)
        cap = None
        next_t = time.monotonic()
        while not self._stop_evt.is_set():
            if self._pause.is_set():
                if cap is not None:
                    cap.release()
                    cap = None
                self._released.set()   # pause() 가 이 시점까지 대기
                time.sleep(0.05)
                continue
            if cap is None:
                cap = self._open()
                if cap is None:        # 열기 실패(잠깐 점유 중일 수 있음) → 잠시 후 재시도
                    time.sleep(0.1)
                    continue
            ret, frame = cap.read()
            if ret and frame is not None:
                cv2.imwrite(str(self.out_dir / _frame_name()), frame)
                self.saved += 1
            # 10fps 페이싱
            next_t += self.period
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.monotonic()
        if cap is not None:
            cap.release()

    def pause(self, timeout=3.0):
        """녹화를 멈추고 장치를 실제로 놓을 때까지(최대 timeout) 대기."""
        self._released.clear()
        self._pause.set()
        self._released.wait(timeout=timeout)

    def resume(self):
        self._pause.clear()

    def stop(self):
        self._stop_evt.set()


def start_recorders(devices, run_dir, fps=10):
    """{이름: 장치} 를 받아 각 카메라 레코더를 시작하고 레지스트리에 등록.

    devices 예: {"front": "/dev/video1", "wrist": "/dev/video0"}.
    반환: 이름->CamRecorder. 종료 시 stop_recorders 로 정리한다.
    """
    recs = {}
    run_dir = Path(run_dir)
    for name, dev in devices.items():
        out_dir = run_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)  # 파이프라인이 먼저 쓸 수도 있으니 미리 생성
        rec = CamRecorder(dev, out_dir, fps=fps)
        _RECORDERS[str(dev)] = rec
        _SAVE_DIRS[str(dev)] = out_dir   # 파이프라인 프레임도 같은 폴더에 저장
        recs[name] = rec
        rec.start()
        print(f"  🎥 녹화 시작: {name} ({dev}) → {out_dir}/  (레코더 + 파이프라인 프레임)")
    return recs


def stop_recorders(recs):
    """모든 레코더를 정지·정리하고 저장 프레임 수를 출력."""
    for name, rec in recs.items():
        rec.stop()
    for name, rec in recs.items():
        rec.join(timeout=3.0)
        _RECORDERS.pop(rec.device, None)
        _SAVE_DIRS.pop(rec.device, None)
        total = len(list(rec.out_dir.glob("*.jpg")))  # 레코더+파이프라인 합산
        print(f"  🎥 녹화 종료: {name} → 총 {total}장 저장 ({rec.out_dir})")


def _ffmpeg_bin():
    """ffmpeg 실행 파일 경로(없으면 None). conda 환경 ffmpeg 를 우선 찾는다."""
    import shutil
    for cand in ("ffmpeg", str(Path(sys.executable).parent / "ffmpeg")):
        p = shutil.which(cand) or (cand if Path(cand).exists() else None)
        if p:
            return p
    return None


def assemble_videos(recs, fps=10):
    """각 레코더 폴더의 %06d.jpg 시퀀스를 <name>.mp4 로 묶는다(ffmpeg).

    ffmpeg 가 없거나 프레임이 0장이면 그 카메라는 건너뛴다(사진은 그대로 남는다).
    반환: {이름: 생성된 mp4 경로}.
    """
    ff = _ffmpeg_bin()
    if ff is None:
        print("  ⚠️ ffmpeg 없음 → 영상 합치기 생략(사진 프레임은 그대로 저장됨).")
        return {}
    made = {}
    for name, rec in recs.items():
        n = len(list(rec.out_dir.glob("*.jpg")))  # 레코더+파이프라인 프레임 합산
        if n == 0:
            print(f"  ⚠️ {name}: 저장된 프레임이 없어 영상 생략.")
            continue
        out_mp4 = rec.out_dir.parent / f"{name}.mp4"
        # 파일명이 타임스탬프라 glob 이름순 = 시간순. glob 패턴으로 입력한다.
        base = [ff, "-y", "-nostdin", "-framerate", str(fps),
                "-pattern_type", "glob", "-i", str(rec.out_dir / "*.jpg")]
        # 우선 h264(QuickTime/브라우저 호환) → 실패 시 mpeg4 로 폴백.
        for vargs in (["-c:v", "libx264", "-pix_fmt", "yuv420p"],
                      ["-c:v", "mpeg4", "-q:v", "3"]):
            cmd = base + vargs + [str(out_mp4)]
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                made[name] = out_mp4
                print(f"  🎬 영상 생성: {name} → {out_mp4} ({n}프레임/{fps}fps)")
                break
        else:
            print(f"  ⚠️ {name}: ffmpeg 영상 합치기 실패(사진은 그대로 남음).")
    return made
