"""저장된 포즈 JSON을 읽어 1~6번 모터를 그 자세로 이동시키는 스크립트.

사용법:
    uv run python set_pose.py [포즈이름]

포즈이름을 인자로 주지 않으면 'ready_pose'를 사용합니다.
읽는 위치: poses/<포즈이름>.json

[중요] 모터는 반드시 POSITION 모드(Operating_Mode=0)에서 움직인다.
    STS3215의 Operating_Mode는 다음과 같다:
        0 = POSITION : Goal_Position 이 '절대 목표 위치'. 모터가 알아서 그 위치로
                       올바른 방향으로 돌아간다. (lerobot 본체/키보드 스크립트와 동일)
        1 = VELOCITY : 등속 회전(바퀴용).
        2 = PWM      : 개루프 속도.
        3 = STEP     : Goal_Position 이 '이동할 스텝 수'이고 bit15가 방향 비트.
                       절대 위치가 아니다!
    예전 버전은 이동 직전에 팔을 mode 3(STEP)로 바꾼 뒤 절대 위치값(예: 2040)을
    Goal_Position에 썼다. STEP 모드에서 이 값은 '스텝 수 + 방향 비트'로 해석되어
    모터가 엉뚱한/반대 방향으로 돌았다. POSITION 모드를 유지하는 것으로 해결한다.

이동 속도는 Operating_Mode를 바꾸지 않고 Acceleration / Goal_Velocity 레지스터로
조절한다. POSITION 모드에서 Goal_Velocity 는 목표까지 가는 '최고 속도'(steps/s),
Acceleration 은 가감속을 의미한다.
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

# === 버스 쓰기 재시도 ===
# 모터 9개가 한 데이지체인에 물려 있어 응답 패킷이 이따금 깨진다("Incorrect status
# packet!"). lerobot의 write/enable_torque 기본값은 num_retry=0 이라 그 한 번의
# 글리치가 곧바로 예외가 되고, 자세 이동 전체(및 이를 호출한 태스크)가 죽는다.
BUS_RETRY = 3

# === 이동 속도/가속 (POSITION 모드에서 유효) ===
# MOVE_VELOCITY: 목표까지 가는 최고 속도(steps/s). 0이면 최대 속도. 작을수록 느리다.
# MOVE_ACCELERATION: 가감속(값 * 100 steps/s^2). 작을수록 부드럽게 출발/정지.
MOVE_VELOCITY = 300
MOVE_ACCELERATION = 20

# === 도달 판정/타임아웃 ===
# TOLERANCE: 목표와의 오차가 이 값 이내면 도달한 것으로 본다(steps).
# TIMEOUT: 이 시간(초) 안에 도달하지 못하면 종료한다.
TOLERANCE = 15
TIMEOUT = 30.0

# === 정체(stall) 감지 ===
# 목표 오차가 tolerance 안에 못 들어와도, 팔이 '더는 안 움직이면'(limit에 걸렸거나
# 물리적으로 막힘) 자세가 완성된 것으로 보고 즉시 멈춘다.
# STALL_EPS: 직전 프레임 대비 그 모터의 위치 변화가 이 값 이하이면 '안 움직임'.
# STALL_FRAMES: '안 움직임'이 연속 이만큼이면 완성으로 판단.
# STALL_GRACE_S: 시작 직후 무거운 관절(특히 2번 shoulder_lift)이 중력을 이기고
#               출발하기까지 토크를 올리는 데 걸리는 시간을 감안한 유예(초). 이 시간이
#               지나기 전에는, '그 모터 자신'이 한 번이라도 실제로 움직였을 때만 정체로
#               본다. 예전(0.5초·전역 moved 플래그)에는 가벼운 모터가 먼저 움직이면
#               곧바로 무거운 2번을 '출발 전에' 막힘으로 오인해 고정(hold)해 버렸다.
STALL_EPS = 4
STALL_FRAMES = 3
STALL_GRACE_S = 2.0

POSES_DIR = Path("/home/comnet02/lerobot/poses")
CALIB_PATH = Path(
    "/home/comnet02/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_awesome_kiwi.json"
)

# 바퀴는 연결하지 않는다. 팔(1~6번) 모터만 등록·초기화한다.
MOTORS = {
    "shoulder_pan":  Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex":    Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex":    Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll":    Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper":       Motor(6, "sts3215", MotorNormMode.DEGREES),
}

ARM_MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]


def _hold_here(bus, names):
    """모든 모터의 현재 위치를 목표로 다시 써서 그 자리에 즉시 정지/고정시킨다.

    POSITION 모드에서는 '현재 위치 = 목표'가 되므로 모터가 그 자리에서 멈춘다.
    """
    for name in names:
        _hold_one(bus, name)


def _hold_one(bus, name):
    """모터 하나를 현재 위치로 고정. 통신 실패는 삼켜서(hold 시도) 흐름을 끊지 않는다."""
    try:
        cur = bus.read("Present_Position", name, normalize=False, num_retry=BUS_RETRY)
        bus.write("Goal_Position", name, cur, normalize=False, num_retry=BUS_RETRY)
    except Exception as e:
        print(f"\n  ⚠️ {name} 고정(hold) 시도 실패({e}) — 무시하고 계속")


def _parse_order(spec, goals):
    """'6;2' 같은 순서 문자열을 [[모터이름,...], ...] phase 리스트로 변환.

    세미콜론(;)이 단계 구분, 콤마(,)가 같은 단계에서 동시 이동. 토큰은 모터
    번호(1~6) 또는 이름(gripper 등). 이 자세의 이동 대상(goals)에 없는 토큰은
    무시하고, --order 에 언급되지 않은 나머지 모터는 마지막 단계로 자동 추가한다.
    """
    num_to_name = {MOTORS[n].id: n for n in MOTORS}
    phases = []
    used = set()
    for chunk in spec.split(";"):
        group = []
        for tok in (t.strip() for t in chunk.split(",")):
            if not tok:
                continue
            name = num_to_name.get(int(tok)) if tok.isdigit() else tok
            if name not in goals:
                print(f"  ⚠️ --order: '{tok}' 는 이 자세의 이동 대상이 아니라 무시합니다.")
                continue
            if name in used:
                continue
            group.append(name)
            used.add(name)
        if group:
            phases.append(group)
    leftover = [n for n in goals if n not in used]
    if leftover:
        phases.append(leftover)  # 순서 미지정 모터는 마지막 단계에서 함께 이동
    return phases


def move_to_goals(bus, goals, tolerance=TOLERANCE, timeout=TIMEOUT, quiet=False):
    """POSITION 모드에서 절대 목표 위치로 이동시키고, 도달할 때까지 진행 상황을 표시.

    동작 방식:
      1) 각 모터에 Goal_Position(절대 목표)를 '한 번' 쓴다. 모터 내부 컨트롤러가
         설정된 Goal_Velocity/Acceleration 으로 올바른 방향으로 알아서 이동한다.
      2) 루프를 돌며 Present_Position 을 읽어 진행 상황을 출력한다.
      3) 모든 모터가 tolerance 이내로 들어오면 완료. timeout 초과 시 중단.
      4) 'q' 가 눌리면 현재 위치를 목표로 다시 써서 그 자리에 즉시 정지한다.

    [모터별 내성] 특정 모터(특히 그리퍼)가 옷/물체에 막혀 목표에 못 닿거나,
    과부하로 read/write 가 실패해도 '오류로 중단'하지 않는다. 그 모터만 현재
    위치로 고정(hold)하고 '완성 간주(stuck)' 처리한 뒤 나머지 모터로 계속 진행한다.
    이렇게 하면 그리퍼가 물건을 문 채 목표 각도까지 못 닫혀도 자세 이동이
    성공으로 끝난다(잡은 것은 계속 물고 있음).
    """
    # 1) 절대 목표를 한 번씩 쓴다(모터별로 실패해도 나머지는 진행).
    for name, goal in goals.items():
        try:
            bus.write("Goal_Position", name, goal, normalize=False, num_retry=BUS_RETRY)
        except Exception as e:
            print(f"\n  ⚠️ {name} 목표 쓰기 실패({e}) → 현재 위치 유지하고 계속")

    is_tty = sys.stdin.isatty()
    old_attr = None
    if is_tty:
        fd = sys.stdin.fileno()
        old_attr = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    prev = {}                        # 직전 프레임의 각 모터 위치
    stall = {n: 0 for n in goals}    # 모터별 연속 '안 움직임' 횟수
    stuck = set()                    # 더 못 움직여 고정(hold)하고 완성 간주한 모터
    moved = {n: False for n in goals}  # 모터별: 그 모터가 시작 이후 실제로 움직였는지
    try:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            # q 입력 감시 (비차단)
            if is_tty:
                r, _, _ = select.select([sys.stdin], [], [], 0)
                if r and sys.stdin.read(1).lower() == "q":
                    print("\n[q] 강제 정지! 현재 위치에서 멈춥니다.")
                    _hold_here(bus, goals)
                    return

            done = True
            parts = []
            cur_all = {}
            for name, goal in goals.items():
                if name in stuck:
                    parts.append(f"{name}:hold")
                    continue
                try:
                    cur = bus.read("Present_Position", name, normalize=False, num_retry=3)
                except Exception as e:
                    # 읽기 실패(과부하/통신오류): 이 모터만 고정하고 완성 간주, 계속 진행.
                    print(f"\n  ⚠️ {name} 읽기 실패({e}) → 고정(hold)하고 계속")
                    _hold_one(bus, name)
                    stuck.add(name)
                    continue
                cur_all[name] = cur
                error = goal - cur
                if abs(error) > tolerance:
                    done = False
                parts.append(f"{name}:{cur}→{goal}({error:+d})")

            if not quiet:
                print("  " + "  ".join(parts), end="\r", flush=True)

            if done:  # 모든 모터가 tolerance 이내 또는 stuck(고정) 상태
                if not quiet:
                    print()  # 진행 표시 줄 마무리
                return

            # === 모터별 정체 감지: 어떤 모터가 연속 STALL_FRAMES 프레임 안 움직이고
            #     아직 목표에 못 닿았으면 그 모터만 고정(hold)하고 완성 간주 ===
            grace_passed = (time.monotonic() - start) > STALL_GRACE_S
            for name, cur in cur_all.items():
                if name in prev:
                    if abs(cur - prev[name]) > STALL_EPS:
                        moved[name] = True
                        stall[name] = 0
                    else:
                        stall[name] += 1
                    # '이 모터 자신'이 한 번이라도 움직였거나(움직이다 리밋/장애물에
                    # 막힌 경우) 시작 유예가 지난 뒤에만 정체로 본다. 아직 출발도 못 한
                    # 무거운 관절을 다른 모터가 먼저 움직였다는 이유로 고정하지 않는다.
                    if ((moved[name] or grace_passed) and stall[name] >= STALL_FRAMES
                            and abs(goals[name] - cur) > tolerance):
                        if not quiet:
                            print(f"\n  '{name}' 더 이상 움직이지 않아 현재 위치로 고정합니다 "
                                  f"(연속 {STALL_FRAMES}프레임 변화 ≤ {STALL_EPS}, 막힘/limit 추정).")
                        _hold_one(bus, name)
                        stuck.add(name)
            prev = cur_all

            time.sleep(0.05)

        print("\n시간 초과로 종료합니다.")
    finally:
        if is_tty and old_attr is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_attr)


def main():
    argv = sys.argv[1:]
    auto = ("--yes" in argv) or ("-y" in argv)
    quiet = ("--quiet" in argv) or ("-q" in argv)
    # --hold: 도달 후 토크를 끄지 않고 유지한다(잡은 것을 놓지 않게). 또한 시작 시
    #   토크를 껐다 켜는 재초기화를 생략한다(직전 play_motion --hold 로 POSITION 모드+
    #   토크 ON 상태를 이어받아, 재초기화 중 그립이 풀려 물건을 떨어뜨리는 것을 막음).
    hold = "--hold" in argv
    # --order "6;2": 지정한 모터를 이 순서(단계)대로 이동한다(세미콜론=단계 구분,
    #   콤마=동시 이동, 번호 또는 이름). 나머지 모터는 마지막 단계에서 함께 이동.
    #   지정하지 않으면 기존처럼 전 모터를 동시에 이동한다.
    order_spec = None
    if "--order" in argv:
        i = argv.index("--order")
        order_spec = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    # --skip "6" 또는 "gripper,3": 지정한 모터는 이번 이동에서 '건드리지 않는다'
    #   (목표를 쓰지 않아 그 모터의 현재 목표/토크가 그대로 유지된다). 순응제어로
    #   그리퍼를 물린 채 팔만 다른 자세로 옮길 때 그리퍼(6)를 제외하는 용도.
    skip_spec = None
    if "--skip" in argv:
        i = argv.index("--skip")
        skip_spec = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
    argv = [a for a in argv if a not in ("--yes", "-y", "--quiet", "-q", "--hold")]
    pose_name = argv[0] if argv else "ready_pose"
    pose_path = POSES_DIR / f"{pose_name}.json"
    if not pose_path.exists():
        print(f"포즈 파일을 찾을 수 없습니다: {pose_path}")
        return

    with open(pose_path) as f:
        pose = json.load(f)

    # --skip 대상 모터 이름 집합(번호/이름 혼용 허용). 이 모터는 목표를 쓰지 않는다.
    skip_names = set()
    if skip_spec:
        num_to_name = {MOTORS[n].id: n for n in MOTORS}
        for tok in (t.strip() for t in skip_spec.split(",")):
            if not tok:
                continue
            name = num_to_name.get(int(tok)) if tok.isdigit() else tok
            if name in MOTORS:
                skip_names.add(name)
            else:
                print(f"  ⚠️ --skip: 알 수 없는 모터 '{tok}' 무시")

    # 캘리브레이션 범위 로드 (calibration limit 바깥으로 못 나가게 클램프)
    limits = {}
    if CALIB_PATH.exists():
        with open(CALIB_PATH) as f:
            calib = json.load(f)
        for name in ARM_MOTOR_NAMES:
            entry = calib.get(f"arm_{name}")
            if entry is not None:
                limits[name] = (entry["range_min"], entry["range_max"])
    else:
        print(f"경고: 캘리브레이션 파일을 찾을 수 없어 범위 제한을 적용하지 않습니다: {CALIB_PATH}")

    bus = FeetechMotorsBus(port=PORT, motors=MOTORS)
    bus.connect()
    try:
        # === 초기화 시퀀스 ===
        # connect 직후 곧바로 read하면 버스 동기화가 안 돼 0이 나온다.
        # disable_torque -> Operating_Mode 설정 -> enable_torque 순서로 버스를
        # 깨운 뒤에 read해야 Present_Position이 정상값으로 읽힌다.
        # Operating_Mode는 EEPROM 레지스터라 쓰기 전에 토크/Lock을 먼저 해제해야 한다.
        if hold:
            # 이미 잡고 있는 것을 놓지 않도록 토크 해제/모드 재설정을 생략하고,
            # 토크만 확실히 ON 으로 둔다(enable_torque 는 켜져 있어도 무해하며,
            # connect 직후 버스를 깨우는 트래픽 역할도 한다). POSITION 모드는
            # 직전 단계(play_motion/이전 set_pose)에서 이미 설정됐다고 가정한다.
            # 모터별로 켠다: 직전 그립에서 그리퍼가 과부하(Overload) 래치된 경우
            # 그 모터의 enable_torque 만 실패할 수 있는데, 그것 때문에 자세 이동
            # 전체가 죽지 않게 한다(나머지 모터는 정상 이동, 이동 루프가 그리퍼는
            # stuck 로 처리).
            for name in ARM_MOTOR_NAMES:
                try:
                    bus.enable_torque(name, num_retry=BUS_RETRY)
                except Exception as e:
                    print(f"  ⚠️ {name} enable_torque 실패(과부하/통신 추정) → "
                          f"건너뜀: {str(e).splitlines()[0]}")
        else:
            bus.disable_torque(ARM_MOTOR_NAMES, num_retry=BUS_RETRY)
            # 팔: POSITION(0) 모드. 이동할 때도 이 POSITION 모드를 그대로 유지한다
            # (절대로 STEP 모드로 바꾸지 않는다).
            for name in ARM_MOTOR_NAMES:
                bus.write("Operating_Mode", name, 0, normalize=False, num_retry=BUS_RETRY)  # 0 = POSITION
            # 팔 토크 ON (Torque_Enable=1 + Lock=1)
            bus.enable_torque(ARM_MOTOR_NAMES, num_retry=BUS_RETRY)

        # 현재 모터 값 읽기 및 목표 위치 계산
        if not quiet:
            print(f"\n'{pose_name}' 자세로 이동 계획:")
            print(f"  {'모터':<14} {'현재':>8} {'목표':>8} {'회전량':>10}")
            print(f"  {'-' * 42}")

        goals = {}
        moves = []  # quiet 요약용: 실제로 움직이는 모터
        for name, pos in pose.items():
            if name not in MOTORS:
                if not quiet:
                    print(f"  (건너뜀: 알 수 없는 모터 '{name}')")
                continue
            if name in skip_names:
                if not quiet:
                    print(f"  (제외(--skip): '{name}' 는 이번 이동에서 건드리지 않음)")
                continue
            try:
                current = bus.read("Present_Position", name, normalize=False, num_retry=3)
            except Exception as e:
                # 계획 표시용 현재값 읽기 실패는 치명적이지 않다. 목표만 쓰고 진행한다.
                if not quiet:
                    print(f"  ({name} 현재값 읽기 실패: {e} → 목표만 설정)")
                current = int(pos)
            goal = int(pos)
            note = ""
            if name in limits:
                lo, hi = limits[name]
                clamped = max(lo, min(hi, goal))
                if clamped != goal:
                    note = f"  (limit [{lo}, {hi}] 적용)"
                    goal = clamped
            delta = goal - current
            goals[name] = goal
            if abs(delta) > TOLERANCE:
                moves.append(f"{name}({delta:+d})")
            if not quiet:
                print(f"  {name:<14} {current:>8} {goal:>8} {delta:>+10}{note}")

        # quiet: 상세 표/진행 없이 '어떻게 움직이는지' 한 줄 요약만.
        if quiet:
            summary = ", ".join(moves) if moves else "이미 목표 자세"
            print(f"  → '{pose_name}' 자세로 이동: {summary}")

        # 사용자 확인 (엔터 입력 시 이동). --yes 면 확인 없이 진행(자동 루프용).
        if not auto:
            answer = input("\n이동하려면 Enter, 취소하려면 그 외 입력 후 Enter: ")
            if answer.strip() != "":
                print("취소되었습니다.")
                return

        # POSITION 모드를 유지한 채 속도/가속만 설정한다(모드 변경 없음).
        # Acceleration 은 EEPROM/SRAM 어느 쪽이든 토크가 켜진 상태에서 써도 된다.
        for name in goals:
            bus.write("Acceleration", name, MOVE_ACCELERATION, normalize=False, num_retry=BUS_RETRY)
            bus.write("Goal_Velocity", name, MOVE_VELOCITY, normalize=False, num_retry=BUS_RETRY)

        if not quiet:
            print(f"'{pose_name}' 자세로 이동 중... "
                  f"(속도 {MOVE_VELOCITY}, 가속 {MOVE_ACCELERATION}, 정지하려면 q)")

        if order_spec:
            phases = _parse_order(order_spec, goals)
            for gi, group in enumerate(phases, 1):
                sub = {n: goals[n] for n in group}
                if not quiet:
                    print(f"[순서 {gi}/{len(phases)}] 이동: {', '.join(group)}")
                move_to_goals(bus, sub, quiet=quiet)
        else:
            move_to_goals(bus, goals, quiet=quiet)
        if not quiet:
            print("완료!" + ("  [--hold] 토크 유지" if hold else ""))
    finally:
        # --hold 면 토크를 켠 채 포트만 닫는다(잡은 것을 놓지 않음).
        bus.disconnect(disable_torque=not hold)


if __name__ == "__main__":
    main()
