"""[2'] 문 열기 단계(신규): 준비 자세 → 오른쪽 strafe → 전진.

빨간 손잡이 접근 후(laundry_task 의 Enter 대기 지점)부터 동작하는 새 버전이다.
기존 open_door.py(play_motion "open" + 왼쪽 strafe)와 별개로, 여기서는
set_pose 로 laundry_openready 자세를 먼저 만든 뒤 이동한다.

현재 구현(초안 — 이후 문 열기 동작을 이어 붙일 예정):
  1. set_pose laundry_openready  — 문 열기 준비 자세.
  2. strafe right 0.5s           — 오른쪽으로 살짝 정렬.
  3. 직진 0.7s                    — 손잡이 쪽으로 전진.
  4. set_pose laundry_opencatch  — 손잡이를 무는 자세(hold, 토크 유지).
  5. 후진 2s + 1번 회전 동시      — 뒤로 빠지면서 shoulder_pan(1번)만 nothing.json
                                    값까지 회전(2·3·4·5·6번은 opencatch 값 유지).
  6. return_home                 — laundry_back 자세(6번→2번 순서) → 시작 위치로 복귀
                                    (여기까지의 strafe/전진/후진을 축별 상쇄, 자동 계산).
  7. play_motion openseasame     — 문 열기 모션.
  8. 모션 후 이동                 — 기존 open_door 와 동일: 왼쪽 strafe 2.0s + 전진 1.0s.

함수:
  open_door2(port, ready_pose, strafe_sec, forward_sec,
             strafe_speed, drive_speed, strafe_sign,
             catch_pose, nothing_pose, backdrive_sec,
             open_motion, open_strafe_sec, open_forward_sec,
             back_pose, order, do_return, wait_enter=False)

  기본은 대기 없이 바로 시작한다. --wait(wait_enter=True) 를 주면 동작 시작
  직전에 Enter 를 기다린다(도착 후 사람 확인용).

단독 실행(단계만 테스트):
  python scripts/skills/laundry/open_door2.py
  python scripts/skills/laundry/open_door2.py --wait --strafe-sec 0.7
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import return_home as stage_return  # noqa: E402
from common import (  # noqa: E402
    MOTIONS_DIR,
    POSES_DIR,
    backdrive_and_rotate_pan,
    drive_straight,
    play_motion,
    pose_value,
    set_pose,
    strafe,
)


def open_door2(port="/dev/ttyACM0", ready_pose="laundry_openready",
               strafe_sec=0.5, forward_sec=0.7,
               strafe_speed=400, drive_speed=500, strafe_sign=1,
               catch_pose="laundry_opencatch", nothing_pose="nothing",
               backdrive_sec=2.0,
               open_motion="openseasame", open_strafe_sec=2.0, open_forward_sec=1.0,
               back_pose="laundry_back", order="6;2",
               do_return=True, wait_enter=False):
    """[2'] 준비 자세 → strafe → 전진 → 손잡이 물기 → 후진+1번 회전. 성공하면 True."""
    if wait_enter:
        print("\n" + "🚩" * 20)
        print("도착. 문 열기(신규) 준비 자세로 이동합니다.")
        input("준비되면 Enter 를 누르세요 (문 열기 시작): ")

    print(f"[2'] 준비 자세 이동 (set_pose {ready_pose})")
    if not set_pose(ready_pose):
        print("  준비 자세 이동 실패"); return False

    print(f"[2'b] 오른쪽 strafe {strafe_sec}s 후 전진 {forward_sec}s")
    strafe(port, +1, strafe_sec, strafe_speed, strafe_sign or 1)
    drive_straight(port, +1, forward_sec, drive_speed)

    # [2'c] 손잡이 무는 자세로 이동(토크 유지) 후, 후진과 동시에 1번만 회전.
    if not (POSES_DIR / f"{catch_pose}.json").exists():
        print(f"[2'c] ⚠️ 자세 파일이 없습니다: {POSES_DIR / f'{catch_pose}.json'}")
        print(f"    먼저 저장하세요: 팔을 손잡이 무는 자세로 두고 "
              f"python scripts/arm/save_current_pose.py {catch_pose}")
        return False
    target_pan = pose_value(nothing_pose, "shoulder_pan")
    if target_pan is None:
        print(f"[2'c] ⚠️ '{nothing_pose}' 자세에서 shoulder_pan 값을 못 읽었습니다.")
        return False

    print(f"[2'c] 손잡이 무는 자세 이동 (set_pose {catch_pose}, 토크 유지)")
    if not set_pose(catch_pose, hold=True):
        print("  손잡이 무는 자세 이동 실패"); return False

    print(f"[2'd] 후진 {backdrive_sec}s + 1번(shoulder_pan)→{nothing_pose} 값({target_pan}) 회전")
    backdrive_and_rotate_pan(port, target_pan, seconds=backdrive_sec, drive_speed=drive_speed)

    # 문 열기 모션은 뒤에서 재생하므로, 먼저 존재 여부부터 확인(없으면 조기 중단).
    open_path = MOTIONS_DIR / f"{open_motion}.json"
    if not open_path.exists():
        print(f"[2'e] ⚠️ 문 열기 모션이 없습니다: {open_path}")
        print(f"    먼저 녹화하세요: python scripts/arm/record_motion.py {open_motion}")
        return False

    # [2'e] 먼저 복귀: laundry_back 자세(6번→2번 순서) → 시작 위치로 주행.
    #   복귀는 '이 단계까지의' base 이동만 상쇄한다(뒤의 문 열기 이동은 복귀 후에 함):
    #     좌우(오른쪽 +): 오른쪽 strafe_sec  → 왼쪽 strafe_sec
    #     전후(전진 +)  : 전진 forward_sec, 후진 backdrive_sec → 전진 (backdrive_sec - forward_sec)
    if do_return:
        net_strafe = strafe_sec                       # 오른쪽으로 간 것만
        net_drive = forward_sec - backdrive_sec       # +면 순 전진
        strafe_dir = -1 if net_strafe > 0 else 1      # 순 오른쪽이면 왼쪽으로 복귀
        drive_dir = -1 if net_drive > 0 else 1        # 순 전진이면 후진으로 복귀
        print(f"[2'e] 복귀: {back_pose} 자세(순서 {order}) → 시작 위치 "
              f"(strafe {'right' if strafe_dir > 0 else 'left'} {abs(net_strafe):.2f}s + "
              f"{'전진' if drive_dir > 0 else '후진'} {abs(net_drive):.2f}s)")
        if not stage_return.return_home(
            port=port, back_pose=back_pose, order=order,
            strafe_dir=strafe_dir, strafe_sec=abs(net_strafe), strafe_speed=strafe_speed,
            drive_dir=drive_dir, drive_sec=abs(net_drive), drive_speed=drive_speed,
            strafe_sign=strafe_sign or 1, wait_enter=False,
        ):
            print("  복귀 실패"); return False

    # [2'f] 문 열기 모션 재생 후, 기존 open_door 와 동일한 사후 이동
    #       (왼쪽 strafe open_strafe_sec + 전진 open_forward_sec).
    print(f"[2'f] 문 열기 모션 재생 (play_motion {open_motion})")
    if not play_motion(open_motion, 1):
        print("  문 열기 모션 실패"); return False
    print(f"[2'g] 모션 후 이동: strafe left {open_strafe_sec}s + 전진 {open_forward_sec}s")
    strafe(port, -1, open_strafe_sec, strafe_speed, strafe_sign or 1)
    drive_straight(port, +1, open_forward_sec, drive_speed)
    return True


def main():
    ap = argparse.ArgumentParser(description="문 열기(신규): 준비 자세 → 오른쪽 strafe → 전진.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--ready-pose", default="laundry_openready",
                    help="문 열기 준비 자세(poses/<이름>.json)")
    ap.add_argument("--strafe-sec", type=float, default=0.5)
    ap.add_argument("--forward-sec", type=float, default=0.7)
    ap.add_argument("--strafe-speed", type=int, default=400)
    ap.add_argument("--drive-speed", type=int, default=500)
    ap.add_argument("--strafe-sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--catch-pose", default="laundry_opencatch",
                    help="손잡이 무는 자세(poses/<이름>.json)")
    ap.add_argument("--nothing-pose", default="nothing",
                    help="1번(shoulder_pan) 목표값을 읽어올 자세(poses/<이름>.json)")
    ap.add_argument("--backdrive-sec", type=float, default=2.0,
                    help="후진 + 1번 회전 동시 진행 시간(초)")
    ap.add_argument("--open-motion", default="openseasame",
                    help="문 열기 모션(motions/<이름>.json)")
    ap.add_argument("--open-strafe-sec", type=float, default=2.0,
                    help="문 열기 모션 후 왼쪽 strafe 시간(초, 기존 open_door 기본값)")
    ap.add_argument("--open-forward-sec", type=float, default=1.0,
                    help="문 열기 모션 후 전진 시간(초, 기존 open_door 기본값)")
    ap.add_argument("--back-pose", default="laundry_back",
                    help="복귀 자세(poses/<이름>.json)")
    ap.add_argument("--order", default="6;2",
                    help="복귀 자세 모터 이동 순서(예: '6;2')")
    ap.add_argument("--no-return", action="store_true",
                    help="복귀 자세/시작 위치 복귀 단계를 생략(backdrive 까지만)")
    ap.add_argument("--wait", action="store_true", help="시작 전 Enter 대기(기본은 대기 없음)")
    args = ap.parse_args()

    ok = open_door2(args.port, args.ready_pose, args.strafe_sec, args.forward_sec,
                    args.strafe_speed, args.drive_speed, args.strafe_sign,
                    args.catch_pose, args.nothing_pose, args.backdrive_sec,
                    args.open_motion, args.open_strafe_sec, args.open_forward_sec,
                    args.back_pose, args.order, do_return=not args.no_return,
                    wait_enter=args.wait)
    if not ok:
        return 1
    print("✅ 문 열기(신규) 초안 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
