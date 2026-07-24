"""[3'] 복귀 단계: laundry_back 자세(순서 이동) → 시작 위치로 주행 복귀.

open_door2 의 base(바퀴) 이동을 되돌려 처음 위치로 돌아온다. 회전은 한 번도
하지 않았으므로(heading 불변) 각 축을 같은 이동종류·같은 속도로 상쇄하면 된다.

  자세: set_pose laundry_back  — 6번(gripper)을 먼저 목표로 옮긴 뒤 2번(shoulder_lift)
        을 움직인다(order="6;2", 나머지 1·3·4·5 는 마지막). 팔 토크는 유지(hold).

  복귀 주행: 방향(strafe_dir/drive_dir)과 시간(strafe_sec/drive_sec)은 호출자가
    계산해서 넘긴다. open_door2 가 지금까지의 base 이동을 축별로 합산해 그 반대
    방향·크기로 지정한다(laundry_open 사후 이동까지 포함). 이 파일 단독 기본값은
    laundry_open 단계 이전 흐름(왼쪽 0.5s + 전진 1.3s)에 맞춰 둔다.

단독 실행:
  python scripts/skills/laundry/return_home.py
  python scripts/skills/laundry/return_home.py --no-wait --strafe-dir 1 --strafe-sec 1.5 --drive-sec 0.3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import drive_straight, set_pose, strafe


def return_home(port="/dev/ttyACM0", back_pose="laundry_back", order="6;2",
                strafe_dir=-1, strafe_sec=0.5, strafe_speed=400,
                drive_dir=1, drive_sec=1.3, drive_speed=500, strafe_sign=1,
                wait_enter=True):
    """[3'] laundry_back 자세(순서 이동) → 시작 위치 복귀. 성공하면 True.

    strafe_dir: +1=오른쪽 / -1=왼쪽,  drive_dir: +1=전진 / -1=후진.
    복귀량(strafe_sec/drive_sec)과 방향은 호출자가 계산해서 넘긴다(open_door2 가
    지금까지의 base 이동을 축별로 상쇄하도록 계산). 기본값은 open_door2 의
    laundry_open 단계 이전 흐름(왼쪽 0.5s + 전진 1.3s)에 맞춰 둔다.
    """
    if wait_enter:
        print("\n" + "🏠" * 20)
        print("복귀 자세로 이동 후 시작 위치로 돌아갑니다.")
        input("준비되면 Enter 를 누르세요 (복귀 시작): ")

    print(f"[3'] 복귀 자세 이동 (set_pose {back_pose}, 순서 {order}, 토크 유지)")
    if not set_pose(back_pose, hold=True, order=order):
        print("  복귀 자세 이동 실패"); return False

    s_label = "right" if strafe_dir > 0 else "left"
    d_label = "전진" if drive_dir > 0 else "후진"
    print(f"[3'b] 시작 위치 복귀: strafe {s_label} {strafe_sec:.2f}s + {d_label} {drive_sec:.2f}s")
    strafe(port, strafe_dir, strafe_sec, strafe_speed, strafe_sign or 1)
    drive_straight(port, drive_dir, drive_sec, drive_speed)
    return True


def main():
    ap = argparse.ArgumentParser(description="복귀: laundry_back 자세(순서) → 시작 위치 주행.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--back-pose", default="laundry_back",
                    help="복귀 자세(poses/<이름>.json)")
    ap.add_argument("--order", default="6;2",
                    help="모터 이동 순서(예: '6;2' = 6번 먼저, 그 다음 2번, 나머지 마지막)")
    ap.add_argument("--strafe-dir", type=int, choices=(-1, 1), default=-1,
                    help="복귀 strafe 방향(+1=오른쪽, -1=왼쪽)")
    ap.add_argument("--strafe-sec", type=float, default=0.5,
                    help="복귀 strafe 시간(초)")
    ap.add_argument("--strafe-speed", type=int, default=400)
    ap.add_argument("--drive-dir", type=int, choices=(-1, 1), default=1,
                    help="복귀 주행 방향(+1=전진, -1=후진)")
    ap.add_argument("--drive-sec", type=float, default=1.3,
                    help="복귀 주행 시간(초)")
    ap.add_argument("--drive-speed", type=int, default=500)
    ap.add_argument("--strafe-sign", type=int, choices=(-1, 1), default=1)
    ap.add_argument("--no-wait", action="store_true", help="시작 전 Enter 대기 생략")
    args = ap.parse_args()

    ok = return_home(args.port, args.back_pose, args.order,
                     args.strafe_dir, args.strafe_sec, args.strafe_speed,
                     args.drive_dir, args.drive_sec, args.drive_speed, args.strafe_sign,
                     wait_enter=not args.no_wait)
    if not ok:
        return 1
    print("✅ 시작 위치 복귀 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
