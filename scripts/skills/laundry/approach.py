"""[0] 주행 자세 + [1] 세탁기 접근 단계.

함수:
  to_drive_pose(pose="straight")           → 팔을 세운 주행 자세로 이동(set_pose subprocess).
  approach(port, front_cam, strafe_sign)   → move_to_destination.py 로 빨간 손잡이까지 접근.

단독 실행(단계만 테스트):
  python scripts/skills/laundry/approach.py                 # 주행자세 → 접근
  python scripts/skills/laundry/approach.py --skip-pose     # 접근만
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    APPROACH_SCRIPT, frame_dir_for_device, pause_device, resume_device,
    run_script, set_pose,
)


def to_drive_pose(pose="laundry_default", hold=False):
    """[0] 주행/기본 자세로 이동. 성공하면 True.

    hold=True 면 이동 후 토크를 유지한다(집은 옷을 유지한 채 판단 자세로 갈 때).
    """
    print(f"[0] 기본 자세로 이동 (set_pose {pose}{' --hold' if hold else ''})")
    return set_pose(pose, hold=hold)


def approach(port="/dev/ttyACM0", front_cam="/dev/video1", strafe_sign=None):
    """[1] move_to_destination.py 로 빨간 손잡이까지 접근. 성공하면 True.

    front_cam 에 연속녹화 폴더가 등록돼 있으면(--record 실행), 접근 중 읽은
    프레임을 그 폴더에 저장하도록 --save-frames 를 넘긴다(레코더가 이 카메라를
    놓은 구간의 영상 공백을 메운다)."""
    print("[1] 세탁기 접근 (move_to_destination, 전면 카메라)")
    cli = ["--port", port, "--cam", front_cam]
    if strafe_sign is not None:
        cli += ["--strafe-sign", str(strafe_sign)]
    save_dir = frame_dir_for_device(front_cam)
    if save_dir is not None:
        cli += ["--save-frames", save_dir]
    # 레코더가 이 카메라를 쓰고 있으면 subprocess 가 열 수 있게 잠깐 놓는다(동시 오픈 불가).
    pause_device(front_cam)
    try:
        return run_script(APPROACH_SCRIPT, *cli)
    finally:
        resume_device(front_cam)


def main():
    ap = argparse.ArgumentParser(description="주행 자세 + 세탁기 접근 단계.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--front-cam", default="/dev/video1",
                    help="접근용 전면 카메라(video1=front, video0=wrist)")
    ap.add_argument("--straight-pose", default="laundry_default",
                    help="접근 전 이동할 기본 자세(poses/<이름>.json)")
    ap.add_argument("--strafe-sign", type=int, choices=(-1, 1), default=None,
                    help="접근 시 오른쪽 strafe 바퀴 부호(중심 정렬이 반대면 뒤집기)")
    ap.add_argument("--skip-pose", action="store_true", help="주행 자세 이동을 생략하고 접근만")
    args = ap.parse_args()

    if not args.skip_pose:
        if not to_drive_pose(args.straight_pose):
            print("  주행 자세 이동 실패"); return 1
    if not approach(args.port, args.front_cam, args.strafe_sign):
        print("  접근 실패"); return 1
    print("✅ 접근 완료.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
