"""세탁물 회수 태스크 v3 (오케스트레이터) — VLA 집기 + 2-stage 분리 판.  [작업중]

⚠️ 아직 미완성 사본. laundry_task2.py 를 그대로 복사한 뒤 집기 스테이지 import 만
   grab_place → grab_place_vla 로 바꿔둔 상태다. 원본(task2/grab_place)은 안 건드림.

계획(상공 VLM 학습 후 실제 구현):
  집기 스테이지를 둘로 분리해 '따로' 관리한다.
    ① grab_and_confirm : 준비자세(grabready) → VLA 집기 → '잡았는지' 판별까지.
                         (판별 authority = 상공 VLM, 커밋 게이트 = 그리퍼 부하/위치)
    ② return_and_throw : 복귀 주행 → 던지기.  (기존 grab_and_return 의 [4][5])
  지금은 grab_place_vla.grab_and_return(합쳐진 스텁)을 그대로 호출한다. 상공 VLM 이
  준비되면 [3]단계를 위 ①②로 쪼개고, VLA 호출/정지(run_vla_grasp)를 실제로 연결한다.

──────────────────────────────────────────────────────────────────────────────
(원본 task2 설명) laundry_task.py 와 동일하되, 무거운 로딩(grasp 판별용 CLIP 모델,

laundry_task.py 와 동작·인자는 동일하되, 무거운 로딩(grasp 판별용 CLIP 모델,
cv2/torch import)을 **시작 Enter 를 누르기 전에** 모두 끝낸다. 원판에서는
grasp 판별기가 [3]단계에 와서야 로드돼(torch+CLIP), 로봇이 접근·문열기를 마친
뒤 한참 멈칫했다. 여기서는 미리 로드해 checker= 로 넘겨주므로 단계 전환이 빠르다.

무엇을 미리 로드하나:
  - grasp 판별기(WasherStatusChecker, CLIP+probe) → grab_and_return(checker=…) 로 전달.
  - cv2 / PIL (손목캠 프레임 취득용) import 워밍업.

여전히 subprocess 인 것(각 호출마다 lerobot 재import 비용이 있음):
  - set_pose / play_motion / move_to_destination — 매 호출 포트를 새로 열고 닫아
    '점유한 채 죽는' 위험을 줄이는 기존 안전 설계라 그대로 둔다.

사용:
  python scripts/skills/laundry_task2.py                 # 기본값
  python scripts/skills/laundry_task2.py --open-strafe-sec 1.5 --strafe-sign -1
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "laundry"))
import time  # noqa: E402

import approach as stage_approach  # noqa: E402
import open_door2 as stage_open  # noqa: E402
import grab_place_vla as stage_grab  # noqa: E402  (VLA 집기판. task2 는 grab_place)
from common import (  # noqa: E402
    REPO, assemble_videos, start_recorders, stop_recorders, stop_wheels,
)


def preload(grasp_probe):
    """시작 전 무거운 것들을 미리 로드한다. grasp 판별기(checker)를 반환(실패 시 None).

    - cv2/PIL: 손목캠 프레임 취득([3]단계) 첫 import 지연을 미리 치른다.
    - WasherStatusChecker: CLIP + probe 로드(가장 큰 지연). checker 로 반환해
      grab_and_return(checker=…) 에 넘긴다.
    """
    print("사전 로딩 중 (라이브러리·모델)... 잠시만 기다려 주세요.")

    # 카메라 프레임용 라이브러리 워밍업(첫 촬영 때 import 로 멈칫하지 않게).
    try:
        import cv2  # noqa: F401
        import PIL  # noqa: F401
        print("  cv2/PIL 준비 완료")
    except Exception as e:
        print(f"  ⚠️ cv2/PIL import 경고: {e}")

    # VLA 상주 서버 워밍업 — 랩탑 SmolVLA(≈865MB)를 '지금' 로드시켜 집기 순간 스톨을 없앤다.
    # 로봇이 접근·문 여는 동안 랩탑에서 병렬로 로드된다. 실패해도 집기 때 one-shot 으로 폴백.
    try:
        if stage_grab.warmup_vla_server():
            print("  VLA 상주 서버 준비 완료(랩탑 SmolVLA 로드됨)")
        else:
            print("  ⚠️ VLA 서버 워밍업 실패 → 집기 순간 로드(스톨 가능)")
    except Exception as e:
        print(f"  ⚠️ VLA 워밍업 경고: {e}")

    # grasp 판별기(CLIP + probe) — Option B(VLA)에선 랩탑이 판별하므로 선택적.
    if not Path(grasp_probe).exists():
        print(f"  ⚠️ grasp probe 파일 없음: {grasp_probe} → 판별기 사전 로딩 건너뜀")
        return None
    checker = stage_grab.load_checker(grasp_probe)
    if checker is None:
        print("  ⚠️ grasp 판별기 사전 로딩 실패")
        return None
    print("사전 로딩 완료. 이제 시작하면 단계 전환이 빠릅니다.")
    return checker


def main():
    ap = argparse.ArgumentParser(description="세탁물 태스크(한 벌): 접근 → 문열기 → 집기. (사전 로딩판)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--front-cam", default="/dev/video1",
                    help="접근용 전면 카메라(video1=front, video0=wrist)")
    ap.add_argument("--default-pose", default="laundry_default",
                    help="접근 전 주행 자세 겸 집기 후 판단 자세(poses/<이름>.json)")
    ap.add_argument("--grab-motion", default="laundry_grab",
                    help="옷 집기 동작(motions/<이름>.json)")
    ap.add_argument("--camera", default="/dev/video0", help="grasp 판별용 손목캠(video0=wrist)")
    ap.add_argument("--grasp-probe", default=None, help="grasp probe 경로(기본 grasp_probe.pt)")
    ap.add_argument("--success-conf", type=float, default=0.70, help="grasp 성공 인정 신뢰도")
    ap.add_argument("--max-attempts", type=int, default=3, help="집기 최대 재시도")
    ap.add_argument("--backup-sec", type=float, default=5.0, help="집은 뒤 후진 시간(초)")
    ap.add_argument("--rotate-sec", type=float, default=16.9,
                    help="180도 회전 시간(초). 실측: 시계방향 speed 300 에서 16.9초")
    ap.add_argument("--rotate-speed", type=int, default=300, help="회전 속도")
    ap.add_argument("--forward-sec", type=float, default=5.0, help="회전 후 직진 시간(초)")
    ap.add_argument("--throw-motion", default="laundry_throw",
                    help="복귀 후 던지기 동작(없으면 건너뜀). 이동 검증 후 녹화")
    ap.add_argument("--strafe-sign", type=int, choices=(-1, 1), default=None,
                    help="접근 시 오른쪽 strafe 바퀴 부호(중심 정렬이 반대면 뒤집기)")
    ap.add_argument("--open-motion", default="openseasame", help="문 열기 동작(motions/<이름>.json)")
    ap.add_argument("--open-strafe-sec", type=float, default=2.0,
                    help="문 열기 후 왼쪽으로 strafe 하는 시간(초)")
    ap.add_argument("--open-forward-sec", type=float, default=1.0,
                    help="왼쪽 strafe 후 앞으로 전진하는 시간(초)")
    ap.add_argument("--drive-speed", type=int, default=500, help="전/후진 바퀴 속도")
    ap.add_argument("--strafe-speed", type=int, default=400, help="strafe(평행이동) 바퀴 속도")
    ap.add_argument("--yes", action="store_true", help="시작 확인 없이 바로 실행")
    ap.add_argument("--record", action="store_true",
                    help="front/wrist 카메라를 10fps JPG 시퀀스로 연속 녹화(runs/<타임스탬프>/). "
                         "접근·집기 순간엔 해당 카메라를 잠깐 놓아 충돌을 피한다(그 구간만 프레임 공백).")
    ap.add_argument("--record-dir", default=None,
                    help="녹화 저장 루트(기본: <repo>/runs/<타임스탬프>)")
    ap.add_argument("--record-fps", type=int, default=10, help="녹화 프레임레이트(장당 저장)")
    args = ap.parse_args()

    grasp_probe = args.grasp_probe or str(REPO / "grasp_probe.pt")

    print("=" * 60)
    print("세탁물 태스크(한 벌): 접근 → 문열기 → 집기  [사전 로딩판]")
    print("=" * 60)

    # --- 시작 Enter '전에' 무거운 로딩을 모두 끝낸다 ---
    checker = preload(grasp_probe)

    if not args.yes:
        if input("\n시작하려면 Enter, 취소하려면 그 외 입력 후 Enter: ").strip() != "":
            print("취소되었습니다.")
            return

    # --- 연속 녹화 시작(--record) : front/wrist 를 10fps JPG 시퀀스로 저장 ---
    recs = {}
    run_dir = None
    if args.record:
        run_dir = Path(args.record_dir) if args.record_dir \
            else REPO / "runs" / time.strftime("%Y%m%d_%H%M%S")
        print(f"\n🎥 카메라 녹화 켜짐 → {run_dir}")
        recs = start_recorders(
            {"front": args.front_cam, "wrist": args.camera},
            run_dir, fps=args.record_fps,
        )

    try:
        # [0] 기본 자세
        if not stage_approach.to_drive_pose(args.default_pose):
            print("  기본 자세 이동 실패 → 중단"); return

        # [1] 접근 (approach 내부에서 front 레코더를 잠깐 놓고, 접근 프레임을 녹화 폴더에 저장)
        if not stage_approach.approach(args.port, args.front_cam, args.strafe_sign):
            print("  접근 실패 → 중단"); return

        # [2] 문 열기 (신규 open_door2: 준비자세 → strafe/전진 → 손잡이 물기 →
        #     후진+1번 회전 → 복귀 → openseasame 모션 → strafe left → 전진)
        if not stage_open.open_door2(
            port=args.port, open_motion=args.open_motion,
            open_strafe_sec=args.open_strafe_sec, open_forward_sec=args.open_forward_sec,
            strafe_speed=args.strafe_speed, drive_speed=args.drive_speed,
            strafe_sign=args.strafe_sign or 1, wait_enter=False,
        ):
            print("  문 열기 실패 → 중단"); return

        # [3] 집기 → grasp 판별(재시도) → 복귀 주행(후진·180도 회전·직진) → 던지기.
        #     checker 는 위에서 미리 로드해 넘긴다(단계 진입 시 멈칫 없음).
        print("\n[3] 집기 → 판별 → 복귀 주행 (grab_place)")
        if not stage_grab.grab_and_return(
            port=args.port, camera=args.camera, grab_motion=args.grab_motion,
            judge_pose=args.default_pose, grasp_probe=grasp_probe,
            success_conf=args.success_conf, max_attempts=args.max_attempts,
            backup_sec=args.backup_sec, rotate_sec=args.rotate_sec,
            rotate_speed=args.rotate_speed, forward_sec=args.forward_sec,
            drive_speed=args.drive_speed, clockwise=True, throw_motion=args.throw_motion,
            checker=checker,
        ):
            print("  집기/복귀 실패 → 중단"); return

        print("\n✅ 접근 → 문열기 → 집기 → 복귀 완료.")

    except KeyboardInterrupt:
        print("\n[중단] 바퀴를 정지합니다.")
        try:
            stop_wheels(args.port)
        except Exception:
            pass
    finally:
        # --- 녹화 종료 → 모든 동작 끝난 뒤 사진을 영상으로 묶기 ---
        if recs:
            print("\n🎥 녹화 정리 중...")
            stop_recorders(recs)
            assemble_videos(recs, fps=args.record_fps)
            print(f"\n📂 결과물: {run_dir}")
            print("   - front/, wrist/ : 원본 JPG 프레임 (%06d.jpg)")
            print("   - front.mp4, wrist.mp4 : 합친 영상(ffmpeg 성공 시)")
            print("   내 컴퓨터로 가져오기(scp) 예시:")
            print(f"     scp -r comnet02@<로봇IP>:{run_dir} .")


if __name__ == "__main__":
    main()
