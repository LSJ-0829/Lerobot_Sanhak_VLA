"""[3] 옷 집기 → grasp 판별(재시도) → 복귀 주행(후진·180도 회전·직진) → 던지기.

옷 한 벌 시나리오. 집은 뒤 손목캠으로 grabbed/failed 를 판별하고, 성공하면
다음 주자 위치로 복귀 주행한 뒤 옷을 던져준다. 이동 내내 팔은 판단 자세
(laundry_default)로 옷을 문 채 유지한다(바퀴 제어는 팔 토크를 건드리지 않음).

동작:
  집기 루프(최대 max_attempts 회):
    1) grab 모션 재생(--hold, 그립 유지).
    2) 판단 자세(judge_pose=laundry_default)로 이동(--hold, 그립 유지).
    3) 손목캠 한 프레임 → grasp probe 로 grabbed/failed 판별.
    4) success_class 이고 신뢰도 >= success_conf 면 성공, 아니면 재시도.
  성공 시 복귀 주행:
    5) backup_sec 초 후진.
    6) 180도 시계방향 회전(실측: speed 300 에서 17.3초 ≈ 180도).
    7) forward_sec 초 직진.
    8) 던지기: throw_motion 이 있으면 재생(옷을 던짐). 아직 없으면 여기서 멈추고
       옷을 문 채로 유지한다(던지기 모션은 이동이 검증된 뒤 녹화한다).
    9) 그리퍼를 연 뒤 retreat_delay 초 대기하고 retreat_sec 초 후진(옷/바구니에서 빠짐).

전제:
  - motions/laundry_grab.json, poses/laundry_default.json, grasp_probe.pt (필수)
  - motions/<throw_motion>.json (선택 — 없으면 던지기 단계 건너뜀)

단독 실행:
  python scripts/skills/laundry/grab_place.py --yes
  python scripts/skills/laundry/grab_place.py --backup-sec 5 --rotate-sec 2.2 --forward-sec 3
"""

import argparse
import os
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


def load_checker(grasp_probe):
    """grabbed/failed 판별기 로드. 실패하면 None."""
    sys.path.insert(0, str(REPO))
    from washer_clip_check import WasherStatusChecker

    print("grasp 판별기 로딩...")
    checker = WasherStatusChecker(probe_path=str(grasp_probe), side_probe_path=None)
    if checker.probe is None:
        print(f"  probe 로드 실패: {grasp_probe}")
        return None
    acc = checker.probe.get("cv_acc")
    print(f"  probe 클래스={checker.probe['classes']}"
          + (f", cv정확도 {acc:.1%}" if acc is not None else ""))
    return checker


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
                    overload_torque=80, protective_torque=80, protection_current=500):
    """집기→판별→(성공 시)복귀 주행→던지기. 성공하면 True.

    compliant=True 면 6번(gripper)에만 순응제어(토크 제한 하 연속 닫기)를 적용한다:
    집기 모션 freeze 를 끄고, 판단 자세에서 6번을 제외하며, 주행 전 재압착해서
    무는 힘이 풀리지 않고 상한(grip_torque)까지 계속 물게 한다. grip_close 로 닫힘
    목표(기본=calib range_min≈완전히 닫힘)를 조절한다.

    [EEPROM 변경은 기본 OFF] 아래 셋은 6번 모터 EEPROM 을 영구 변경하고 teleop 에도
    영향을 주므로 명시적으로 켤 때만 적용한다(과열·손상 위험):
      grip_max_torque : 값을 주면 Max_Torque_Limit 을 그 값으로 올림(기본 None=안 건드림).
      relax_overload  : 과부하 문턱/전류 완화(기본 False).
      disable_overload: 과부하 차단(unload) 비트 제거(기본 False).
    복구는 common.restore_gripper_defaults(port).
    """
    grasp_probe = Path(grasp_probe or (REPO / "grasp_probe.pt"))

    # 필수 준비물 점검(throw 모션은 선택)
    grab_path = MOTIONS_DIR / f"{grab_motion}.json"
    judge_path = POSES_DIR / f"{judge_pose}.json"
    missing = [str(p) for p in [grab_path, judge_path, grasp_probe] if not p.exists()]
    if missing:
        print("다음 파일이 없어 실행할 수 없습니다:")
        for m in missing:
            print(f"  - {m}")
        return False

    if checker is None:
        checker = load_checker(grasp_probe)
        if checker is None:
            return False
    classes = checker.probe["classes"]
    if success_class not in classes:
        print(f"  success_class '{success_class}' 가 probe 클래스 {classes} 에 없습니다")
        return False

    # 순응제어 준비: 집기 모션 '전'에 6번 힘 상한을 세팅한다(옷 물기 전이라 토크 꺼도 안전).
    #   (a) Max_Torque_Limit(EEPROM 절대 상한)을 grip_max_torque 로 올려 더 세게 물 수 있게.
    #   (b) Torque_Limit(런타임 상한)을 grip_torque 로 맞춰 이후 모든 그립을 compliance 로.
    # set_pose/play_motion 은 이 두 레지스터를 안 건드려 subprocess 경계를 넘어 유지된다.
    if compliant:
        print(f"[0] 6번(gripper) 순응제어 준비: Torque_Limit={grip_torque}")
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
        compliant_grip(port, torque_limit=grip_torque, command_close=False)

    # ---- 집기 루프 ----
    success = False
    for attempt in range(1, max_attempts + 1):
        print(f"\n===== 집기 시도 {attempt}/{max_attempts} =====")
        # 집기 모션은 freeze(막히면 미는 걸 멈추는 소프트웨어 과부하 예방)를 켜 둔다.
        # 모션은 ~수초간 닫힘을 밀므로 freeze 가 없으면 그 사이 과부하 래치 위험이 크다.
        # 실제 firm 그립은 아래 2단계 compliant_grip 이 짧게 담당한다(안전).
        print(f"[1] 집기 모션 재생: '{grab_motion}' (그립 유지, freeze 켬)")
        if not play_motion(grab_motion, 1, hold=True, grip_freeze=True):
            print("  집기 모션 실패 → 중단"); return False

        # 2단계 순응 물기: 짧게 세게 압착(torque)해 문 뒤 현재위치 유지(hold_torque)로
        # 낮춰 과부하 래치 없이 firm grip 을 확보한다. 판별 전에 실제로 꽉 물게 한다.
        if compliant:
            print("[1b] 2단계 순응 물기(6번): 압착→현재위치 유지")
            compliant_grip(port, close_pos=grip_close, torque_limit=grip_torque,
                           hold_torque=grip_hold_torque, squeeze_s=grip_squeeze_sec)

        # 판단 자세로 이동하되, 순응제어 중이면 6번(gripper)은 제외한다. 그래야 [1b]의
        # 순응 압착(완전히 닫힘, 상한 힘)이 판별·유지 내내 그대로 이어진다(판단자세의
        # 그리퍼 목표 2026 을 다시 쓰면 stall-hold 로 무는 힘이 풀린다).
        print(f"[2] 판단 자세로 이동: '{judge_pose}' (그립 유지"
              + (", 6번 제외)" if compliant else ")"))
        if not set_pose(judge_pose, hold=True, skip="gripper" if compliant else None):
            print("  자세 이동 실패 → 중단"); return False

        print("[3] 손목캠 촬영/판별")
        image = grab_frame(camera)
        if image is None:
            print("  촬영 실패 → 중단"); return False
        label, conf, scores = checker.predict(image)
        print(f"  판별: {label} (신뢰도 {conf:.0%})  raw={scores}")

        if label == success_class and conf >= success_conf:
            print(f"  ✅ 집기 성공(신뢰도 {conf:.0%}).")
            success = True
            break
        print(f"  ✗ 집기 미확인(label={label}, conf={conf:.0%}) → 재시도")

    if not success:
        print(f"\n⚠️ {max_attempts}회 시도했지만 집기를 확인하지 못했습니다. 사람이 확인/개입하세요.")
        return False

    # 안전(테이블 위 등): 주행 없이 집기+판별만 검증. 옷을 문 채 정지한다.
    if not drive:
        if compliant:
            print("\n[4] --no-drive: 2단계 순응 물기 재적용(6번)하고 옷을 문 채 정지.")
            compliant_grip(port, close_pos=grip_close, torque_limit=grip_torque,
                           hold_torque=grip_hold_torque, squeeze_s=grip_squeeze_sec)
        print("\n[4] --no-drive: 복귀 주행/던지기 생략. 옷을 문 채 판단 자세로 정지.")
        print("✅ 집기 → 판별 성공(주행 생략).")
        return True

    # 복귀 주행 '전'에 순응 물기 재적용: 판단 자세 이동 때 그리퍼 목표가 조금
    # 열리는 값(2026)이라 얇은 옷에서 그립이 느슨해질 수 있다. 다시 완전히 닫아
    # 긴 주행 동안 상한 힘으로 꽉 물게 한다.
    if compliant:
        print("\n[3b] 복귀 주행 전 2단계 순응 물기 재적용(6번)")
        compliant_grip(port, close_pos=grip_close, torque_limit=grip_torque,
                       hold_torque=grip_hold_torque, squeeze_s=grip_squeeze_sec)

    # ---- 복귀 주행 (팔은 laundry_default 로 옷을 문 채 유지) ----
    print("\n[4] 복귀 주행 (팔은 판단 자세로 옷을 문 채 유지)")
    print(f"  후진 {backup_sec}s → 180도 회전 → 직진 {forward_sec}s")
    drive_straight(port, -1, backup_sec, drive_speed)   # 후진
    rotate(port, rotate_sec, rotate_speed, clockwise=clockwise)  # 180도(시간 근사)
    drive_straight(port, +1, forward_sec, drive_speed)  # 직진

    # ---- 던지기 (모션이 있으면 재생, 없으면 여기서 유지한 채 종료) ----
    throw_path = MOTIONS_DIR / f"{throw_motion}.json"
    if throw_path.exists():
        print(f"\n[5] 던지기 모션 재생: '{throw_motion}'")
        # 던지는 순간 옷을 놓아야 하므로 hold 하지 않는다(모션 끝에 토크 해제).
        if not play_motion(throw_motion, 1, hold=keep_held):
            print("  던지기 모션 실패 → 중단(옷 유지)"); return False
        if not keep_held:
            release_arm(port)
        if retreat_sec > 0:
            print(f"\n[6] 그리퍼 열림 → {retreat_delay}s 대기 → 후진 {retreat_sec}s")
            time.sleep(retreat_delay)
            drive_straight(port, -1, retreat_sec, drive_speed)
        print("\n✅ 집기 → 복귀 → 던지기 완료.")
    else:
        print(f"\n[5] 던지기 모션 없음({throw_path.name}) → 복귀 지점에서 옷을 문 채 정지/유지.")
        print("    이동이 검증되면 던지기 모션을 녹화한 뒤 --throw-motion 으로 연결하세요.")
        print("\n✅ 집기 → 복귀 주행 완료(던지기 대기).")
    return True


def main():
    ap = argparse.ArgumentParser(description="집기→판별→복귀 주행(후진·180도·직진)→던지기.")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--camera", default="/dev/video0", help="grasp 판별용 손목캠(video0=wrist)")
    ap.add_argument("--grab-motion", default="laundry_grab")
    ap.add_argument("--judge-pose", default="laundry_default", help="판단(촬영) 자세")
    ap.add_argument("--grasp-probe", default=str(REPO / "grasp_probe.pt"))
    ap.add_argument("--success-class", default="grabbed")
    ap.add_argument("--success-conf", type=float, default=0.70)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--backup-sec", type=float, default=5.0, help="집은 뒤 후진 시간(초)")
    ap.add_argument("--rotate-sec", type=float, default=17.3,
                    help="180도 회전 시간(초). 실측: 시계방향 speed 300 에서 17.3초 ≈ 180도")
    ap.add_argument("--rotate-speed", type=int, default=300, help="회전 속도(실측 180도 기준 300)")
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
                    help="6번 순응제어(토크 제한 연속 닫기) 끄기(기존 고정 목표 방식)")
    ap.add_argument("--grip-torque", type=int, default=500,
                    help="2단계 순응: 1단계 압착 토크 상한(짧게 세게, ≤ Max_Torque_Limit=500)")
    ap.add_argument("--grip-hold-torque", type=int, default=300,
                    help="2단계 순응: 유지 토크 상한(현재위치 유지, 작을수록 과부하 안전)")
    ap.add_argument("--grip-squeeze-sec", type=float, default=6.0,
                    help="2단계 순응: 1단계 최고출력 압착 시간(초). 두꺼운 수건이 2150까지 "
                         "파고들 시간(기본 6.0). 길수록 더 깊이 물지만 발열↑")
    ap.add_argument("--grip-max-torque", type=int, default=None,
                    help="[EEPROM 영구변경] 6번 Max_Torque_Limit 을 이 값으로 올림(기본 안 건드림)")
    ap.add_argument("--grip-close", type=int, default=None,
                    help="순응 닫힘 목표(steps). 기본=calib range_min≈완전히 닫힘")
    ap.add_argument("--grip-pgain", type=int, default=32,
                    help="[EEPROM] 6번 P_Coefficient(위치 P게인). 기본 32(계측상 16→32에서 "
                         "그립이 캡까지 세짐). --grip-pgain 0 미만/None 개념은 미지원, 안 바꾸려면 16 지정")
    ap.add_argument("--no-relax-overload", action="store_true",
                    help="[EEPROM] 6번 과부하 문턱 완화를 끄기(기본 ON: Overload_Torque 25→80). "
                         "끄면 기본 보호라 수건 등 저항 있는 물체를 물 때 힘이 끊길 수 있음")
    ap.add_argument("--disable-overload", action="store_true",
                    help="[EEPROM 영구변경] 6번 과부하 차단(unload) 비트 제거. 기본 OFF(보통 불필요)")
    ap.add_argument("--overload-torque", type=int, default=80,
                    help="과부하 트리거 문턱 Overload_Torque(%%, 기본 25→80). 클수록 덜 걸림")
    ap.add_argument("--protective-torque", type=int, default=80,
                    help="과부하 후 남는 토크 Protective_Torque(%%, 기본 20→80). 클수록 그립 유지")
    ap.add_argument("--protection-current", type=int, default=500,
                    help="과전류 보호 문턱 Protection_Current(기본 250→500)")
    ap.add_argument("--yes", action="store_true", help="시작 확인 없이 바로 실행")
    args = ap.parse_args()

    print(f"[계획] 집기 → 판단({args.judge_pose}) → grabbed≥{args.success_conf:.0%}면 "
          f"후진 {args.backup_sec}s·180도 회전·직진 {args.forward_sec}s → 던지기.")
    if not args.yes:
        if input("시작하려면 Enter, 취소하려면 그 외 입력 후 Enter: ").strip() != "":
            print("취소되었습니다.")
            return 0

    try:
        ok = grab_and_return(
            port=args.port, camera=args.camera, grab_motion=args.grab_motion,
            judge_pose=args.judge_pose, grasp_probe=args.grasp_probe,
            success_class=args.success_class, success_conf=args.success_conf,
            max_attempts=args.max_attempts, backup_sec=args.backup_sec,
            rotate_sec=args.rotate_sec, rotate_speed=args.rotate_speed,
            forward_sec=args.forward_sec, drive_speed=args.drive_speed,
            clockwise=not args.ccw, throw_motion=args.throw_motion, keep_held=args.keep_held,
            retreat_delay=args.retreat_delay, retreat_sec=args.retreat_sec,
            drive=not args.no_drive,
            compliant=not args.no_compliant, grip_torque=args.grip_torque,
            grip_hold_torque=args.grip_hold_torque, grip_squeeze_sec=args.grip_squeeze_sec,
            grip_close=args.grip_close, grip_pgain=args.grip_pgain,
            grip_max_torque=args.grip_max_torque,
            relax_overload=not args.no_relax_overload,
            disable_overload=args.disable_overload,
            overload_torque=args.overload_torque, protective_torque=args.protective_torque,
            protection_current=args.protection_current,
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
