# Lerobot Sanhak VLA — 세탁물 회수 태스크 v3 (`laundry_task3`)

LeKiwi(모바일 매니퓰레이터)로 **세탁기에서 빨래를 꺼내 바구니에 던지는** 전체 파이프라인.
집기(grasp) 단계는 **SmolVLA(VLA)** 로 수행하고, "정말 잡았는지"는 **CLIP 선형 probe(VLM)** 로 판별한다.

> 이 저장소 하나에 태스크 실행에 필요한 **코드·포즈·모션·probe 가중치**가 전부 들어 있다.
> 무거운 SmolVLA 가중치(≈865MB)만 용량 때문에 제외했고, HF 에서 받는다(아래 [3. VLA 가중치](#3-vla-가중치-내려받기) 참고).

---

## 1. 구조 — Jetson + 랩탑 하이브리드

집기 단계에만 랩탑 GPU 가 필요하므로 **두 대의 머신**이 협력한다.

```
┌──────────────────────────── Jetson (로봇 호스트) ────────────────────────────┐
│  scripts/skills/laundry_task3.py   ← 오케스트레이터(진입점)                    │
│    [1] 접근(approach)   [2] 문 열기(open_door2)                                │
│    [3] 집기(grab_place_vla) ──SSH──▶  랩탑 grab_vla_headless.py 호출           │
│    [4] 복귀 주행(return)  [5] 던지기(throw)                                     │
│  ※ [1][2][4][5] 는 모터버스 직접 제어(set_pose / play_motion / move_to_dest)   │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │  ZMQ(lekiwi_host) + SSH
                                  ▼
┌──────────────────────────── 랩탑 (GPU) ──────────────────────────────────────┐
│  examples/lekiwi/grab_vla_headless.py                                          │
│    · SmolVLA(VLA) 로 뻗기→집기                                                  │
│    · 상공/손목 CLIP probe(VLM, grasp_clip.py) 로 grabbed 2단계 게이트           │
│    · 확정되면 그리퍼 압착 후 종료코드 0 반환 → Jetson 이 다음 단계 진행         │
└──────────────────────────────────────────────────────────────────────────────┘
```

집기 핸드오프(`grab_place_vla.run_vla_grasp`) 흐름: Jetson 이 준비자세로 이동 → `lekiwi_host` 기동 →
랩탑으로 SSH 해 `grab_vla_headless.py` 실행(랩탑이 VLA 집기 + VLM 확정 + 압착) → host kill(토크 유지) →
Jetson 이 팔에 빨래를 문 채 복귀 주행 → 던지기.

---

## 2. 저장소 구성

| 경로 | 역할 | 실행 머신 |
|------|------|-----------|
| `scripts/skills/laundry_task3.py` | **진입점.** 접근→문열기→집기→복귀→던지기 오케스트레이터 | Jetson |
| `scripts/skills/laundry/` | 스테이지 모듈: `approach` · `open_door2` · `return_home` · `grab_place` · `grab_place_vla`(VLA 집기) · `common`(공용 유틸) · `move_to_destination` | Jetson |
| `scripts/arm/set_pose.py`, `play_motion.py` | 포즈/모션을 모터버스로 재생하는 subprocess 헬퍼 | Jetson |
| `washer_clip_check.py` | **VLM** `WasherStatusChecker`(CLIP + probe grasp 판별) | Jetson |
| `examples/lekiwi/grab_vla_headless.py` | **VLA** 헤드리스 집기(SmolVLA + 2단계 게이트). SSH 로 호출됨 | 랩탑 |
| `examples/lekiwi/grasp_clip.py` | **VLM** CLIP probe 게이트(`GraspGate`) | 랩탑 |
| `examples/lekiwi/lekiwi_pose.py` | 저장 포즈(raw 스텝)→정규화 액션 변환 후 재생 | 랩탑 |
| `examples/lekiwi/lekiwi_calibration.json` | LeKiwi 팔 캘리브레이션(정규화에 필요) | 랩탑 |
| `poses/*.json` | 저장 포즈(원시 모터 스텝) | 공용 |
| `motions/*.json` | 저장 모션 시퀀스(문열기·집기·던지기 등) | 공용 |
| `grasp_probe*.pt` | CLIP 선형 probe 가중치(손목·상공·기본) | 공용 |
| `red_approach.json` | 접근 파라미터 | Jetson |

---

## 3. 설치

### 3.1 LeRobot 라이브러리 (두 머신 모두)

이 저장소는 **태스크 코드**만 담고 있고, `lerobot` 라이브러리 본체는 별도로 설치해야 한다.
Python 3.12 환경(예: conda env `lerobot`)에서:

```bash
# 공식 저장소: https://github.com/huggingface/lerobot
pip install "lerobot[smolvla,feetech]"
# 또는 소스 설치:
#   git clone https://github.com/huggingface/lerobot && cd lerobot
#   pip install -e ".[smolvla,feetech]"
```

> 학습 때 쓰던 것과 **같은 lerobot 버전**을 권장한다(정책 로딩/프로세서 호환).

### 3.2 이 저장소 클론 (두 머신 모두, `~/lerobot` 로)

코드가 `~/lerobot` 경로와 `~/lerobot/models/...` 체크포인트 경로를 기본값으로 쓴다.
그대로 쓰려면 **홈 디렉토리에 `lerobot` 이름으로** 클론한다.

```bash
git clone https://github.com/LSJ-0829/Lerobot_Sanhak_VLA.git ~/lerobot
cd ~/lerobot
```

> 다른 경로에 두려면 `LEKIWI_CHECKPOINT`, `LAPTOP_SSH`, `HEADLESS_GRAB_CMD` 등 env 로 덮어쓰면 된다(아래 표).

### 3.3 카메라 캘리브레이션

`examples/lekiwi/lekiwi_calibration.json` 은 예시 캘리브레이션이다. **본인 로봇 값**으로 교체:

```bash
scp jetson:~/.cache/huggingface/lerobot/calibration/robots/lekiwi/my_awesome_kiwi.json \
    ~/lerobot/examples/lekiwi/lekiwi_calibration.json
```

---

## 4. VLA 가중치 내려받기 (랩탑)

SmolVLA 미세조정 체크포인트는 용량 때문에 저장소에서 제외했다. HF 에서 받는다.

- **미세조정 정책(이 태스크용)**: [`HyeonseokE/smolvla_lekiwi_spin_cycle`](https://huggingface.co/HyeonseokE/smolvla_lekiwi_spin_cycle)
- 베이스 모델: [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base) · 비전 백본: [`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)

```bash
# 코드가 기대하는 기본 경로에 그대로 받는다.
hf download HyeonseokE/smolvla_lekiwi_spin_cycle \
    --local-dir ~/lerobot/models/smolvla_lekiwi_spin_cycle
# (구버전 CLI 는:  huggingface-cli download ... )
```

> 다른 위치에 두면 `export LEKIWI_CHECKPOINT=/경로/to/checkpoint` 로 지정.

---

## 5. 네트워크 준비

| 항목 | 기본값 | 설명 |
|------|--------|------|
| Jetson ↔ 랩탑 | 유선 `192.168.55.x` | 집기 단계 SSH·ZMQ 통신 |
| `LAPTOP_SSH` | `andy2@192.168.55.100` | Jetson→랩탑 SSH 대상(**무비밀번호 키 로그인** 필요) |
| `REMOTE_IP`(랩탑) | 유선 `192.168.55.1` / 무선 `192.168.0.19` | 랩탑→Jetson `lekiwi_host` ZMQ 주소 |
| 상공 카메라 `OVERHEAD_CAM` | `/dev/video32` | 랩탑에 연결된 세탁기 상공 USB 캠 |

1. Jetson→랩탑 **무비밀번호 SSH** 설정: `ssh-copy-id andy2@192.168.55.100`
2. 랩탑에 상공 USB 카메라 연결(`/dev/video32`).
3. `lekiwi_host` 는 집기 단계에서 Jetson 이 **자동으로** 띄웠다가 내린다(수동 기동 불필요).

---

## 6. 실행

**Jetson 에서** 진입점 하나만 실행하면 전체 파이프라인이 돈다:

```bash
cd ~/lerobot
python scripts/skills/laundry_task3.py            # 접근→문열기→VLA 집기→복귀→던지기
python scripts/skills/laundry_task3.py --yes      # 시작 확인 없이 바로
python scripts/skills/laundry_task3.py --record    # 프레임 녹화
```

자주 쓰는 인자(전체는 `--help`):

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--port` | `/dev/ttyACM0` | 모터버스 포트 |
| `--open-motion` | `openseasame` | 문 열기 모션(`motions/<이름>.json`) |
| `--rotate-sec` | `16.9` | 복귀 시 180° 회전 시간(실기 재보정) |
| `--throw-motion` | `laundry_throw` | 던지기 모션 |
| `--max-attempts` | `3` | 집기 재시도 횟수 |
| `--record` | off | front/wrist/상공 프레임 녹화 |

집기 단계 세부 튜닝은 **랩탑 env** 로(예): `GRASP_THRESHOLD`(상공 게이트 0.55), `WRIST_THRESHOLD`(손목 확정 0.5),
`GRIP_SQUEEZE_SEC`(압착 1.2s), `GRAB_TIMEOUT`(40s), `READY_POSE`(`laundry_grabready`). 필요 시
`grab_place_vla.py` 의 `HEADLESS_GRAB_CMD` 에 넣어 넘긴다.

### 랩탑 집기 단독 테스트

핸드오프 없이 VLA 집기만 확인하려면 랩탑에서 직접:

```bash
cd ~/lerobot
python examples/lekiwi/grab_vla_headless.py       # 준비자세→VLA 집기→2단계 게이트→압착
```

---

## 7. 동작 원리 요약

- **VLA(집기)** — `grab_vla_headless.py` 가 SmolVLA 로 매 프레임 액션을 예측해 뻗기·집기를 수행.
  상공 CLIP probe 가 `grabbed` 를 연속 N회 확정하면 자동 정지.
- **VLM(판별)** — `grasp_clip.py`/`washer_clip_check.py` 의 CLIP(ViT-B-32) feature 위에 로지스틱 회귀 probe
  (`grasp_probe*.pt`)를 얹어 grasp 성공/실패를 판별. 상공(1차)·손목(2차) 2단계 게이트로 오탐을 거른다.
- **핸드오프** — Jetson 은 규칙 기반 주행/문열기/던지기를 직접버스로, GPU 가 필요한 집기만 랩탑에 위임.

---

## 8. 트러블슈팅

| 증상 | 확인 |
|------|------|
| `probe 를 찾을 수 없습니다` | `grasp_probe_overhead.pt` / `grasp_probe_wrist.pt` 가 `~/lerobot` 루트에 있는지 |
| `from_pretrained` 실패 | VLA 가중치를 `~/lerobot/models/smolvla_lekiwi_spin_cycle` 에 받았는지([4](#4-vla-가중치-내려받기-랩탑)) |
| 집기 단계에서 멈춤 | Jetson→랩탑 무비밀번호 SSH(`LAPTOP_SSH`), 랩탑 상공 카메라(`OVERHEAD_CAM`) 연결 |
| 캘리브레이션 오류 | `lekiwi_calibration.json` 을 본인 로봇 값으로 교체([3.3](#33-카메라-캘리브레이션)) |
| 회전이 180°가 아님 | `--rotate-sec` 실기 재보정 |

---

원본: [huggingface/lerobot](https://github.com/huggingface/lerobot) 기반. 이 저장소는 LeKiwi 세탁물 태스크 v3 번들.
