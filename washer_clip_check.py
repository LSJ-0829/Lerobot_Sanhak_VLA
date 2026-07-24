import argparse
import os
import time

import torch
from PIL import Image
import open_clip

DEFAULT_PROBE = "/home/comnet02/lerobot/washer_probe.pt"
DEFAULT_SIDE_PROBE = "/home/comnet02/lerobot/washer_side_probe.pt"


def _load_probe(path, device):
    """probe 체크포인트를 dict 로 로드. 없으면 None."""
    if not (path and os.path.exists(path)):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    return {
        "W": ckpt["W"].to(device),
        "b": ckpt["b"].to(device),
        "classes": ckpt["classes"],
        "cv_acc": ckpt.get("cv_acc"),
        "model_name": ckpt.get("model_name"),
        "pretrained": ckpt.get("pretrained"),
    }


class WasherStatusChecker:
    """세탁기 내부 empty/has_clothes 판별기 (+ 옷이 있으면 좌/우 위치 판별).

    - 1단계 probe(washer_probe.pt): empty vs clothes. 검증된 신뢰도. 반드시 있어야 함.
    - 2단계 side probe(washer_side_probe.pt): 옷이 '있을 때만' 좌/우(/both) 판별.
      없으면 side=None 을 돌려주고, 호출측(clear_washer.py)이 좌·우 둘 다 집는
      blind sweep 으로 폴백한다. 1단계는 절대 건드리지 않는다.
    없으면(1단계도 없으면) zero-shot 프롬프트로 폴백한다.
    """

    def __init__(self, model_name="ViT-B-32", pretrained="/home/comnet02/lerobot/open_clip_model.safetensors",
                 device=None, probe_path=DEFAULT_PROBE, side_probe_path=DEFAULT_SIDE_PROBE):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 두 probe 를 로드. 1단계 probe 안의 model_name/pretrained 를 우선 사용(학습과 일치).
        self.probe = _load_probe(probe_path, self.device)
        self.side_probe = _load_probe(side_probe_path, self.device)
        if self.probe is not None:
            model_name = self.probe["model_name"] or model_name
            pretrained = self.probe["pretrained"] or pretrained

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.empty_prompts = [
            "an empty washing machine drum, no clothes inside",
            "the inside of an empty washer, nothing inside",
            "a photo of an empty washing machine tub",
        ]
        self.full_prompts = [
            "a washing machine drum full of clothes",
            "clothes and laundry inside a washing machine",
            "a photo of a washer tub filled with fabric",
        ]

        all_prompts = self.empty_prompts + self.full_prompts
        with torch.no_grad():
            text_tokens = self.tokenizer(all_prompts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self.text_features = text_features
        self.n_empty = len(self.empty_prompts)

    @torch.no_grad()
    def _image_features(self, image: Image.Image):
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(img_tensor)
        feats /= feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def _check_side(self, image_features):
        """옷 좌/우(/both) 판별. (side, 신뢰도, 점수dict) 반환."""
        logits = image_features @ self.side_probe["W"].t() + self.side_probe["b"]
        proba = logits.softmax(dim=-1)[0]
        classes = self.side_probe["classes"]
        top = int(proba.argmax())
        scores = {c: float(proba[i]) for i, c in enumerate(classes)}
        return classes[top], float(proba[top]), scores

    @torch.no_grad()
    def predict(self, image: Image.Image):
        """probe 의 클래스를 재매핑 없이 top-1 그대로 반환. (class_name, conf, scores).

        check() 는 empty/has_clothes 로 재매핑하는 세탁기 전용 로직이라
        grabbed/failed 같은 임의 2-클래스 probe(grasp 판별 등)에는 맞지 않는다.
        이 메서드는 probe.classes 를 그대로 돌려주므로 범용으로 쓸 수 있다.
        probe 가 없으면(None) 사용할 수 없다(zero-shot 폴백 없음).
        """
        if self.probe is None:
            raise RuntimeError("predict() 에는 학습된 probe 가 필요합니다(zero-shot 미지원).")
        feats = self._image_features(image)
        logits = feats @ self.probe["W"].t() + self.probe["b"]
        proba = logits.softmax(dim=-1)[0]
        classes = self.probe["classes"]
        top = int(proba.argmax())
        scores = {c: float(proba[i]) for i, c in enumerate(classes)}
        return classes[top], float(proba[top]), scores

    @torch.no_grad()
    def check(self, image: Image.Image):
        image_features = self._image_features(image)

        if self.probe is not None:
            logits = image_features @ self.probe["W"].t() + self.probe["b"]
            proba = logits.softmax(dim=-1)[0]
            classes = self.probe["classes"]
            top = int(proba.argmax())
            status = "empty" if classes[top] == "empty" else "has_clothes"
            confidence = float(proba[top])
            scores = {c: float(proba[i]) for i, c in enumerate(classes)}
            scores["source"] = "probe"

            # 2단계: 옷이 '있을 때만' 좌/우 판별(있으면). 1단계는 그대로 둔다.
            if status == "has_clothes" and self.side_probe is not None:
                side, side_conf, side_scores = self._check_side(image_features)
                scores["side"] = side
                scores["side_conf"] = side_conf
                scores["side_scores"] = side_scores

            return status, confidence, scores

        # 폴백: zero-shot 프롬프트
        similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        scores = similarity[0].tolist()

        empty_score = sum(scores[: self.n_empty])
        full_score = sum(scores[self.n_empty :])

        status = "empty" if empty_score > full_score else "has_clothes"
        confidence = max(empty_score, full_score)

        return status, confidence, {"empty_score": empty_score, "full_score": full_score, "source": "zeroshot"}


def move_to_inspect_pose(robot=None):
    pass


def capture_and_check(checker, cam_device, settle_sec: float = 0.5, warmup: int = 15):
    """raw cv2.VideoCapture 로 프레임을 잡아 판별한다.

    lerobot OpenCVCamera 는 이 카메라에서 width/height 검증에 실패하므로
    다른 vision 스크립트들과 동일하게 raw cv2 를 사용한다(BGR→RGB 변환).
    """
    import cv2

    cam = int(cam_device) if str(cam_device).isdigit() else cam_device
    # V4L2 백엔드 + MJPG 를 강제해야 640x480 이 나온다. 안 하면 YUYV 160x120 으로 떨어진다.
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"카메라를 열 수 없습니다: {cam_device}")
        return None, 0.0
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        for _ in range(warmup):  # 자동노출/화이트밸런스 안정화
            cap.read()
            time.sleep(0.03)
        time.sleep(settle_sec)
        ret, frame = cap.read()
    finally:
        cap.release()

    if not ret or frame is None:
        print("프레임을 읽지 못했습니다.")
        return None, 0.0

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status, confidence, scores = checker.check(image)
    label = "옷 없음 (empty)" if status == "empty" else "옷 있음 (has_clothes)"
    side = scores.get("side")
    side_str = f" | 위치: {side}({scores.get('side_conf', 0):.0%})" if side else ""
    print(f"결과: {label} | 신뢰도: {confidence:.2%}{side_str} | raw: {scores}")
    return status, confidence


def run_on_image(checker, image_path):
    image = Image.open(image_path).convert("RGB")
    status, confidence, scores = checker.check(image)
    label = "옷 없음 (empty)" if status == "empty" else "옷 있음 (has_clothes)"
    side = scores.get("side")
    side_str = f" | 위치: {side}({scores.get('side_conf', 0):.0%})" if side else ""
    print(f"결과: {label} | 신뢰도: {confidence:.2%}{side_str} | raw: {scores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--camera-index", type=str, default="/dev/video0")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--probe", type=str, default=DEFAULT_PROBE,
                        help="학습된 1단계 CLIP probe 경로. 'none'이면 zero-shot 강제.")
    parser.add_argument("--side-probe", type=str, default=DEFAULT_SIDE_PROBE,
                        help="2단계 좌/우 probe 경로. 'none'이면 사용 안 함.")
    args = parser.parse_args()

    probe_path = None if args.probe.lower() == "none" else args.probe
    side_probe_path = None if args.side_probe.lower() == "none" else args.side_probe
    checker = WasherStatusChecker(probe_path=probe_path, side_probe_path=side_probe_path)
    if checker.probe is not None:
        acc = checker.probe.get("cv_acc")
        acc_str = f", cv정확도 {acc:.1%}" if acc is not None else ""
        side_str = f", side={checker.side_probe['classes']}" if checker.side_probe else ""
        print(f"모델 로드 완료 (device={checker.device}, mode=probe{acc_str}{side_str})")
    else:
        print(f"모델 로드 완료 (device={checker.device}, mode=zero-shot 프롬프트)")

    if args.image:
        run_on_image(checker, args.image)

    elif args.once:
        move_to_inspect_pose(robot=None)
        capture_and_check(checker, args.camera_index, settle_sec=0.5)

    else:
        print("--image 또는 --once 옵션을 지정해주세요.")
