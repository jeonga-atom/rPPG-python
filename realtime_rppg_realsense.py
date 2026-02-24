#!/usr/bin/env python3
"""Real-time rPPG inference from Intel RealSense (or webcam fallback).
   <<<추론 - 체크포인트를 불러와서 실시간 실행하는 코드>>

Usage examples:
  python realtime_rppg_realsense.py --model tscan --checkpoint ./final_model_release/PURE_TSCAN.pth --use-realsense
  python realtime_rppg_realsense.py --model physnet --checkpoint ./final_model_release/PURE_PhysNet_DiffNormalized.pth --use-realsense
"""

import argparse
import glob
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import scipy.signal
import torch
from scipy.signal import butter
from scipy.sparse import spdiags

from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.TS_CAN import TSCAN


def _detrend(input_signal, lambda_value=100):
    signal_length = input_signal.shape[0]
    h_mat = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    d_mat = spdiags(diags_data, diags_index, (signal_length - 2), signal_length).toarray()
    return np.dot((h_mat - np.linalg.inv(h_mat + (lambda_value ** 2) * np.dot(d_mat.T, d_mat))), input_signal)


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

# 프레이간 변화량을 정규화된 차분으로 만드는 방식
def diff_normalize_data(data):
    n, h, w, c = data.shape
    diffnormalized_len = n - 1
    diffnormalized_data = np.zeros((diffnormalized_len, h, w, c), dtype=np.float32)
    diffnormalized_data_padding = np.zeros((1, h, w, c), dtype=np.float32)
    for j in range(diffnormalized_len):
        diffnormalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
            data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7
        ) # 조명,피부색 절대값 같은 느린변화를 줄이고/ 혈류로 인한 미세 변화를 강조하기 위한 목적
    diffnormalized_data = diffnormalized_data / np.std(diffnormalized_data)     # 전체 std로 나누어 스케일 통일
    diffnormalized_data = np.append(diffnormalized_data, diffnormalized_data_padding, axis=0) 
    diffnormalized_data[np.isnan(diffnormalized_data)] = 0
    return diffnormalized_data


def standardized_data(data):
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data

# freq, pxx = periodogram(sig, fs)
# mask: 0.6 ~ 3.3 Hz
# bpm = peak_freq * 60
def estimate_bpm_from_signal(ppg_signal, fs, diff_flag=True):
    # 최소 3초 이상 신호가 있어야 BPM 추정이 가능하도록 함 (짧은 신호는 잡음이 너무 많아서 정확한 추정이 어려움)
    if len(ppg_signal) < max(32, int(fs * 3)):
        return None, None
    sig = np.asarray(ppg_signal, dtype=np.float64)
    # 누적합(cumsum)으로 원래 파형 같은 형태로 복원하기 위헤/ _dtrenf로 저주파 드리프트 제거
    if diff_flag:
        sig = _detrend(np.cumsum(sig), 100)
    else:
        sig = _detrend(sig, 100)
    # 0.6Hz=36bpm, 3.3Hz=198bpm 범위의 밴드패스 필터링으로 심박이 나올 만한 대역의 신호만 남기기
    b, a = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype="bandpass")
    sig = scipy.signal.filtfilt(b, a, sig)
    nfft = _next_power_of_2(sig.shape[0])
    freq, pxx = scipy.signal.periodogram(np.expand_dims(sig, 0), fs=fs, nfft=nfft, detrend=False)
    fmask = np.argwhere((freq >= 0.6) & (freq <= 3.3))
    if fmask.size == 0:
        return None, sig
    mask_freq = np.take(freq, fmask)
    mask_pxx = np.take(pxx, fmask)
    bpm = float(np.take(mask_freq, np.argmax(mask_pxx, 0))[0] * 60.0)
    return bpm, sig


def load_checkpoint_flexible(model, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    try:
        model.load_state_dict(state, strict=True)
        return
    except RuntimeError:
        pass

    add_module = all(not k.startswith("module.") for k in state.keys())
    if add_module:
        state = {f"module.{k}": v for k, v in state.items()}
        try:
            model.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            pass

    stripped = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(stripped, strict=True)


def find_face_bbox(frame_bgr, detector, prev_bbox=None, scale=1.5):
    # scale=1.5로 박스를 조금 넓혀서(얼굴 주변까지) ROI로 씀
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        cx = x + w / 2.0
        cy = y + h / 2.0
        side = max(w, h) * scale
        x1 = int(max(0, cx - side / 2.0))
        y1 = int(max(0, cy - side / 2.0))
        x2 = int(min(frame_bgr.shape[1], cx + side / 2.0))
        y2 = int(min(frame_bgr.shape[0], cy + side / 2.0))
        return (x1, y1, x2, y2)
    return prev_bbox


def make_panel_background(width, height):
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[:] = (5, 5, 15)
    y_mid = height // 2
    cv2.line(bg, (0, y_mid), (width - 1, y_mid), (70, 80, 110), 1, cv2.LINE_AA)
    divisions = 10
    for px in range(0, width, max(1, width // divisions)):
        cv2.line(bg, (px, 0), (px, height - 1), (16, 20, 38), 1, cv2.LINE_AA)
        cv2.line(bg, (px, y_mid - 2), (px, y_mid + 2), (20, 30, 48), 2, cv2.LINE_AA)
    return bg


def update_ekg_trace(canvas, background, value, state, stats):
    width = stats["width"]
    height = stats["height"]
    x_float = state["pos"]
    x_idx = int(x_float) % width
    clear_len = stats.get("clear_pixels", 5)
    for dx in range(1, clear_len + 1):
        canvas[:, (x_idx + dx) % width] = background[:, (x_idx + dx) % width]
    canvas[:, x_idx] = background[:, x_idx]

    mean = stats["mean"]
    scale = stats["scale"]
    vert_range = stats["vert_range"]
    y_mid = stats["y_mid"]
    normed = np.clip((value - mean) / scale, -1.0, 1.0)
    y = int(np.clip(y_mid - normed * vert_range, 0, height - 1))

    next_pos = x_float + stats["step"]
    wrapped = next_pos >= width
    next_pos_mod = next_pos % width
    draw_line = state["prev_point"] is not None and not wrapped
    color = (0, 0, 255)

    if draw_line:
        cv2.line(canvas, state["prev_point"], (x_idx, y), color, 2, cv2.LINE_AA)
    else:
        cv2.circle(canvas, (x_idx, y), 1, color, -1, cv2.LINE_AA)

    state["prev_point"] = None if wrapped else (x_idx, y)
    state["pos"] = next_pos_mod
    state["total_drawn"] += 1


def draw_right_panel(panel, canvas, last_bpm, disp_fps, warning_text):
    panel[:] = canvas
    h, w = panel.shape[:2]
    bpm_text = f"BPM: {last_bpm:.1f}" if last_bpm is not None else "BPM: --"
    fps_text = f"FPS: {disp_fps:.1f}"
    cv2.putText(panel, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, bpm_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    if warning_text:
        cv2.putText(panel, warning_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1, cv2.LINE_AA)


def next_log_path(base_dir=".", base_name="bpm_log", ext="txt"):
    os.makedirs(base_dir, exist_ok=True)
    pattern = os.path.join(base_dir, f"{base_name}_*.{ext}")
    highest = 0
    for entry in glob.glob(pattern):
        name = os.path.splitext(os.path.basename(entry))[0]
        if "_" not in name:
            continue
        suffix = name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    candidate = f"{base_name}_{highest + 1}.{ext}"
    return os.path.join(base_dir, candidate)


class FrameSource:
    def __init__(self, use_realsense, width, height, fps, camera_index):
        self.use_realsense = use_realsense
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.pipeline = None
        self.cap = None
        self.rs = None

    def open(self):
        if self.use_realsense:
            try:
                import pyrealsense2 as rs

                self.rs = rs
                self.pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self.pipeline.start(cfg)
                return
            except Exception as exc:
                print(f"[WARN] RealSense open failed ({exc}). Falling back to cv2 camera.")

        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera stream.")

    def read(self):
        if self.pipeline is not None:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            img = np.asanyarray(color_frame.get_data())
            return True, img
        ok, img = self.cap.read()
        return ok, img

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
        if self.cap is not None:
            self.cap.release()


def make_model(args, device):
    if args.model == "tscan":
        model = TSCAN(frame_depth=args.tscan_frame_depth, img_size=args.input_size)
        if device.type == "cuda":
            model = torch.nn.DataParallel(model, device_ids=[device.index if device.index is not None else 0])
        model = model.to(device).eval()
        load_checkpoint_flexible(model, args.checkpoint, device)
        return model

    model = PhysNet_padding_Encoder_Decoder_MAX(frames=args.physnet_frame_num).to(device).eval()
    load_checkpoint_flexible(model, args.checkpoint, device)
    return model


def infer_signal(model, model_name, clip_rgb, device, args):
    clip_rgb = clip_rgb.astype(np.float32)
    # TSCAN은 motion branch(차분) + appearance branch(원본/정규화)같이 두 종류 특징을 같이 쓰는 방식
    if model_name == "tscan":
        diff = diff_normalize_data(clip_rgb)
        raw = standardized_data(clip_rgb)
        data = np.concatenate([diff, raw], axis=-1)
        data = np.transpose(data, (0, 3, 1, 2))
        if data.shape[0] % args.tscan_frame_depth != 0: # frame_depth 단위로 처리하기 위해 길이를 맞추는 방식
            valid = (data.shape[0] // args.tscan_frame_depth) * args.tscan_frame_depth
            data = data[:valid]
        x = torch.from_numpy(data).to(device, dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).squeeze(-1).detach().cpu().numpy()
        return pred

    # physNet은 차분 정규화된 데이터만 쓰는 방식/ 배치,채널,높이,너비 형태의 3D conv스타일 입력을 받는 경우가 많아서 이렇게 진행
    data = diff_normalize_data(clip_rgb)
    data = np.transpose(data, (3, 0, 1, 2))
    x = torch.from_numpy(data).unsqueeze(0).to(device, dtype=torch.float32)
    with torch.no_grad():
        pred, _, _, _ = model(x)
    return pred.squeeze(0).detach().cpu().numpy()


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time rPPG with RealSense/Webcam using TSCAN or PhysNet.")
    parser.add_argument("--model", choices=["tscan", "physnet"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-realsense", action="store_true")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--input-size", type=int, default=72)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--infer-stride", type=int, default=15)
    parser.add_argument("--diff-flag", dest="diff_flag", action="store_true")
    parser.add_argument("--no-diff-flag", dest="diff_flag", action="store_false")
    parser.set_defaults(diff_flag=True)
    parser.add_argument("--tscan-frame-depth", type=int, default=10)
    parser.add_argument("--tscan-clip-len", type=int, default=180)      # TSCAN은 기본 프레임 180
    parser.add_argument("--physnet-frame-num", type=int, default=64)   # PhysNet은 기본 프레임 128
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("[WARN] CUDA is not available. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    clip_len = args.tscan_clip_len if args.model == "tscan" else args.physnet_frame_num
    frame_buffer = deque(maxlen=clip_len)
    bpm_signal_buffer = deque(maxlen=max(int(args.window_seconds * args.fps), clip_len))
    ppg_wave_buffer = deque(maxlen=max(300, args.cam_width))  # 파형 출력용
    PANEL_W = args.cam_width
    PANEL_H = args.cam_height
    ekg_background = make_panel_background(PANEL_W, PANEL_H)
    ekg_canvas = ekg_background.copy()
    ekg_state = {"pos": 0.0, "prev_point": None, "total_drawn": 0}

    cascade_path = "./dataset/haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise RuntimeError(f"Failed to load Haar cascade from: {cascade_path}")

    model = make_model(args, device)
    source = FrameSource(args.use_realsense, args.cam_width, args.cam_height, args.fps, args.camera_index)
    source.open()

    last_bbox = None
    frame_idx = 0
    last_bpm = None
    kf_x = None     # 현재 추정 BPM
    kf_p = 1.0      # 추정 오차 공분산
    kf_q = 0.3      # (BPM이 얼마나 빨리 변할 수 있는가)크면 측정값 변화를 빠르게 받아들이므로 응답성이 좋아지고, 노이즈를 더 받음/ 작으면 더 부드럽지만 반응이 둔해짐
    kf_r = 10.0     # (추론 값을 얼마나 신뢰하는가)크면 측정값을 덜 믿고 이전 추정치를 유지하려고 하므로 안정적이지만 변화에 덜 민감함
    warning_text = ""
    t0 = time.time()

    log_dir = os.path.join(os.getcwd(), "log")
    log_path = next_log_path(log_dir)
    log_file = open(log_path, "w")
    log_file.write("time(s),bpm\n")
    print(f"Logging BPM to {log_path}")

    t0 = time.time()
    start_time = time.time()

    print("Press 'q' to quit.")
    try:
        while True:
            ok, frame_bgr = source.read()
            if not ok or frame_bgr is None:
                continue

            # 얼굴 탐색은 15프레임마다만
            if frame_idx % 15 == 0:
                last_bbox = find_face_bbox(frame_bgr, face_detector, prev_bbox=last_bbox, scale=1.5)

            # ROI 처리는 매 프레임, 얼굴만 잘라서 72x72로 리사이즈해서 영상을 버퍼에 저장
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                roi_bgr = frame_bgr[y1:y2, x1:x2]
                if roi_bgr.size > 0:
                    roi_bgr = cv2.resize(roi_bgr, (args.input_size, args.input_size), interpolation=cv2.INTER_AREA)
                    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
                    frame_buffer.append(roi_rgb.astype(np.float32))
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(frame_buffer) == clip_len and (frame_idx % args.infer_stride == 0):
                clip = np.stack(frame_buffer, axis=0)
                pred_sig = infer_signal(model, args.model, clip, device, args)
                tail = min(args.infer_stride, len(pred_sig))
                for v in pred_sig[-tail:]:
                    bpm_signal_buffer.append(float(v))
                # 딥러닝이 바로 BPM내는 것이 아닌/ 딥러닝->rPPG 신호(시간파형)->신호처리로 BPM흐름
                bpm, filtered_sig = estimate_bpm_from_signal(list(bpm_signal_buffer), args.fps, diff_flag=args.diff_flag)
                new_samples = []
                if filtered_sig is not None and tail > 0:
                    start = max(0, len(filtered_sig) - tail)
                    new_samples = [float(v) for v in filtered_sig[start:]]
                    for v in new_samples:
                        ppg_wave_buffer.append(v)
                    if new_samples:
                        arr = np.asarray(ppg_wave_buffer, dtype=np.float32)
                        mean = np.mean(arr)
                        max_dev = max(np.max(arr - mean), np.max(mean - arr))
                        scale = max(max_dev, 1e-3)
                        frame_rate = args.fps if args.fps > 0 else 30
                        bpm_value = last_bpm if (last_bpm is not None and last_bpm > 0) else 72.0
                        samples_per_cycle = max(2, int(round(frame_rate * 60.0 / bpm_value)))
                        # 한 화면에 몇 주기가 보이게 할지 step을 조정 (현재는 5주기)
                        visible_samples = max(samples_per_cycle * 7, 1)
                        # BPM이 더 빨라지면 더 촘촘히, 느리면 더 넓게 보이도록
                        step = PANEL_W / visible_samples
                        clear_pixels = max(3, int(PANEL_W * 0.03))
                        stats = {
                            "mean": mean,
                            "scale": scale,
                            "vert_range": max(1, (PANEL_H // 2) - 12),
                            "y_mid": PANEL_H // 2,
                            "width": PANEL_W,
                            "height": PANEL_H,
                            "step": step,
                            "clear_pixels": clear_pixels,
                        }
                        for v in new_samples:
                            update_ekg_trace(ekg_canvas, ekg_background, v, ekg_state, stats)
                if bpm is not None:
                    if kf_x is None:
                        kf_x = bpm
                    else:
                        kf_p = kf_p + kf_q
                        kf_k = kf_p / (kf_p + kf_r)
                        kf_x = kf_x + kf_k * (bpm - kf_x)
                        kf_p = (1 - kf_k) * kf_p
                    elapsed = time.time() - start_time
                    last_bpm = kf_x # 최종 표시 로그는 kf_x(칼만 필터로 보정된 값)
                    log_file.write(f"{elapsed:.2f},{last_bpm:.1f}\n")
                    log_file.flush()
                    # 알람이 뜨는 BPM의 범위를 설정 (현재는 50미만 또는 130초과일 때) / BPM이 너무 낮거나 높으면 심장 이상 신호일 수 있어서 경고문구 띄우도록
                    warning_text = "Abnormal Heart Rate Detection. Caution!" if last_bpm < 50 or last_bpm > 130 else ""

            disp_fps = 1.0 / max(time.time() - t0, 1e-6)
            t0 = time.time()
            if warning_text:
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                center_x = (frame_bgr.shape[1] - text_size[0]) // 2
                y_base = frame_bgr.shape[0] - 40
                cv2.putText(frame_bgr, warning_text, (center_x, y_base), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            right_panel = ekg_canvas.copy()
            draw_right_panel(right_panel, ekg_canvas, last_bpm, disp_fps, warning_text)
            combined = np.hstack([frame_bgr, right_panel])
            cv2.imshow("Real-time rPPG", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            frame_idx += 1
    finally:
        source.close()
        log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
