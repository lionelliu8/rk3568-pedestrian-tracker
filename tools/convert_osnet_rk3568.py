#!/usr/bin/env python3
import argparse
import importlib.util
from pathlib import Path

import cv2
import torch
from rknn.api import RKNN


def load_osnet_module(source: Path):
    spec = importlib.util.spec_from_file_location("torchreid_osnet", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def export_onnx(source_root: Path, checkpoint: Path, output: Path):
    osnet = load_osnet_module(source_root / "torchreid/models/osnet.py")
    model = osnet.osnet_x0_25(num_classes=751, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    dummy = torch.zeros(1, 3, 256, 128)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["images"],
        output_names=["features"],
        opset_version=12,
        do_constant_folding=True,
    )


def make_calibration(video: Path, directory: Path, dataset_file: Path):
    directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    images = []
    for index in range(32):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index * frame_count // 32)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        crop_width = max(2, width // 3)
        x = (width - crop_width) // 2
        crop = frame[:, x : x + crop_width]
        path = directory / f"calibration_{index:02d}.jpg"
        cv2.imwrite(str(path), crop)
        images.append(path)
    capture.release()
    if not images:
        raise RuntimeError(f"could not extract calibration images from {video}")
    dataset_file.write_text("\n".join(str(path) for path in images) + "\n", encoding="ascii")


def convert(onnx_path: Path, dataset: Path, output: Path):
    rknn = RKNN(verbose=True)
    ret = rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],
        std_values=[[58.395, 57.12, 57.375]],
        target_platform="rk3568",
        optimization_level=3,
    )
    if ret != 0:
        raise RuntimeError(f"RKNN config failed: {ret}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        raise RuntimeError(f"RKNN load_onnx failed: {ret}")
    ret = rknn.build(do_quantization=True, dataset=str(dataset))
    if ret != 0:
        raise RuntimeError(f"RKNN build failed: {ret}")
    ret = rknn.export_rknn(str(output))
    rknn.release()
    if ret != 0:
        raise RuntimeError(f"RKNN export failed: {ret}")


def main():
    parser = argparse.ArgumentParser(description="Build an RK3568 OSNet x0.25 ReID model")
    parser.add_argument("--torchreid", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.work_dir / "osnet_x0_25_market.onnx"
    dataset = args.work_dir / "dataset.txt"
    export_onnx(args.torchreid, args.checkpoint, onnx_path)
    make_calibration(args.video, args.work_dir / "calibration", dataset)
    convert(onnx_path, dataset, args.output)
    print(f"RK3568 model written to {args.output}")


if __name__ == "__main__":
    main()
