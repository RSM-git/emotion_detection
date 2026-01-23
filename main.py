from emotion_detection.data import download_dataset, parse_grayscale_png
from emotion_detection.cli import start_up, select_model, setup_hparams
from emotion_detection.utils import load_model, preprocess_frame
from pathlib import Path
from train import Trainer

import torch
import cv2
import sys

def inference_loop(model_path: str | Path) -> None:
    model = load_model(model_path)

    capture = cv2.VideoCapture(0)

    if not cap.isOpened():
        exit()

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Cannot receive frame from capture...", file=sys.stderr)
            break

        frame = torch.from_numpy(frame)
        # preprocess the frame for the model
        frame = preprocess_frame(frame)
        model_prediction = model(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()


def main():
    task_map = {"Train model": 0, "Inference": 1}

    task = start_up()

    match task_map[task]:
        case 0:
            train_hparams = setup_hparams()
            trainer = Trainer()
            trainer.train(*train_hparams)

        case 1:
            model_path = select_model()
            inference_loop()

if __name__ == "__main__":
    main()

