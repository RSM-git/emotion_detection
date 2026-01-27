from emotion_detection.data import download_dataset, parse_grayscale_png
from emotion_detection.cli import start_up, select_model, setup_hparams
from emotion_detection.utils import load_model, preprocess_frame, label_to_emotion
from emotion_detection.model import EmotionNet
from emotion_detection.dataset import get_train_and_val_dataloader
from pathlib import Path
from train import Trainer

import torch
import cv2
import sys

def inference_loop(model_path: str | Path) -> None:
    model = EmotionNet().cuda()
    model = load_model(model, model_path)

    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        exit()

    while True:
        ret, frame = capture.read()

        if not ret:
            print("Cannot receive frame from capture...", file=sys.stderr)
            break

        raw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        tensor_frame = torch.from_numpy(raw_frame)
        # preprocess the frame for the model
        tensor_frame = preprocess_frame(tensor_frame)
        model_prediction = model(tensor_frame)

        emotion = label_to_emotion(model_prediction.argmax().item())
        cv2.putText(raw_frame, emotion, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Input image", raw_frame)

        if cv2.waitKey(1) == ord('q'):
            break

        

    capture.release()
    cv2.destroyAllWindows()

def main():
    task_map = {"Train model": 0, "Inference": 1}

    task = start_up()

    match task_map[task]:
        case 0:
            train_hparams = setup_hparams()

            print("Initializing model...")
            model = EmotionNet()

            print("Initializing dataloaders...")
            train_dataloader, val_dataloader = get_train_and_val_dataloader()
            trainer = Trainer(model, **train_hparams)
            trainer.train(train_dataloader, val_dataloader)

        case 1:
            model_path = select_model()
            inference_loop(model_path)

if __name__ == "__main__":
    main()

