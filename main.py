from src.data import download_dataset, parse_grayscale_png

from pathlib import Path

import cv2
import sys

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        exit()
    while True:
        ret, frame = cap.read()


        if not ret:
            print("Can't receive frame...", file=sys.stderr)
            break

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

