import cv2
import torch
import os
import numpy as np


if __name__ == '__main__':

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    cap = cv2.VideoCapture("video/min_trim.mkv")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.mkv', fourcc, 30.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = model(frame)
        boxes = result.xyxy[0].numpy().tolist()
        labels = result.names

        for box in boxes:
            x1, y1, x2, y2, confidence, class_idx = box
            label = labels[int(class_idx)]
            if label in ['bicycle', 'car', 'motorcycle', 'bus', 'train', 'truck']:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


