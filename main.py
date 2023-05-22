import cv2
import torch

from tracker import *

if __name__ == '__main__':

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    tracker = DistTracker()

    cap = cv2.VideoCapture("video/min_trim.mkv")
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.mkv', fourcc, 30.0, (640, 480))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = model(processed_frame)
        boxes = result.xyxy[0].numpy().tolist()
        labels = result.names

        boxes_ids = tracker.update(boxes)

        for box in boxes_ids:
            x, y, w, h, class_idx, id = box
            label = labels[int(class_idx)]
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, label + str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        cv2.imshow("frame", frame)
        cv2.waitKey(0)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
