import cv2
import torch

from tracker import *

INPUT_PATH = './video/min_trim.mkv'


def draw_boxes(frame, boxes):
    '''
    This function draw boxes with tracks on frame
    :param frame: the original frame
    :param boxes: boxes from detector
    :return: frame with drawn boxes and tracks
    '''

    # update the ids of boxes
    boxes_ids = tracker.update(boxes, 50)

    # draw each box
    for box in boxes_ids:
        x, y, w, h, class_idx, id = box
        label = labels[int(class_idx)]
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, label + str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # return frame
    return frame


if __name__ == '__main__':

    # load the model and initialize the tracker
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    tracker = DistTracker()

    # open the video
    cap = cv2.VideoCapture(INPUT_PATH)
    cap.open(INPUT_PATH)

    width = int(cap.get(3))
    height = int(cap.get(4))

    # initialize the output
    out = cv2.VideoWriter('out_simple.mkv',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          30.0, (width, height))

    while cap.isOpened():

        # read the frame
        flag, frame = cap.read()
        if not flag:
            break

        # preprocess the frame
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        result = model(processed_frame)
        boxes = result.xyxy[0].numpy().tolist()
        labels = result.names

        # draw boxes
        frame = draw_boxes(frame, boxes)

        # write frame to the output and show
        out.write(frame)
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)

        # escape if user wants
        if key == 27:
            break

    # release and destroy windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
