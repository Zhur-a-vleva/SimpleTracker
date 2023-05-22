import cv2

from inference import Network
from tracker import *

MODEL_PATH = './model/vehicle-detection-adas-0002.xml'
INPUT_PATH = './video/min_trim.mkv'
CONFIDENCE_BOUND = 0.4
COLOR = (0, 255, 0)
tracker = DistTracker()


def draw_boxes(frame, result, width, height):
    '''
    This function draw boxes with tracks on frame
    :param frame: the original frame
    :param result: boxes from detector
    :param width: the original width
    :param height: the original height
    :return: frame with drawn boxes and tracks
    '''
    boxes = []

    # filter boxes and append them to boxes in the form suitable for tracker
    for box in result[0][0]:
        if box[2] > CONFIDENCE_BOUND:
            boxes.append(
                [int(box[3] * width), int(box[4] * height), int(box[5] * width), int(box[6] * height), box[2], 1])

    # update the ids of boxes
    boxes_ids = tracker.update(boxes, 100)

    # draw each box
    for box in boxes_ids:
        x, y, w, h, class_idx, id = box
        label = "vehicle "
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        cv2.putText(frame, label + str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # return frame
    return frame


if __name__ == "__main__":

    plugin = Network()

    # load the model
    plugin.load_model(MODEL_PATH, 'CPU')
    net_input_shape = plugin.get_input_shape()

    # open the video
    cap = cv2.VideoCapture(INPUT_PATH)
    cap.open(INPUT_PATH)

    width = int(cap.get(3))
    height = int(cap.get(4))

    # initialize the output
    out = cv2.VideoWriter('out_open_vino.mkv',
                          cv2.VideoWriter_fourcc(*'MJPG'),
                          30.0, (width, height))

    while cap.isOpened():

        # read the frame
        flag, frame = cap.read()
        if not flag:
            break

        # preprocess the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # get boxes
        result = plugin.inference(p_frame)

        # draw boxes
        frame = draw_boxes(frame, result, width, height)

        # write frame to the output and show
        out.write(frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)

        # escape if user wants
        if key == 27:
            break

    # release and destroy windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()
