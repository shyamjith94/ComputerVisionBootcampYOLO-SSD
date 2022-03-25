import numpy as np
import cv2
import os

from config import config

SSD_INPUT_SIZE = 320
THRESHOLD = 0.4
SUPPRESSION_THRESHOLD = 0.3


def get_file_path(rl_path=r"Ssd/class_names.txt"):
    """
    :param rl_path: relative path
    :return: full path
    """
    os.chdir(config.ROOT_DIR)
    current_dir = os.getcwd()
    path = os.path.join(current_dir, rl_path)
    if os.path.exists(path):
        os.chdir('../')
        return path
    else:
        raise Exception(f"{rl_path} not found in directory")


def construct_class_names(file_name):
    """
    read class labels from file
    :return: [label class names]
    """
    with open(file_name, "rt") as file:
        names = file.read().rstrip('\n').split('\n')
    return names


def setup_neural_network():
    """
    :return: DNN
    """
    dnn_network = cv2.dnn_DetectionModel(get_file_path('Ssd/ssd_weights.pb'), get_file_path(
        'Ssd/ssd_mobilenet_coco_cfg.pbtxt'))
    dnn_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    dnn_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # set input size
    dnn_network.setInputSize(SSD_INPUT_SIZE, SSD_INPUT_SIZE)
    dnn_network.setInputScale(1.0 / 127.5)
    dnn_network.setInputMean([127.5, 127.5, 127.5])
    # swap color channel R to B
    dnn_network.setInputSwapRB(True)
    return dnn_network


def show_detected_object(frame, box_to_index, all_bbox, object_names, class_lebel_id):
    """
    :param frame: video or image frame
    :param box_to_index:
    :param conf:
    :param all_bbox:
    :return:
    """
    for index in box_to_index:
        box = all_bbox[index]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x,y), (x+w, y+h), color=(255, 0, 0), thickness=1)
        cv2.putText(frame, object_names[class_lebel_id[index]-1].upper(), (box[0], box[1] -10),
                                                                                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7,
                                                                                 (255, 0, 0), 1)
        cv2.imshow('SSD Algorithm', frame)
        cv2.waitKey(5)


def detect_object(frame, dnn_net):
    """detect frames and eliminate unnecessary bonding box
        non-maximum suppression region

    :param frame image or video frame
    :param dnn_net deep neural network

    :return box_to_keep(index of bbox), confidence, bbox(all bbox)
    """
    class_label_id, confidence, bbox = dnn_net.detect(frame)
    bbox = list(bbox)
    confidence = np.array(confidence).reshape(1, -1).tolist()[0]
    # index of bonding box
    box_to_keep = cv2.dnn.NMSBoxes(bbox, confidence, THRESHOLD, SUPPRESSION_THRESHOLD)
    return class_label_id, box_to_keep, bbox,


def main(file_format, file_path):
    """
    :param file_format: image or video file, 0 video file like mp4, 1 is image file
    :param file_path:  relative path of a file
    :return: None
    """
    file_format = int(file_format)
    neural_network = setup_neural_network()
    class_names = construct_class_names(file_name=get_file_path("Ssd/class_names"))
    if file_format == 0:
        capture = cv2.VideoCapture(get_file_path(file_path))
        while True:
            frame_grabbed, frame = capture.read()
            class_label_id, box_to_keep, all_box, = detect_object(frame, neural_network)
            show_detected_object(frame, box_to_keep, all_box, class_names, class_label_id)
            if not frame_grabbed:
                break
        capture.release()
        cv2.destroyAllWindows()

    elif file_format == 1:
        pass


if __name__ == '__main__':
    main(file_format=0, file_path=r"Data\objects.mp4")
