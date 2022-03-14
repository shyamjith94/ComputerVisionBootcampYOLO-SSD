import numpy as np
from config import config
import cv2
import os

THRESHOLD = 0.3  # probabilities need  >30
SUPPRESSION_THRESHOLD = 0.3  # lower value the fewer bonding boxes will remain
YOLO_IMAGE_SIZE = 320
classes = ['car', 'person', 'bus']  # data set index (0:person, 2:car, 5:bus)


def setup_neural_network():
    """
    To download config and weight file
        https://pjreddie.com/darknet/yolo/
    place to yolo folder, download 320 pixel files
    :return: DNN
    """
    # we have 80(90) possible out put classes in coco data set. we are dealing with 3 possible ouput
    dnn_network = cv2.dnn.readNetFromDarknet(get_file_path(r"Yolo/yolov3.cfg"),
                                             get_file_path(r"Yolo/yolov3.weights"))
    # define run whether CPU or GPU
    dnn_network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    dnn_network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return dnn_network


def find_output_layer(dnn_wrk, image):
    """
    :param dnn_wrk deep neural network
    :param image its image or video frame
    :return: output layer
    """
    # transform image BLOB Yolo using BLOB, RGB -> BGR
    # 1/255 normalize the pixel value 320 is image scale size using yolo 320
    blob = cv2.dnn.blobFromImage(image, 1 / 255, (YOLO_IMAGE_SIZE, YOLO_IMAGE_SIZE), True, crop=False)
    dnn_wrk.setInput(blob)
    layer_names = dnn_wrk.getLayerNames()
    # neural_network has 3 out put layer
    output_layers = [layer_names[index[0] - 1] for index in dnn_wrk.getUnconnectedOutLayers()]
    # forward propagation
    output = dnn_wrk.forward(output_layers)
    return output


def get_file_path(rl_path=r"Data/camus.jpg"):
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


def find_objects(model_outputs):
    """
    fun used to extract prediction values from model
    we will filter the values using THRESHOLD point
    THRESHOLD 0.3 means need probabilities greater than 30 of predication class

    Apply non-max suppression
        same object will detect multiple times, consider highest probability and do the IOU(intersection over union)
    and make one single bonding box

    :param model_outputs: YOLO algorithm model
    :return: x,y,w,h array and class of probabilities and confidence value
    """
    bounding_box_locations = []
    class_ids = []
    confidence_values = []

    for outputs in model_outputs:
        for prediction in outputs:
            class_probabilities = prediction[5:]
            class_id = np.argmax(class_probabilities)
            confidence = class_probabilities[class_id]

            if confidence > THRESHOLD:
                w, h = int(prediction[2] * YOLO_IMAGE_SIZE), int(prediction[3] * YOLO_IMAGE_SIZE)
                # the center of the bounding box (we should transform these values)
                x, y = int(prediction[0] * YOLO_IMAGE_SIZE - w / 2), int(prediction[1] * YOLO_IMAGE_SIZE - h / 2)
                bounding_box_locations.append([x, y, w, h])
                class_ids.append(class_id)
                confidence_values.append(float(confidence))

    box_indexes_to_keep = cv2.dnn.NMSBoxes(bounding_box_locations, confidence_values, THRESHOLD, SUPPRESSION_THRESHOLD)

    return box_indexes_to_keep, bounding_box_locations, class_ids, confidence_values


def show_predicted_object(org_mage, predicted_obj_id, bbox_location, c_label_id, conf_level, width_ratio, height_ratio):
    """it show the predicted bonding box original image

    :param org_mage -> the original image
    :param predicted_obj_id -> the predicted object id (bonding box index)
    :param bbox_location -> bonding box location
    :param c_label_id -> class label ids
    :param conf_level -> confidence level or confidence value
    :param height_ratio, width_ratio -> original size image can be any size, but YOLO algorithm will take 320*320
                need to resize the image
    """
    for index in predicted_obj_id:
        bonding_box = bbox_location[index[0]]
        x, y, w, h = int(bonding_box[0]), int(bonding_box[1]), int(bonding_box[2]), int(bonding_box[3])
        # we have transform the location adn to coordinate value because resize image
        x = int(x * width_ratio)
        y = int(y * height_ratio)
        w = int(w * width_ratio)
        h = int(h * height_ratio)
        # we detecting only person and car
        if c_label_id[index[0]] == 2:  # in classes list index id 2 for car
            cv2.rectangle(org_mage, (x, y), (x + w, y + h), (255, 0, 0))  # BGR blue box
            class_with_confidence = 'CAR' + str(int(conf_level[index[0]] * 100)) + '%'
            cv2.putText(org_mage, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 0),
                        1)
        if c_label_id[index[0]] == 0:
            cv2.rectangle(org_mage, (x, y), (x + w, y + h), (255, 0, 0))  # BGR blue box
            class_with_confidence = 'PERSON' + str(int(conf_level[index[0]] * 100)) + '%'
            cv2.putText(org_mage, class_with_confidence, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0,
                                                                                                            0), 1)
        cv2.imshow('YOLO ALGORITHM', org_mage)
        cv2.waitKey(1)


def main(file_format, file_path):
    """
    :param file_format: image or video file, 0 video file like mp4, 1 is image file
    :param file_path:  relative path of a file
    :return: None
    """
    file_format = int(file_format)
    neural_network = setup_neural_network()
    if file_format == 0:
        capture = cv2.VideoCapture(get_file_path(file_path))
        while True:
            frame_grabbed, frame = capture.read()

            if not frame_grabbed:
                break

            output_layer = find_output_layer(neural_network, frame)
            # first layer (300, 85) 300 prediction bonding box and 85 prediction vector
            # 85 object have (x,y w,h, confidence)
            predicted_object_id, bbox_locations, class_label_ids, conf_value = find_objects(output_layer)
            # showing the image with bbox
            original_width, original_height = frame.shape[1], frame.shape[0]
            show_predicted_object(frame, predicted_obj_id=predicted_object_id, bbox_location=bbox_locations,
                                  c_label_id=class_label_ids, conf_level=conf_value,
                                  width_ratio=original_width / YOLO_IMAGE_SIZE,
                                  height_ratio=original_height / YOLO_IMAGE_SIZE)
        capture.release()
        cv2.destroyAllWindows()
    elif file_format == 1:
        image = cv2.imread(get_file_path())
        original_width, original_height = image.shape[1], image.shape[0]
        output_layer = find_output_layer(neural_network, image)
        # first layer (300, 85) 300 prediction bonding box and 85 prediction vector
        # 85 object have (x,y w,h, confidence)
        predicted_object_id, bbox_location, class_label_ids, conf_value = find_objects(output_layer)
        # showing the image with bbox
        show_predicted_object(image, predicted_obj_id=predicted_object_id, bbox_location=bbox_location,
                              c_label_id=class_label_ids, conf_level=conf_value,
                              width_ratio=original_width / YOLO_IMAGE_SIZE,
                              height_ratio=original_height / YOLO_IMAGE_SIZE)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(file_format=0, file_path=r"Data\yolo_test.mp4")
