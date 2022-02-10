import cv2
import os


def get_file_path(rl_path=r"FaceDetection/haarcascade_frontalface_alt.xml"):
    """
    :param rl_path: relative path
    :return: full path
    """
    os.chdir('../')
    current_dir = os.getcwd()
    path = os.path.join(current_dir, rl_path)
    if os.path.exists(path):
        return path
    else:
        raise Exception(f"{rl_path} not found in directory")


cascade_classifier = cv2.CascadeClassifier(get_file_path())


def detect_face(classifier=cascade_classifier):
    """
    detect face from video frame
    :param classifier open cv cascade classifier default is front face we switch
    :return:
    """
    # open default camera - 0 web cam
    video_capture = cv2.VideoCapture(0)
    # setting width and height of video frame
    video_capture.set(3, 640)
    video_capture.set(4, 480)
    while True:
        # return video frame
        ret, img = video_capture.read()
        # convert to gray scale
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=10,
                                                     minSize=(30, 30))
        for x, y, width, height in detected_faces:
            cv2.rectangle(img, (x, y), (x + width, y + height), (0, 0, 255), 10)
        # set title of video frame
        cv2.imshow("real time face detection", img)
        # exit from video frame, key pressed or ESC to quit
        key = cv2.waitKey(30) & 0xff
        if key == 27:  # 27 int representation of ESC
            break
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_face()
