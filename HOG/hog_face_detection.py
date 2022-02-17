# using SVM to classification of images human or non-human
# using  HOG feature.hog values are using to train model
# the test data also resize and convert to feature.hog
# labels are mode 1 and 0, i is human face and 0 is non-human face

from skimage import data, feature, transform
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from itertools import chain
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from skimage.io import imread
import numpy as np
import os


def get_file_path(rl_path=r"Data/bird.jpg"):
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


def collect_data():
    """
    prepare data sample for human and non-human face.
    non-human faces are ['moon', 'text', 'coins']
    negative sample have three images. after-
    generate num sample

    :return: negative ang positive sample of data
    """
    # load dataset human people (positive image)
    human_faces = fetch_lfw_people()
    positive_image = human_faces.images[:10000]
    # load dataset without faces (negative images)
    non_human_faces = ['moon', 'text', 'coins']
    negative_image = [(getattr(data, name)()) for name in non_human_faces]
    return positive_image, negative_image


def generate_random_samples(image, num_generated_image=100, patch_size=(62, 47)):
    """will use the PatchGenerator to generate several variant of these image

    :param patch_size need images same size of default take first positive image size
    :param image sample image
    :param num_generated_image num of images required
    """
    extractor = PatchExtractor(patch_size=patch_size, max_patches=num_generated_image, random_state=42)
    patches = extractor.transform((image[np.newaxis]))  # default 2 dim need one
    return patches


def create_model(p_images, n_images):
    """
    create svm classification model and train the data
    before train all data point are convert to HOG values
    create label data values are 0 and 1

    :param p_images: positive_images
    :param n_images: negative_images
    :return: SVM LinearSVC
    """
    # we construct the training set with the output variable (labels) using hog
    x_train = np.array([feature.hog(image) for image in chain(p_images, n_images)])
    # create labels 0, 1 0-non human face, 1-human face
    y_train = np.zeros(x_train.shape[0])  # num of rows
    y_train[:p_images.shape[0]] = 1

    # create svm , train model
    svm = LinearSVC()
    svm.fit(x_train, y_train)
    return svm


def predict_images(svm_model, image_path, abs_path=True):
    """
    :param image_path path new test image
    :param svm_model train classification model of SVM LinearSVC
    :param abs_path its a bool value if we passing
    :return: None
    """
    # test model
    if abs_path is not True:
        image_path = get_file_path(image_path)
    print(image_path)
    org_image = imread(image_path)
    test_image = transform.resize(org_image, positive_images[0].shape)
    # test image convert to HOG
    test_image_hog = np.array([feature.hog(test_image, multichannel=True)])
    prediction = svm_model.predict(test_image_hog)
    print(prediction)
    if int(prediction) == 0:
        text = "Non-Human"
    else:
        text = "Human"
    plt.imshow(org_image)
    plt.title(f"SVM LinearSVC Prediction\n"
              f"Its a {text}")
    plt.show()


if __name__ == '__main__':
    positive_images, negative_sample = collect_data()
    negative_images = np.vstack(generate_random_samples(im, 1000, positive_images[0].shape) for im in negative_sample)
    model = create_model(p_images=positive_images, n_images=negative_images)
    predict_images(svm_model=model, image_path=r"Data/girl_face.png", abs_path=False)
