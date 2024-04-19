from typing import Any

import pandas as pd
from PIL import Image
import os
import joblib
import cv2
import numpy as np
import random
from skimage.transform import rotate


########################################################################################################################
# Data augmentation function
########################################################################################################################


def preprocess_image_no_blur_function(image):
    preprocess_image = np.array(image)
    preprocess_image = cv2.resize(preprocess_image, (224, 224))
    preprocess_image = cv2.cvtColor(
        preprocess_image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Randomly decide whether to rotate the image
    if random.random() < 0.1:  # 1/10 chance
        # Randomly choose an angle from [90, 180, 270]
        angle = random.choice([90, 180, 270])
        preprocess_image = rotate(
            preprocess_image, angle, preserve_range=True).astype(np.uint8)
    elif 0.1 < random.random() < 0.2:
        preprocess_image = cv2.flip(preprocess_image, 1)

    return preprocess_image

########################################################################################################################
# Data Loading functions
########################################################################################################################


def load_image_labels(labels_file_path: str):
    """
    Loads the labels from CSV file.

    :param labels_file_path: CSV file containing the image and labels.
    :return: Pandas DataFrame
    """
    df = pd.read_csv(labels_file_path)
    return df


def load_predict_image_names(predict_image_list_file: str):
    """
    Reads a text file with one image file name per line and returns a list of files
    :param predict_image_list_file: text file containing the image names
    :return list of file names:
    """
    with open(predict_image_list_file, 'r') as file:
        lines = file.readlines()
    # Remove trailing newline characters if needed
    lines = [line.rstrip('\n') for line in lines]
    return lines


def load_single_image(image_file_path: str) -> Image:
    """
    Load the image.

    NOTE: you can optionally do some initial image manipulation or transformation here.

    :param image_file_path: the path to image file.
    :return: Image (or other type you want to use)
    """
    # Load the image
    image = Image.open(image_file_path)

    # The following are examples on how you might manipulate the image.
    # See full documentation on Pillow (PIL): https://pillow.readthedocs.io/en/stable/

    # To make the image 50% smaller
    # Determine image dimensions
    # width, height = image.size
    # new_width = int(width * 0.50)
    # new_height = int(height * 0.50)
    # image = image.resize((new_width, new_height))

    # To crop the image
    # (left, upper, right, lower) = (20, 20, 100, 100)
    # image = image.crop((left, upper, right, lower))

    # To view an image
    # image.show()

    # Return either the pixels as array - image_array
    # To convert to a NumPy array
    # image_array = np.asarray(image)
    # return image_array

    # or return the image
    return image


########################################################################################################################
# Model Loading and Saving Functions
########################################################################################################################


def save_model(model: Any, target: str, output_dir: str):
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation.

    Common Deep Learning Model File Formats are:

        SavedModel (TensorFlow)
        Pros: Framework-agnostic format, can be deployed in various environments. Contains a complete model representation.
        Cons: Can be somewhat larger in file size.

        HDF5 (.h5) (Keras)
        Pros: Hierarchical structure, good for storing model architecture and weights. Common in Keras.
        Cons: Primarily tied to the Keras/TensorFlow ecosystem.

        ONNX (Open Neural Network Exchange)
        Pros: Framework-agnostic format aimed at improving model portability.
        Cons: May not support all operations for every framework.

        Pickle (.pkl) (Python)
        Pros: Easy to save and load Python objects (including models).
        Cons: Less portable across languages and environments. Potential security concerns.

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param model: the model that you want to save.
    :param target: the target value - can be useful to name the model file for the target it is intended for
    :param output_dir: the output directory to same one or more model files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the full path to save the model
    model_path = os.path.join(output_dir, f"{target}.joblib")

    # Save the model using joblib
    joblib.dump(model, model_path)
    print(f"Model saved successfully at {model_path}")


def load_model(trained_model_dir: str, target_column_name: str) -> Any:
    """
    Given a model and target label, save the model file in the output_directory.

    The specific format depends on your implementation and should mirror save_model()

    IMPORTANT: Add additional arguments as needed. We have just given the model as an argument, but you might have
    multiple files that you save.

    :param trained_model_dir: the directory where the model file(s) are saved.
    :param target_column_name: the target value - can be useful to name the model file for the target it is intended for
    :returns: the model
    """
    # TODO: implement your model loading code here
    raise RuntimeError("load_model() is not implemented.")
