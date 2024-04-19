import argparse
import os
from typing import Any

import tensorflow as tf
import pandas as pd
from PIL import Image
from train import load_train_resources, extract_features
from common import load_model, load_predict_image_names, load_single_image, preprocess_image_no_blur_function
import joblib
from tensorflow.keras.applications import ResNet152V2, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import torch

########################################################################################################################


def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir',
                        required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True,
                        help='Path to CSV where to write the predictions')
    args = parser.parse_args()
    return args


def predict(model: Any, image: Image) -> str:
    """
    Generate a prediction for a single image using the model, returning a label of 'Yes' or 'No'

    IMPORTANT: The return value should ONLY be either a 'Yes' or 'No' (Case sensitive)

    :param model: the model to use.
    :param image: the image file to predict.
    :return: the label ('Yes' or 'No)
    """
    predicted_label = 'No'
    # TODO: Implement your logic to generate prediction for an image here.
    raise RuntimeError("predict() is not implemented.")
    return predicted_label


def main(predict_data_image_dir: str,
         predict_image_list: str,
         target_column_name: str,
         trained_model_dir: str,
         predicts_output_csv: str):
    """
    The main body of the predict.py responsible for:
     1. load model
     2. load predict image list
     3. for each entry,
           load image
           predict using model
     4. write results to CSV

    :param predict_data_image_dir: The directory containing the prediction images.
    :param predict_image_list: Name of text file within predict_data_image_dir that has the names of image files.
    :param target_column_name: The name of the prediction column that we will generate.
    :param trained_model_dir: Path to the directory containing the model to use for predictions.
    :param predicts_output_csv: Path to the CSV file that will contain all predictions.
    """
    if target_column_name == "Is Epic":
        # load pre-trained models or resources at this stage.
        resnet_model = load_train_resources(
            target_column_name).get('resnet50_model')

        # Load in the image list
        image_list_file = os.path.join(
            predict_data_image_dir, predict_image_list)
        image_filenames = load_predict_image_names(image_list_file)
        predict_images = [tf.keras.utils.load_img((os.path.join(
            predict_data_image_dir, image)), target_size=(224, 224)) for image in image_filenames]
        features = extract_features(predict_images, resnet_model)
        model = joblib.load(trained_model_dir + "/" + "Is Epic.joblib")
        # Iterate through the image list to generate predictions
        predictions = model.predict(features)
        predictions = ['Yes' if pred == 1 else 'No' for pred in predictions]
        print("predictions", predictions)
        df_predictions = pd.DataFrame(
            {'Filenames': image_filenames, target_column_name: predictions})
        # Finally, write out the predictions to CSV
        df_predictions.to_csv(predicts_output_csv, index=False)
    elif target_column_name == "Needs Respray":
        # Load a model
        model = YOLO("./runs/detect/train2/weights/best.pt")

        image_list_file = os.path.join(
            predict_data_image_dir, predict_image_list)
        image_filenames = load_predict_image_names(image_list_file)

        image_paths = []
        # load all the image path that ends with .png into the images_paths list
        for filename in image_filenames:
            image_path = os.path.join(predict_data_image_dir, filename)
            image_paths.append(image_path)

        yes_no = []
        for image in image_path:
            results = model(image)
            r = r.results[0]
            yes_no.append(predict(image, r))

        predictions = ['Yes' if pred == 1 else 'No' for pred in yes_no]
        df_predictions = pd.DataFrame(
            {'Filenames': image_filenames, target_column_name: predictions})
        # Finally, write out the predictions to CSV
        df_predictions.to_csv(predicts_output_csv, index=False)

    elif target_column_name == "Is GenAI":
        image_list_file = os.path.join(
            predict_data_image_dir, predict_image_list)
        image_filenames = load_predict_image_names(image_list_file)

        image_paths = []
        # load all the image path that ends with .png into the images_paths list
        for filename in image_filenames:
            image_path = os.path.join(predict_data_image_dir, filename)
            image_paths.append(image_path)

        # Load all PNG images in the image_path list and its label
        images = [Image.open(image_path) for image_path in image_paths]

        predict_images = [preprocess_image_no_blur_function(
            image) for image in images]

        # Preprocess the images
        predict_images = np.array(predict_images)

        # Load pre-trained Resnet152 model
        resnet_model = ResNet152(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Load the model architecture
        loaded_model = tf.keras.Sequential([
            resnet_model,
            GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model (necessary before loading weights)
        loaded_model.compile(
            optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

        # Load the weights into the model
        loaded_model.load_weights(
            trained_model_dir + "ai_human_classify_model_weights_resnet152.h5")

        # Predict the labels for the images
        predictions = loaded_model.predict(predict_images)
        binary_predictions = (predictions > 0.5).astype(int)

        print("binary_predictions", binary_predictions)
        predictions_results = ['Yes' if pred ==
                               1 else 'No' for pred in binary_predictions]

        df_predictions = pd.DataFrame(
            {'Filenames': image_filenames, target_column_name: predictions_results})

        # Finally, write out the predictions to CSV
        df_predictions.to_csv(predicts_output_csv, index=False)


def CompareBrickColor(image_path):
    # Read image
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for brick color in HSV
    lower_brick = np.array([0, 50, 50])
    upper_brick = np.array([15, 255, 255])

    # Create mask for brick color
    mask_brick = cv2.inRange(hsv, lower_brick, upper_brick)

    # Bitwise-AND mask and original image
    result1 = cv2.bitwise_and(image, image, mask=mask_brick)

    # Define the lower and upper bounds for the RGB range
    lower_rgb = np.array([40, 40, 40])
    upper_rgb = np.array([255, 215, 230])

    # Create mask for the specified RGB range
    mask = cv2.inRange(image, lower_rgb, upper_rgb)

    # Split the image into its BGR components
    b, g, r = cv2.split(image)

    # Create mask for red > green
    mask_red_greater_than_green = (r >= g).astype(np.uint8) * 255

    # Combine masks
    final_mask = cv2.bitwise_and(mask, mask_red_greater_than_green)

    # Bitwise-AND mask and original image
    result2 = cv2.bitwise_and(image, image, mask=final_mask)

    # Count non-zero pixels in each mask
    count_result1 = cv2.countNonZero(mask_brick)
    count_result2 = cv2.countNonZero(final_mask)

    # Choose the mask with more non-zero pixels
    if count_result1 > count_result2:
        return result1
    else:
        return result2


def calculate_dark_pixel_percentage(image, x1, y1, x2, y2):
    # Extract the region of interest (ROI) from the image
    roi = image[y1:y2, x1:x2]

    # Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for dark colors in HSV
    lower_dark = np.array([0, 0, 0])
    # Adjust the threshold for darkness as needed
    upper_dark = np.array([179, 255, 50])

    # Create a mask to extract pixels within the dark color range
    mask_dark = cv2.inRange(hsv_roi, lower_dark, upper_dark)

    # Calculate the number of dark pixels
    dark_pixel_count = cv2.countNonZero(mask_dark)

    # Calculate the total number of pixels in the ROI
    total_pixels = (x2 - x1) * (y2 - y1)

    # Calculate the percentage of dark pixels
    dark_pixel_percentage = (dark_pixel_count / total_pixels) * 100

    return dark_pixel_percentage, total_pixels
# Assuming results is a list of detections


def predict(image_path, r):
    image_test = CompareBrickColor(image_path)
    number_of_alive_weed = 0
    number_of_dead_weed = 0
    for i, box in enumerate(r.boxes.xyxy):
        x1_tensor, y1_tensor, x2_tensor, y2_tensor = box
        x1 = int(torch.tensor(x1_tensor))
        y1 = int(torch.tensor(y1_tensor))
        x2 = int(torch.tensor(x2_tensor))
        y2 = int(torch.tensor(y2_tensor))
        percentage_dark_pixels, area = calculate_dark_pixel_percentage(
            image_test, x1, y1, x2, y2)
        if (percentage_dark_pixels < 80):
            if (r.boxes[i].cls == 0):
                number_of_alive_weed += 1
            if (r.boxes[i].cls == 1):
                number_of_dead_weed += 1
    if (number_of_alive_weed == 0 and number_of_dead_weed == 0):
        return 0
    elif (number_of_alive_weed > number_of_dead_weed):
        return 1
    else:
        return 0


if __name__ == '__main__':
    """
    Example usage:

    python predict.py -d "path/to/Data - Is Epic Intro Full" -l "Is Epic Files.txt" -t "Is Epic" -m "path/to/Is Epic/model" -o "path/to/Is Epic Full Predictions.csv"

    """
    args = parse_args()
    predict_data_image_dir = args.predict_data_image_dir
    predict_image_list = args.predict_image_list
    target_column_name = args.target_column_name
    trained_model_dir = args.trained_model_dir
    predicts_output_csv = args.predicts_output_csv
    main(predict_data_image_dir, predict_image_list,
         target_column_name, trained_model_dir, predicts_output_csv)

########################################################################################################################
