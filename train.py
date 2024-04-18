import argparse
import os
import tensorflow as tf
from typing import Any, List
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from common import load_image_labels, load_single_image, save_model

########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv', required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True, help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir', required=True, help='Output directory for trained model')
    parser.add_argument('-c', '--case', required=True, choices=['Is Epic', 'Needs Respray', 'Is GenAI'], help='Specify the case for the model training and prediction')
    args = parser.parse_args()
    return args

def load_train_resources(case_type: str):
    """
    Load specific resources like pre-trained models or data files based on the case type.
    :param case_type: a string identifier for the case to determine which resources to load
    :return: a dictionary of loaded resources
    """
    resources = {}
    if case_type == 'Is Epic':
        print("Loading ResNet50 model for 'Is Epic' case...")
        resources['resnet50_model'] = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
        print("ResNet50 model loaded.")
    return resources

def train(images , labels, output_dir: str, case_type: str, resources) -> Any:
    """
    Trains a classification model using the training images and corresponding labels.

    :param images: the list of image (or array data)
    :param labels: the list of training labels (str or 0,1)
    :param output_dir: the directory to write logs, stats, etc to along the way
    :return: model: model file(s) trained.
    """
    # TODO: Implement your logic to train a problem specific model here
    # Along the way you might want to save training stats, logs, etc in the output_dir
    # The output from train can be one or more model files that will be saved in save_model function.
    if case_type == 'Is Epic':
        resnet_model = resources.get('resnet50_model')
        features = extract_features(images, resnet_model)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=59, stratify=labels)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logistic_model = LogisticRegression(C=0.01, penalty='l2', solver='liblinear', max_iter=1000)
        logistic_model.fit(X_train_scaled, y_train)

        y_pred = logistic_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.2f}")
        # Save the trained model
        return logistic_model


def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str, case_type: str):
    """
    The main body of the train.py responsible for
     1. loading resources
     2. loading labels
     3. loading data
     4. transforming data
     5. training model
     6. saving trained model

    :param train_input_dir: the folder with the CSV and training images.
    :param train_labels_file_name: the CSV file name
    :param target_column_name: Name of the target column within the CSV file
    :param train_output_dir: the folder to save training output.
    :param case_type: the case for the model training and prediction
    """

    if case_type == 'Is Epic':
        # load pre-trained models or resources at this stage.
        resources = load_train_resources(case_type)

        labels_file_path  = os.path.join(train_input_dir, train_labels_file_name)
        df_labels = pd.read_csv(labels_file_path)

        train_images = [tf.keras.utils.load_img((os.path.join(train_input_dir, row['Filename'])), target_size=(224, 224)) for index, row in df_labels.iterrows()]
        train_labels = [1 if row[target_column_name] == 'Yes' else 0 for index, row in df_labels.iterrows()]

        os.makedirs(train_output_dir, exist_ok=True)
        model = train(train_images, train_labels, train_output_dir, case_type, resources)
        save_model(model, target_column_name, train_output_dir)

def extract_features(images, model):
    """ Extract features from a batch of images using the provided model. """
    features = []
    for img in images: # Ensure image is resized to match model input
        img_array = tf.keras.utils.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        processed_img = tf.keras.applications.resnet50.preprocess_input(img_array_expanded)
        feature = model.predict(processed_img)
        features.append(feature.flatten())
    return np.array(features)

if __name__ == '__main__':
    """
    Example usage:

    python train.py -d "path/to/Data - Is Epic Intro 2024-03-25" -l "Labels-IsEpicIntro-2024-03-25.csv" -t "Is Epic" -o "path/to/models"

    """
    args = parse_args()
    train_data_image_dir = args.train_data_image_dir
    train_data_labels_csv = args.train_data_labels_csv
    target_column_name = args.target_column_name
    trained_model_output_dir = args.trained_model_output_dir
    case_type = args.case

    main(train_data_image_dir, train_data_labels_csv, target_column_name, trained_model_output_dir, case_type)

########################################################################################################################
