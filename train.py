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
from common import load_image_labels, load_single_image, save_model, preprocess_image_no_blur_function
from ultralytics import YOLO
from tqdm import tqdm
import seaborn as sns
from PIL import Image

from tensorflow.keras.applications import ResNet152V2, ResNet152
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers

from skimage.transform import rotate
from sklearn.model_selection import train_test_split

import shutil
########################################################################################################################


def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--train_data_image_dir',
                        required=True, help='Path to image data directory')
    parser.add_argument('-l', '--train_data_labels_csv',
                        required=True, help='Path to labels CSV')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of the column with target label in CSV')
    parser.add_argument('-o', '--trained_model_output_dir',
                        required=True, help='Output directory for trained model')
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
        resources['resnet50_model'] = tf.keras.applications.ResNet50(
            weights='imagenet', include_top=False, pooling='avg')
        print("ResNet50 model loaded.")
    return resources


def train(images, labels, output_dir: str, case_type: str, resources) -> Any:
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

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=59, stratify=labels)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logistic_model = LogisticRegression(
            C=0.01, penalty='l2', solver='liblinear', max_iter=1000)
        logistic_model.fit(X_train_scaled, y_train)

        y_pred = logistic_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.2f}")
        # Save the trained model
        return logistic_model


def main(train_input_dir: str, train_labels_file_name: str, target_column_name: str, train_output_dir: str):
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

    if target_column_name == 'Is Epic':
        # load pre-trained models or resources at this stage.
        resources = load_train_resources(target_column_name)

        labels_file_path = os.path.join(
            train_input_dir, train_labels_file_name)
        df_labels = pd.read_csv(labels_file_path)

        train_images = [tf.keras.utils.load_img((os.path.join(train_input_dir, row['Filename'])), target_size=(
            224, 224)) for index, row in df_labels.iterrows()]
        train_labels = [1 if row[target_column_name] ==
                        'Yes' else 0 for index, row in df_labels.iterrows()]

        os.makedirs(train_output_dir, exist_ok=True)
        model = train(train_images, train_labels,
                      train_output_dir, target_column_name, resources)
        save_model(model, target_column_name, train_output_dir)
    elif target_column_name == 'Needs Respray':
        if os.path.exists('./runs'):
            os.rmdir('./runs')
        new_resource = YOLO('yolov8x')
        new_resource.train(data='./resources-20240419T052918Z-001/resources/mydata4.yaml',
                           epochs=50)
        resource = YOLO('./runs/detect/train/weights/best.pt')
        resource.train(data='./resources-20240419T052918Z-001/resources/mydata.yaml',
                       epochs=50)

    elif target_column_name == "Is GenAI":
        # get label of data
        labels_file_path = os.path.join(
            train_input_dir, train_labels_file_name)
        df_data = load_image_labels(labels_file_path)
        labels = df_data['Is GenAI'].tolist()
        labels = [1 if label == 'Yes' else 0 for label in labels]
        print("train_input_dir", train_input_dir)
        # get all images
        data_path = train_input_dir
        images_name = df_data['Filename'].tolist()
        image_paths = []

        # load all the image path that ends with .png into the images_paths list
        for filename in images_name:
            image_path = os.path.join(data_path, filename)
            image_paths.append(image_path)

        # Load all PNG images in the image_path list and its label
        images = [Image.open(image_path) for image_path in image_paths]

       # load all the image path that ends with .png into the images_paths list
        for filename in images_name:
            image_path = os.path.join(data_path, filename)
            image_paths.append(image_path)

        preprocessed_images = []
        for image in images:
            preprocessed_images.append(
                preprocess_image_no_blur_function(image))

        # Split the data into train and test sets (80% train, 20% validation)
        train_images, val_images, train_labels, val_labels = train_test_split(
            preprocessed_images, labels, test_size=0.2, random_state=42)

        # Convert lists of images to NumPy arrays
        train_images = np.array(train_images)
        val_images = np.array(val_images)

        # Convert lists of labels to NumPy arrays
        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)

        def lr_scheduler(epoch, lr):
            decay_rate = 0.6
            decay_step = 10
            if epoch % decay_step == 0 and epoch:
                return lr * decay_rate
            return lr

        weight_decay = 1e-5  # Adjust this value as needed

        # Load pre-trained Resnet152 model
        resnet_model = ResNet152(
            weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        # Freeze all layers except the last few
        for layer in resnet_model.layers:
            layer.trainable = False

        # Add classification head
        model = Sequential([
            resnet_model,
            GlobalAveragePooling2D(),
            Flatten(),
            Dropout(0.2),
            Dense(256, activation='relu',
                  kernel_regularizer=regularizers.l2(weight_decay)),
            Dropout(0.2),
            Dense(1, activation='sigmoid',
                  kernel_regularizer=regularizers.l2(weight_decay)),
        ])

        # Compile the model with SGD optimizer and initial learning rate
        initial_learning_rate = 0.0002
        sgd = SGD(learning_rate=initial_learning_rate)
        model.compile(optimizer=sgd, loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Define the learning rate scheduler
        lr_callback = LearningRateScheduler(lr_scheduler)

        # Train the model with the learning rate scheduler
        history = model.fit(train_images, train_labels,
                            validation_data=(val_images, val_labels),
                            epochs=20, batch_size=32, verbose=1,
                            callbacks=[lr_callback])

        # Freeze all layers except the last few
        for layer in resnet_model.layers:
            layer.trainable = True

        # Train the model with the learning rate scheduler
        history = model.fit(train_images, train_labels,
                            validation_data=(val_images, val_labels),
                            epochs=1, batch_size=32, verbose=1,
                            callbacks=[lr_callback])

        # Save the model weight
        model.save_weights(train_output_dir +
                           "ai_human_classify_model_weights_resnet152.h5")


def extract_features(images, model):
    """ Extract features from a batch of images using the provided model. """
    features = []
    for img in images:  # Ensure image is resized to match model input
        img_array = tf.keras.utils.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)
        processed_img = tf.keras.applications.resnet50.preprocess_input(
            img_array_expanded)
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

    main(train_data_image_dir, train_data_labels_csv,
         target_column_name, trained_model_output_dir)

########################################################################################################################
