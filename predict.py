import argparse
import os
from typing import Any

import tensorflow as tf
import pandas as pd
from PIL import Image
from train import load_train_resources, extract_features
from common import load_model, load_predict_image_names, load_single_image


########################################################################################################################

def parse_args():
    """
    Helper function to parse command line arguments
    :return: args object
    """
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--predict_data_image_dir', required=True, help='Path to image data directory')
    parser.add_argument('-l', '--predict_image_list', required=True,
                        help='Path to text file listing file names within predict_data_image_dir')
    parser.add_argument('-t', '--target_column_name', required=True,
                        help='Name of column to write prediction when generating output CSV')
    parser.add_argument('-m', '--trained_model_dir', required=True,
                        help='Path to directory containing the model to use to generate predictions')
    parser.add_argument('-o', '--predicts_output_csv', required=True, help='Path to CSV where to write the predictions')
    parser.add_argument('-c', '--case', required=True, choices=['Is Epic', 'Needs Respray', 'Is GenAI'], help='Specify the case for the model training and prediction')
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
         predicts_output_csv: str,
         case_type: str):
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
    if case_type == "Is Epic" :
        # load pre-trained models or resources at this stage.
        resnet_model = load_train_resources(case_type).get('resnet50_model')

        # Load in the image list
        image_list_file = os.path.join(predict_data_image_dir, predict_image_list)
        image_filenames = load_predict_image_names(image_list_file)
        predict_images = [tf.keras.utils.load_img((os.path.join(predict_data_image_dir, image)), target_size=(224, 224)) for image in image_filenames]
        features = extract_features(predict_images, resnet_model)
        model = tf.keras.models.load_model(os.path.join(trained_model_dir, target_column_name))
        # Iterate through the image list to generate predictions
        predictions = model.predict(features)
        predictions = ['Yes' if pred == 1 else 'No' for pred in predictions]
        df_predictions = pd.DataFrame({'Filenames': image_filenames, target_column_name: predictions})

        # Finally, write out the predictions to CSV
        df_predictions.to_csv(predicts_output_csv, index=False)


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
    case_type = args.case
    main(predict_data_image_dir, predict_image_list, target_column_name, trained_model_dir, predicts_output_csv, case_type)

########################################################################################################################
