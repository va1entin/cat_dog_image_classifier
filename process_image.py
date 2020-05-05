import cv2
import keras
import numpy as np


px_rows = 150
px_columns = 150

path = "ml/model_keras.h5"
model = keras.models.load_model(path)
model._make_predict_function()

def preprocess_image(image_file):
    read_image = cv2.resize(cv2.imread(image_file, cv2.IMREAD_COLOR), (px_rows, px_columns), interpolation=cv2.INTER_CUBIC)
    preprocessed_image = np.expand_dims(read_image, axis=0)
    return preprocessed_image

def predict(model, preprocessed_image):
    numeric_prediction = model.predict(preprocessed_image)
    if numeric_prediction > 0.5:
        return "dog"
    else:
        return "cat"

def process_image(image_file_name):
    preprocessed_image = preprocess_image(image_file_name)
    prediction = predict(model, preprocessed_image)
    return prediction
