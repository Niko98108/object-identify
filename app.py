from flask import Flask, request
from flask_cors import CORS
import json
from keras.models import load_model as lm
import cv2
import numpy as np
import base64
from imageio import imread
import io
from PIL import Image

import Backend.color_detector as cd

app = Flask(__name__)
CORS(app)

part_02_model = lm('Backend/imageclassifier.h5')



@app.route('/obects', methods=['GET', 'POST'])
def part_02():
    if request.method == 'POST':
        imager = request.files['image']

        imager.save("upload_test/test_image.png")

        img = cv2.imread("upload_test/test_image.png")
    # cv2.imwrite('upload/base64.jpg', img)
    img = np.asarray(img)
    img = cv2.resize(img, (128, 128))
    img = preProcessing_2(img)
    img = img.reshape(1, 128, 128, 3)
    predict_x = part_02_model.predict(img)
    classes_x = np.argmax(predict_x, axis=1)

    dominat_color = cd.get_dominant_color('upload_test/test_image.png')


   # correct_class = 2
    if dominat_color == (244, 67, 54):
        correct_class = 0
    elif dominat_color == (255, 235, 59):
        correct_class = 1
    else:
        correct_class = 2

    if classes_x[0] == correct_class:
        return_str = '{ "object_code" : ' + str(classes_x) + ', "color_accuracy" : "correct"}'
    else:
        return_str = '{ "object_code" : ' + str(classes_x) + ', "color_accuracy" : "incorrect"}'

    return json.loads(return_str)


def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

def preProcessing_2(img):
    img = img / 255
    return img


def base_64_cv2(string_64):
    img = imread(io.BytesIO(base64.b64decode(string_64)))
    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return cv2_img


if __name__ == '__main__':
    app.run(debug=True)
