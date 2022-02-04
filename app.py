from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import argparse
from joblib import load
from flask_cors import CORS


labels = load('labels.joblib')

model_name = 'xray_model'


def binary_balanced_accuracy(y_true, y_pred):
    y_true = y_true.numpy()
    y_pred = y_pred.numpy()

    y_true = y_true.ravel()
    y_pred = np.round(y_pred.ravel())
    num_classes = len(np.unique(y_true))

    cm = confusion_matrix(y_true, y_pred).T
    balanced_accuracy = 0
    for i in range(num_classes):
        num = cm[i, i]
        den = np.sum(cm[:, i])
        if num == 0:
            acc = 0
        else:
            acc = num / den
        balanced_accuracy += acc

    return (balanced_accuracy / num_classes)


model = tf.keras.models.load_model(model_name, custom_objects={
                                   'binary_balanced_accuracy': binary_balanced_accuracy})

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET', 'POST'])
def pneumonia():
    if request.method == 'GET':
        return jsonify({'message': 'Upload X-ray Image ',
                        'Key value for uploading file': 'file'
                        'This key value takes Xray image as input image'})

    elif request.method == 'POST':
        img = request.files['file'].read()

        x = tf.io.decode_image(img)
        try:
            x = tf.image.rgb_to_grayscale(x)
        except:
            pass
        x = tf.image.resize(x, [150, 150])
        x = x / 255.0
        x = tf.expand_dims(x, axis=0)
        cls = model.predict(x)
        cls = np.round(cls.ravel())
        text = 'Predicted Class : ' + str(labels[int(cls[0])])
        return jsonify({'message': text})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=8080)
    args = parser.parse_args()
    app.run(debug=True, port=args.port)
