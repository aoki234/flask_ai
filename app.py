from flask import Flask, redirect, request, jsonify,render_template
from keras import models
import numpy as np
from PIL import Image
import io
from keras.models import model_from_json
import tensorflow as tf
from time import sleep

app = Flask(__name__)
model = None
graph = tf.get_default_graph()
img = None
#global model, graph


def load_model():
    global model
    keras_model="Save_model.json"
    keras_param="Save_model.hdf5"
    model = model_from_json(open(keras_model).read())
    model.load_weights(keras_param)
    #model.summary()


@app.route('/')
def index():
    global img
    img = None
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.files and 'picfile' in request.files:
        global img
        img = request.files['picfile'].read()
        img = Image.open(io.BytesIO(img))
        img.save('./tmp/test.jpg')
        image = img.resize((50,50))
        image = np.asarray(image) / 255.0
        image = np.expand_dims(image, axis=0)
        global graph
        with graph.as_default():
            load_model()
            pred = model.predict(image)
        persons = [
            'ちょまど',
            '池澤あやか',
            '石原さとみ',
            '剛力彩芽'
        ]

        confidence = int(round(max(pred[0]), 3)*100)
        pred = persons[np.argmax(pred)]

        data = dict(pred=pred, confidence=str(confidence))
        return jsonify(data)

    return 'Picture info did not get saved.'


@app.route('/currentimage', methods=['GET'])
def current_image():
    if img:
        fileob = open('./tmp/test.jpg', 'rb')
        data = fileob.read()
        return data
    return None


if __name__ == '__main__':
    #load_model()
    app.run(threaded=True)
