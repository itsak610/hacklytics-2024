
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from numpy import * 
import numpy as np
import pandas as pd 


model = load_model('./facenet_keras.h5')      


def get_embedding(model, face_pixels):               
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def extract_face(filename, required_size=(160, 160)): 
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


def verfication(patients, user_input_url):
    user_input = extract_face(user_input_url)
    test_user = get_embedding(model, user_input)

    distances = np.linalg.norm(test_user - patients, axis=1)
    min_dist_index = np.argmin(distances)

    return min_dist_index


def register(input_url, patients):
    new_face = extract_face(input_url)
    new_pat = get_embedding(model, new_face)
    patients.append(new_pat)

originalface = extract_face('./samplefaces/shot1.jpg')
testface     = extract_face('./samplefaces/shot2.jpeg')

originalembedding = get_embedding(model,originalface)    
testembedding = get_embedding(model,testface)

dist = linalg.norm(testembedding-originalembedding)    