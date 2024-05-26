import os
import shutil
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf
from PIL import Image
import pandas as pd
from tensorflow import keras
from keras.layers import Dense, Input, InputLayer, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.utils import load_img
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score


def modlod(data_x, y_data):
    cnnmodel = load_model('model.h5')
    cnnmodel.evaluate(x=tf.cast(np.array(data_x), tf.float64), 
                    y=tf.cast(list(map(int, y_data)), tf.int32), batch_size=32)

    data_pred = cnnmodel.predict(data_x)
    data_pred = np.argmax(data_pred, axis=1)

    print(classification_report(data_pred, y_data))
    
    return data_pred


def gfc(patient_folder, images):
    for folders in os.listdir('./testPatient'): 
        if os.path.isdir(os.path.join('./testPatient', folders)):  
            patient_folder += 1

    patient_folder = patient_folder - 1

    for i in range(1, patient_folder+1):
        fp = './testPatient/Patient_{}/*thresh.png'.format(i)
        x=0
        for img in glob.glob(fp):
            x += 1
        images.append(x)

    return patient_folder, images


def create_results(images, data_pred):
    n = len(images) 
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity']
    preds = data_pred
    i = 1
    for j in range(0, n):
        t = images[j]
        Label = []
        Label.append(preds[0:t])
        preds = preds[t:]
        IC_Number = list(range(1, t+1))

        df = pd.DataFrame()
        df_data = {'IC_Number': IC_Number, 'Label': Label[0]}
        df = pd.DataFrame(df_data, columns = ['IC_Number', 'Label'])
        df.to_csv('./testPatient/Patient_{}/Results.csv'.format(i), index=False)

        pred_label = pd.read_csv('./testPatient/Patient_{}/Results.csv'.format(i))
        pred_label = pred_label['Label'].tolist()

        temp_label = pd.read_csv('./testPatient/newPatient_{}_Labels.csv'.format(i))
        temp_label = temp_label['Label'].tolist()
        
        mv = []

        lm = confusion_matrix(temp_label, pred_label)
        print(lm)

        accuracy = accuracy_score(temp_label, pred_label)
        precision = precision_score(temp_label, pred_label, average=None)
        spec = lm[0,0] / (lm[0,0] + lm[0,1])
        sensitivity = lm[1,1] / (lm[1,0] + lm[1,1])

        mv.append(accuracy)
        mv.append(precision[0])
        mv.append(spec)
        mv.append(sensitivity)

        df = pd.DataFrame()
        df_data = {'Metric': metrics, 'Score': mv}
        df = pd.DataFrame(df_data, columns = ['Metric', 'Score'])
        df.to_csv('./testPatient/Patient_{}/Metrics.csv'.format(i), index=False)

        i += 1