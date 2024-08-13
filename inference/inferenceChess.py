import cv2
import typing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import os
from urllib.request import urlopen
import tarfile
from io import BytesIO
from zipfile import ZipFile

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.torch.losses import CTCLoss

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = ['#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'B', 'K', 'N', 'O', 'P', 'Q', 'R', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']

    def predict(self, image: np.ndarray):
        try:
            image = cv2.resize(image, self.input_shape[:2][::-1])
        except: # Error comes from the folder
            print("ERROR")
            return 
        
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(preds, self.vocab)[0]

        return text

if __name__ == "__main__":

    model = ImageToWordModel(model_path="C:/Users/ericz/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Augmentation/Finetuned (0.002)/model.onnx")

    df = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/Models/08_handwriting_recognition_torch/V1 - ClassifierOnly (0.01)/val.csv").values.tolist()
    # df = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/Augmentation_Ground_Truth_9.csv").values.tolist()

    accum_cer = []
    accum_wer = []
    
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        label = label.strip()
        
        try:
            cer = get_cer(prediction_text, label)
            # print(f"Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            wer = get_wer(prediction_text, label)
            accum_cer.append(cer)
            accum_wer.append(wer)

        except: 
            print("Error")
        
    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
