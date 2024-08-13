import cv2
import typing
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn

import os
from urllib.request import urlopen
import tarfile
from io import BytesIO
from zipfile import ZipFile

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = '!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    def predict(self, image: np.ndarray):
        # plt.title("Original Image")
        # plt.imshow(image)
        # plt.show()
        try:
            image = cv2.resize(image, self.input_shape[:2][::-1])
        except:
            print("ERROR")
            return

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        # print(f"shape of image_pred {image_pred.shape}")

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(preds, self.vocab)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/GoldenChild/model.onnx")

    df = pd.read_csv("C:/Users/ericz/OneDrive/Desktop/IAM Model/validationSet.csv").values.tolist()
    #df[0] is the file path and df[1] is the label
    accum_cer = []
    accum_wer = []
    accum_loss = []
    # ctc_loss = nn.CTCLoss(blank=78)

    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        
        prediction_text = model.predict(image)
        
        try:
            cer = get_cer(prediction_text, label)
            # print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            wer = get_wer(prediction_text, label)
            # print(f"WER: {wer}")
            # loss = ctc_loss(prediction_text, label)

            accum_cer.append(cer)
            accum_wer.append(wer)
            # accum_loss.append(loss)

        except: 
            print()
        
    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
    #Loss: {np.average(accum_loss)}
