import cv2
import typing
import numpy as np
import matplotlib.pyplot as plt

import os
from urllib.request import urlopen
import tarfile
from io import BytesIO
from zipfile import ZipFile

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
# from mltu.utils.text_utils import ctc_decoder_with_constraints

# most updated version for inference
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = ['#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'B', 'K', 'N', 'O', 'P', 'Q', 'R', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']

    def predict(self, image: np.ndarray):
        image = cv2.resize(image, self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(preds, self.vocab)[0]
        # text = ctc_decoder_with_constraints(preds, self.vocab)[0]

        return text
    
    def chooseDataset(self, type: int):
        if type ==1: # val 
            df = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/val.csv").values.tolist() 
        if type==2: # test data
            df = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/test.csv").values.tolist()
        if type==3: # train data
            df = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/Augmentation_Ground_Truth_2.csv").values.tolist()
        if type == 4: # mouseClikcer data
            df = pd.read_csv("C:/Users/ericz/Mouse Clicker/groundTruth1.csv").values.tolist() 
        return df
    
    @staticmethod
    def chooseModel(type: int):
        if type==1: # control, base-line model
            model = ImageToWordModel(model_path="C:/Users/ericz/Desktop/IAM Model/Models/08_handwriting_recognition_torch/MajidParitions - Chess Trained/model.onnx")
        if type==2: # fine-tuned model
            model = ImageToWordModel(model_path="C:/Users/ericz/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Augment V11/model.onnx")
        return model
            
if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    ### Compare results of just training model on chess data vs fine-tuning an IAM pretrained model ###
    accum_cer = []
    accum_wer = []

    model = ImageToWordModel.chooseModel(type = 2) # 1 = control model, 2 = fine-tuned model (may need to change name of model)
    df = model.chooseDataset(type = 1) # 1 = val, 2 = test, 3 = train, 4 = mouseClicker
    # invalid_moves_path = "C:/Users/ericz/Mouse Clicker/invalidMoves.csv"

    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)
        prediction_text = model.predict(image)
        label = label.strip()
        
        try:
            cer = get_cer(prediction_text, label)
            cer = round(cer, 2)
            # print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}") # used to print all values

            # use the bottom code to verify varient checker
            # if cer !=0:
            #     print(f"Label: {label}, Prediction: {prediction_text}, CER: {cer}")
                
                # value = image_path + "," + label
                # print(f"my Prediction: {prediction_text}, path: {value}")
                # with open(invalid_moves_path, "a") as f:
                #     f.write()

            wer = get_wer(prediction_text, label)

            accum_cer.append(cer)
            accum_wer.append(wer)

        except: 
            print("ERROR")
        
    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
