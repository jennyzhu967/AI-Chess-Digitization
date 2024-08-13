import cv2
import typing
import numpy as np

import os
from urllib.request import urlopen
import tarfile
from io import BytesIO
from zipfile import ZipFile

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer

# og code for inference model
class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab = get_vocab(dataset_path=os.path.join("Datasets", "IAM_Words"))
        print(self.vocab)

    def predict(self, image: np.ndarray):
        try:
            image = cv2.resize(image, self.input_shape[:2][::-1])
        except:
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, ///////////////")
            return

        image_pred = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: image_pred})[0]
        
        text = ctc_decoder(preds, self.vocab)[0]

        return text
    
def get_vocab(dataset_path):
    vocab = set()
    words = open(os.path.join(dataset_path, "words.txt"), "r").readlines()
    for line in tqdm(words):
        if line.startswith("#"):
            continue

        line_split = line.split(" ")
        if line_split[1] == "err":
            continue

        folder1 = line_split[0][:3]
        folder2 = "-".join(line_split[0].split("-")[:2])
        file_name = line_split[0] + ".png"
        label = line_split[-1].rstrip("\n")

        rel_path = os.path.join(dataset_path, "words", folder1, folder2, file_name)
        if not os.path.exists(rel_path):
            print(f"File not found: {rel_path}")
            continue

        vocab.update(list(label))
        vocab2 = "".join(sorted(vocab))
    return vocab2


if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = ImageToWordModel(model_path="C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Trained Model/model.onnx")

    df = pd.read_csv("C:/Users/ericz/OneDrive/Desktop/IAM Model/Models/08_handwriting_recognition_torch/Pre-Trained Model/val.csv").values.tolist()

    accum_cer = []
    for image_path, label in tqdm(df):
        image = cv2.imread(image_path)

        prediction_text = model.predict(image)
        
        cer = get_cer(prediction_text, label)
        # print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")
        
        accum_cer.append(cer)

    print(f"Average CER: {np.average(accum_cer)}")