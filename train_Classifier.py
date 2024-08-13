import os
import tarfile
from tqdm import tqdm
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen

import torch
import torch.optim as optim
from torchsummaryX import summary

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, ReduceLROnPlateau

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from model import Network
from model import ChessNetwork
from configs import ModelConfigs
import pandas as pd

df_train = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/Augmentation_Ground_Truth_2.csv").values.tolist()
df_test = pd.read_csv("C:/Users/ericz/Desktop/IAM Model/val.csv").values.tolist()

vocab = ['#', '+', '-', '1', '2', '3', '4', '5', '6', '7', '8', '=', 'B', 'K', 'N', 'O', 'P', 'Q', 'R', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'x']
max_len =6

# Save vocab and maximum text length to configs
configs = ModelConfigs()
configs.vocab = "".join(sorted(vocab))
configs.max_text_length = max_len
configs.save()

train_dataProvider= DataProvider(
    dataset = df_train,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

test_dataProvider = DataProvider(
    dataset = df_test,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        # ImageShowCV2(), # uncomment to show images when iterating over the data provider
        ImageResizer(configs.width, configs.height, keep_aspect_ratio=False),
        LabelIndexer(configs.vocab),
        LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    use_cache=True,
)

# Augment training data with random brightness, rotation and erode/dilate
train_dataProvider.augmentors = [
    RandomBrightness(), 
    RandomErodeDilate(),
    RandomSharpen(),
    ]

# NOTE: Don't need to change any of the code except for the different leraning rates
pre_trained_network = Network(78, activation="leaky_relu", dropout=0.3)
weights_path = "C:/Users/ericz/Desktop/IAM Model/model.pt" # model weights for IAM Model
pre_trained_network.load_state_dict(torch.load(weights_path))

# Create a chess model that takes in the parameters of the pretrained model layers but has different output shape
chess_network = ChessNetwork(num_chars = len(configs.vocab), preTrained = pre_trained_network)

# Debugging Code to see if all the other layers are the same:
# for x in range(len(pre_trained_network.pretrained)):
#     print (pre_trained_network.pretrained[x] == chess_network.pretrained[x])

# feature extraction is when you freeze all the layers
chess_network.fineTune(feature_extract=True) # Turn on gradient tracking (True = freeze the layers)
# Debugging Code to see if gradient descent is off:
# for x in pre_trained_network.pretrained:
#     print(f"Parameters: \n {list(x.parameters())}")
#     for p in x.parameters():
#         print(f"grad: {p.requires_grad}")

loss = CTCLoss(blank=len(configs.vocab))
optimizer = optim.Adam(chess_network.parameters(), lr=configs.learning_rate)

# create callbacks
earlyStopping = EarlyStopping(monitor="val_CER", patience=20, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
tb_callback = TensorBoard(configs.model_path + "/logs")
reduce_lr = ReduceLROnPlateau(monitor="val_CER", factor=0.9, patience=10, verbose=1, mode="min", min_lr=1e-6)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.height, configs.width, 3), 
    verbose=1,
    metadata={"vocab": configs.vocab}
    )

model = Model(chess_network, optimizer, loss, metrics=[CERMetric(configs.vocab), WERMetric(configs.vocab)])

# train the chess_network model
model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=configs.train_epochs, 
    callbacks=[earlyStopping, modelCheckpoint, tb_callback, reduce_lr, model2onnx]
    )

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))