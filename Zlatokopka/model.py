import torch
import torch.nn as nn
from torch.optim import adam
from torchvision import transforms
from typing import Union, Optional, List, Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
from dataclasses import dataclass
import enum
from tqdm import tqdm 
from datetime import datetime
import os
import pathlib
import cv2 as cv


class activations(enum.Enum):
    RELU = "RELU"
    LEAKY_RELU = "LEAKY_RELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"

@dataclass
class HParams:
    convCount: int = 1
    kernelSize: int = 1
    convChannels: int = 1
    fullyConnectedCount: int = 1
    poolingInterval: int = 1
    poolingSize: int = 2
    fullyConnectedSize: int = 10
    fullyConnectedSizeDecay: float = 1.0
    dropout: float = 0.0    
    regularization: float = 0.0
    learningRate: float = 0.1
    activation: activations = activations.RELU
    batchSize: int = 64

    def __str__(self) -> str:
        return ";".join([
            str(self.convCount),
            str(self.kernelSize),
            str(self.convChannels),
            str(self.fullyConnectedCount),
            str(self.poolingInterval),
            str(self.poolingSize),
            str(self.fullyConnectedSize),
            str(self.fullyConnectedSizeDecay),
            str(self.dropout),
            str(self.regularization),
            str(self.learningRate),
            str(self.activation.value),
            str(self.batchSize),
        ])
    
    @staticmethod
    def parseFromString(str: str) -> "HParams":
        params = str.split(";")
        return HParams(
            convCount = int(params[0]),
            kernelSize = int(params[1]),
            convChannels = int(params[2]),
            fullyConnectedCount = int(params[3]),
            poolingInterval = int(params[4]),
            poolingSize = int(params[5]),
            fullyConnectedSize = int(params[6]),
            fullyConnectedSizeDecay = float(params[7]),
            dropout = float(params[8]),
            regularization = float(params[9]),
            learningRate = float(params[10]),
            activation = activations[params[11]],
            batchSize = int(params[12]),
        )

    

class FashionModel:
    def __init__(self, hparams: HParams, classes: List[str], imageSize: Tuple[int, int] = (1440, 1080), device: Optional[torch.device] = None):
        self._device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self._device == "cpu":
            print("VAROVÁNÍ: Model běží na CPU! Vše funguje stejně, ale trénování takového modelu může být výrazně pomalejší než při použití GPU. Pokud vaše GPU nepodporuje CUDA development, doporučuje se použití externích výpořetních prostředků, jako je například Google Colab.")
        self._model = Model(hparams, imageSize, len(classes), self._device)
        self._model.to(self._device)
        self._batchSize = hparams.batchSize
        self.hparams = hparams
        self.imageSize = imageSize
        self.classes = classes

    def predict(self, image: Union[torch.Tensor, np.ndarray]) -> str:
        if isinstance(image, np.ndarray):
            return self.predictBatch([image])[0]
        return self.predictBatch(image.reshape((1, image.shape[0], image.shape[1], image.shape[2])))[0]

    def predictBatch(self, images: Union[torch.Tensor, List[torch.Tensor], List[np.ndarray]]) -> List[str]:
        if isinstance(images, list):
            if isinstance(images[0], np.ndarray):
                images = torch.stack([transforms.ToTensor()(cv.resize(i, self.imageSize, interpolation=cv.INTER_LINEAR)) for i in images])
            else:
                images = torch.stack(images)
        images = images.to(self._device)
        self._model.eval()
        res = self._model(images)
        indexes = torch.argmax(res, dim=1)
        return [self.classes[i.item()] for i in indexes]

    def train(self, dataset: Dataset, epochs: Optional[int] = None, showEpochProgressBar: bool = True, showTotalProgressBar: bool = False) -> List[float]:
        if epochs is None or epochs < 1:
            epochs = 1
        res = []
        for _ in (tqdm(range(epochs)) if epochs > 1 and showTotalProgressBar else range(epochs)):
            loader = DataLoader(dataset, batch_size=self._batchSize, shuffle=True)
            totalLoss = 0.0
            for batch in (tqdm(loader) if showEpochProgressBar else loader):
                totalLoss += self._model.trainingStep(batch["img"], batch["label"]) * len(batch["img"])
            if showEpochProgressBar:
                print(f"Loss: {totalLoss / len(dataset)}")
        res.append(totalLoss / len(dataset))          


    def evaluate(self, dataset: Dataset) -> Tuple[float, float]:
        loader = DataLoader(dataset, batch_size=self._batchSize, shuffle=True)
        totalLoss = 0.0
        totalAccuracy = 0.0
        for batch in loader:
            l, a = self._model.evaluationStep(batch["img"], batch["label"])
            totalLoss += l * len(batch["img"])
            totalAccuracy += a * len(batch["img"])
        return (totalLoss / len(dataset), totalAccuracy / len(dataset))

    def save(self, name: Optional[str] = None, path: Optional[str] = None) -> str:
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if path is None:
            path = "models"
        if name is None:
            name = str(datetime.now().timestamp())
        finalPath = os.path.join(root, path, name)
        while os.path.exists(finalPath):
            finalPath += "_" + str(datetime.now().timestamp())
        pathlib.Path(finalPath).mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), os.path.join(finalPath, "model.pt"))
        with open(os.path.join(finalPath, "params.csv"), "w") as f:
            f.write(str(self.hparams) + "\n" + str(self.imageSize[0]) + ";" + str(self.imageSize[1]) + "\n" + ";".join(self.classes))
        return finalPath
        

    @staticmethod
    def load(name: str, path: Optional[str] = None, device: Optional[torch.device] = None) -> 'FashionModel':
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        if path is None:
            path = "models"
        if name is None:
            name = str(datetime.now().timestamp())
        finalPath = os.path.join(root, path, name)

        rawParams = []
        with open(os.path.join(finalPath, "params.csv"), "r") as f:
            rawParams = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        hParams = HParams.parseFromString(rawParams[0])
        imageWidthRaw, imageHeightRaw = tuple(rawParams[1].split(";"))
        res = FashionModel(hParams, rawParams[2].split(";"), (int(imageWidthRaw), int(imageHeightRaw)), device)
        res._model.load_state_dict(torch.load(os.path.join(finalPath, "model.pt")))
        return res
    

def _getWeightInitGain(hParams: HParams) -> float:
    return nn.init.calculate_gain('relu') if activations.RELU == hParams.activation else nn.init.calculate_gain("leaky_relu", 0.01) if activations.LEAKY_RELU == hParams.activation else nn.init.calculate_gain("sigmoid") if activations.SIGMOID == hParams.activation else nn.init.calculate_gain("tanh")

class Model(nn.Module):
    def __init__(self, hParams: HParams, inputSize: Tuple[int, int], outputSize: int, device: torch.device):
        super().__init__()

        self.device = device
        
        convolutionalPart = []
        size = inputSize
        for i in range(hParams.convCount):
            convolutionalPart.append(
                nn.Conv2d(
                    in_channels=3 if i == 0 else hParams.convChannels,
                    out_channels=hParams.convChannels,
                    kernel_size=hParams.kernelSize,
                    padding=hParams.kernelSize // 2,
                    stride=1
                )
            )
            nn.init.xavier_normal_(convolutionalPart[-1].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(convolutionalPart[-1].bias)
            convolutionalPart.append(nn.ReLU())
            if i % hParams.poolingInterval == 0 and i != 0:
                convolutionalPart.append(nn.MaxPool2d(kernel_size=hParams.poolingSize, stride=hParams.poolingSize))
                size = (size[0] // hParams.poolingSize, size[1] // hParams.poolingSize)
            convolutionalPart.append(nn.ReLU())

        fullyConnectedPart = []
        decayFactor = 1.0
        for i in range(hParams.fullyConnectedCount):
            if i == 0:
                fullyConnectedPart.append(nn.Linear(size[0] * size[1] * hParams.convChannels, hParams.fullyConnectedSize))
            else:
                fullyConnectedPart.append(nn.Linear(round(hParams.fullyConnectedSize * decayFactor), round(hParams.fullyConnectedSize * decayFactor * hParams.fullyConnectedSizeDecay)))
                decayFactor *= hParams.fullyConnectedSizeDecay
            nn.init.xavier_normal_(fullyConnectedPart[-1].weight, gain=_getWeightInitGain(hParams))
            nn.init.zeros_(fullyConnectedPart[-1].bias)
            fullyConnectedPart.append(
                nn.ReLU() if hParams.activation == activations.RELU else 
                nn.LeakyReLU() if hParams.activation == activations.LEAKY_RELU else
                nn.Sigmoid() if hParams.activation == activations.SIGMOID else
                nn.Tanh()
            )
            fullyConnectedPart.append(nn.Dropout(hParams.dropout))

        self.model = nn.Sequential(
            *convolutionalPart, nn.Flatten(), *fullyConnectedPart, nn.Linear(round(hParams.fullyConnectedSize * decayFactor), outputSize)
        )
        self.optimizer = adam.Adam(self.model.parameters(), lr=hParams.learningRate, weight_decay=hParams.regularization)
        self.lossFunction = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def evaluationStep(self, batch: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
        self.eval()
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        res = self.forward(batch)
        loss = self.lossFunction(res, labels)
        correct = torch.argmax(res, dim=1) == torch.argmax(labels, dim=1)
        accuracy = torch.mean(correct.float())
        return (loss.item(), accuracy.item())
    
    def trainingStep(self, batch: torch.Tensor, labels: torch.Tensor) -> float:
        self.train()
        self.optimizer.zero_grad()
        batch = batch.to(self.device)
        labels = labels.to(self.device)

        res = self.forward(batch)
        loss = self.lossFunction(res, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    