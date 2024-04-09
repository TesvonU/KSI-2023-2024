import os
import cv2 as cv
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import random

from downloadUtils import downloadDataset



class BaseDataset(Dataset):
    def __init__(self, root: str, download_url: str = None, force_download: bool = False):
        self.root_path = root
        if download_url is not None:
            dataset_zip_name = download_url[download_url.rfind('/')+1:]
            self.dataset_zip_name = dataset_zip_name
            downloadDataset(
                url=download_url,
                data_dir=root,
                dataset_zip_name=dataset_zip_name,
                force_download=force_download,
            )


class FashionDataset(BaseDataset):
    def __init__(
            self, size: Tuple[int, int] = (1440, 1080), transform: Optional[callable] = None, cacheToRAM: bool = False, maxImagesPerClass: Optional[int] = None, 
            classes: Optional[List[str]] = None, forceDownload: bool = False, _extracted: bool = False):
        
        super().__init__(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "dataset"), "https://stastnyjakub.com/dataset.zip", forceDownload and not _extracted)
        if not _extracted:
            print("Creating dataset")
        self._transform =  transforms.ToTensor()
        self._cacheToRAM = cacheToRAM
        self._randomizedTransform = transform
        self._imageSize = size

        if not _extracted:
            self.labels = sorted(os.listdir(self.root_path))
            if classes is not None and len(classes) > 0:
                self.labels = [l for l in self.labels if l in classes]

            self._files = []
            for (i,l) in  enumerate(self.labels):
                for (j,f) in enumerate(os.listdir(os.path.join(self.root_path, l))):
                    if maxImagesPerClass is not None and j >= maxImagesPerClass:
                        break
                    lbl = [0.0] * len(self.labels)
                    lbl[i] = 1.0
                    self._files.append((os.path.join(self.root_path, l, f), torch.tensor(lbl)))
        
            self._cache = [None] * len(self._files)
            if self._cacheToRAM:
                print("caching images to RAM")
                for i, (file, _) in enumerate(tqdm(self._files)):
                    self._cache[i] = self._getImage(file)
            print("Dataset complete")

    @staticmethod
    def tensorToOpenCVImage(tensor: torch.Tensor) -> np.ndarray:
        return np.transpose(tensor.numpy(), (1, 2, 0))

    def _getImage(self, file: str) -> torch.Tensor:
        return self._transform(cv.resize(cv.imread(file), self._imageSize, interpolation=cv.INTER_LINEAR))

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            return None
        rawImage = self._cache[idx] if self._cacheToRAM else self._getImage(self._files[idx][0])
        return {
            "img": self._randomizedTransform(rawImage) if self._randomizedTransform is not None else rawImage,
            "label": self._files[idx][1],
        }        
    
    def extractSubset(self, percentage: float = 0.1, balanceClasses: bool = True, transform: Optional[callable] = None) -> 'FashionDataset':
        res = FashionDataset(size=self._imageSize, cacheToRAM=self._cacheToRAM, maxImagesPerClass=None, transform=transform, forceDownload=True, _extracted=True)
        indexes = []
        if balanceClasses:
            for index in range(len(self.labels)):
                indexesInClass = [i for i in range(len(self)) if torch.argmax(self._files[i][1]).item() == index]
                indexes += random.sample(indexesInClass, int(len(indexesInClass) * percentage))
            indexes = sorted(indexes)[::-1]
        else:
            indexes = sorted(random.sample(range(len(self)), int(len(self) * percentage)))[::-1]
        res.labels = self.labels

        res._files = []
        if res._cacheToRAM:
            res._cache = []
        for i in indexes:
            res._files.append(self._files[i])
            del self._files[i]
            if res._cacheToRAM:
                res._cache.append(self._cache[i])
                del self._cache[i]
        return res
