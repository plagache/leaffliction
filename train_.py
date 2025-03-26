import argparse
import numpy as np
from PIL import Image
from resnet import ResNet
from utils import DatasetFolder, Dataloader

from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.helpers import getenv
from tiny_utils import train, evaluate


class ComposeTransforms:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        for t in self.trans:
            x = t(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train our model on a dataset from a train and test directory")
    parser.add_argument("train_directory", help="the train set directory to parse")
    parser.add_argument("test_directory", help="the validation set directory to parse")
    args = parser.parse_args()

    train_folder: DatasetFolder = DatasetFolder(args.train_directory)
    test_folder: DatasetFolder = DatasetFolder(args.test_directory)
    # print(folder.mapped_dictionnary)
    # print(folder.count_dictionnary)
    # print(folder.indices_dictionnary)
    loader: Dataloader = Dataloader(train_folder, 1000)
    test_loader: Dataloader = Dataloader(test_folder, 100, shuffle=True)
    X_train, Y_train = loader.get_tensor()
    X_test, Y_test = test_loader.get_tensor()
    classes = len(train_folder.classes)

    model = ResNet(getenv("NUM", 18), num_classes=classes)
    model.load_from_pretrained()

    lr = 5e-3
    transform = ComposeTransforms([
        lambda x: [Image.fromarray(xx, mode="RGB").resize((224, 224)) for xx in x],
        lambda x: [np.asarray(xx).reshape(3, 224, 224).astype(np.float16) for xx in x],
        lambda x: np.stack(x, 0),
        lambda x: x / 255.0,
    ])

    for _ in range(2):
        optimizer = optim.SGD(get_parameters(model), lr=lr, momentum=0.9)
        train(model, X_train, Y_train, optimizer, 100, BS=32, transform=transform)
        evaluate(model, X_test, Y_test, num_classes=classes, transform=transform)
        lr /= 1.2
        print(f"reducing lr to {lr:.7f}")
