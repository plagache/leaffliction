import argparse

import numpy as np
from PIL import Image
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters
from tinygrad.tensor import Tensor

from resnet import ResNet
from tiny_utils import evaluate, train
from utils import Dataloader, DatasetFolder


class ComposeTransforms:
  def __init__(self, trans):
    self.trans = trans

  def __call__(self, x):
    for t in self.trans:
      x = t(x)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train our model on a dataset from a train and test directory")
    parser.add_argument("train_directory",
                        help="the train set directory to parse")
    parser.add_argument(
        "test_directory", help="the validation set directory to parse")
    args = parser.parse_args()

    train_folder: DatasetFolder = DatasetFolder(args.train_directory)
    test_folder: DatasetFolder = DatasetFolder(args.test_directory)
    print("1")

    train_folder.to_numpy()
    print(train_folder)
    X_train, Y_train = zip(*train_folder)
    print(X_train.shape)
    print(Y_train.shape)

    # loader: Dataloader = Dataloader(train_folder, 1000)
    # test_loader: Dataloader = Dataloader(test_folder, 100, shuffle=True)
    # print("2")
    # X_train, Y_train = loader.get_tensor()
    # print("3")
    # X_test, Y_test = test_loader.get_tensor()
    # print("4")

    model = ResNet(18, len(train_folder.classes))
    model.load_from_pretrained()
    # print(f"this is a format string non the less")
    # acc = (model(X_test).argmax(axis=1) == Y_test).mean()

    # print(acc.item())  # ~10% accuracy, as expected from a random model

    # print(f"and this is also a format string non the less")

    transform = ComposeTransforms([
        # lambda x: [Image.fromarray(xx, mode='L').resize((64, 64)) for xx in x],
        # lambda x: np.stack([np.asarray(xx) for xx in x], 0),
        lambda x: x / 255.0,
        # lambda x: np.tile(np.expand_dims(x, 1), (1, 3, 1, 1)).astype(np.float32),
    ])

    classes = len(train_folder.classes)

    samp = np.random.randint(0, X_train.shape[0], size=(1))
    x = Tensor(transform(X_train[samp]), requires_grad=False)
    y = Tensor(transform(Y_train[samp]))
    print("x shape", x.numpy().shape, X_train.shape)
    print("y shape", y.numpy().shape)
    exit(0)


    # lr = 5e-3
    # for _ in range(5):
    #     optimizer = optim.SGD(get_parameters(model), lr=lr, momentum=0.9)
    #     train(model, X_train, Y_train, optimizer, 100, BS=1, transform=transform)
    #     evaluate(model, X_test, Y_test, num_classes=classes, transform=transform)
    #     lr /= 1.2
    #     print(f'reducing lr to {lr:.7f}')
