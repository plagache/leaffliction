import argparse
import numpy as np
from PIL import Image
from utils import DatasetFolder, Dataloader
import time

from tinygrad import Tensor, TinyJit, nn, GlobalCounters
from tinygrad.helpers import getenv, colored, trange
from tinygrad.nn.state import get_parameters
from tinygrad.nn import optim
from tinygrad.nn.state import safe_save, get_state_dict
from tiny_utils import train, evaluate


class ComposeTransforms:
    def __init__(self, trans):
        self.trans = trans

    def __call__(self, x):
        for t in self.trans:
            x = t(x)
        return x


class AlexNet:
    def __init__(self, num_classes):
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.conv1(x).relu().max_pool2d((3, 3), 2)
        x = self.conv2(x).relu().max_pool2d((3, 3), 2)
        x = self.conv3(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu().max_pool2d((3, 3), 2)
        # x = x.flatten(1)
        # x = self.fc1(x.dropout(0.5)).relu()
        x = self.fc1(x.flatten(1).dropout(0.5)).relu()
        x = self.fc2(x.dropout(0.5)).relu()
        return self.fc3(x)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train our model on a dataset from a train and test directory")
    parser.add_argument("train_directory", help="the train set directory to parse")
    parser.add_argument("test_directory", help="the validation set directory to parse")
    args = parser.parse_args()

    # TESTING USING ONLY VALIDATION SET
    train_folder: DatasetFolder = DatasetFolder(args.test_directory)
    # test_folder: DatasetFolder = DatasetFolder(args.test_directory)

    loader: Dataloader = Dataloader(train_folder, 1000)
    # test_loader: Dataloader = Dataloader(test_folder, 100, shuffle=True)

    X_train, Y_train = loader.get_tensor(new_size=(227,227))

    # X_test, Y_test = test_loader.get_tensor(new_size=(227,227))

    classes = len(train_folder.classes)

    model = AlexNet(num_classes=classes)
    opt = nn.optim.Adam(nn.state.get_parameters(model))
    # model = ResNet(18, num_classes=classes)
    # TRANSFER = getenv('TRANSFER')
    # if TRANSFER:
    #     model.load_from_pretrained()

    @TinyJit
    @Tensor.train()
    def train_step(samples) -> Tensor:
        opt.zero_grad()
        # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
        loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
        opt.step()
        return loss

    @TinyJit
    @Tensor.test()
    def get_test_acc(samples) -> Tensor: return (model(X_train[samples]).argmax(axis=1) == Y_train[samples]).mean()*100

    test_acc = float('nan')
    for i in (t:=trange(getenv("STEPS", 140))):
        GlobalCounters.reset()   # NOTE: this makes it nice for DEBUG=2 timing
        samples = Tensor.randint(getenv("BS", 32), high=X_train.shape[0])
        loss = train_step(samples)
        if i%10 == 9: test_acc = get_test_acc(samples).item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    # verify eval acc
    if target := getenv("TARGET_EVAL_ACC_PCT", 0.0):
        if test_acc >= target and test_acc != 100.0: print(colored(f"{test_acc=} >= {target}", "green"))
        else: raise ValueError(colored(f"{test_acc=} < {target}", "red"))

    state_dict = get_state_dict(model)
    model_name = f"{model.__class__.__name__}.safetensor"
    safe_save(state_dict, model_name)
