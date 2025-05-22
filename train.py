import json
from pathlib import Path
import argparse
from sample import random_split, copy_dataset
import shutil

from utils import DatasetFolder
import torch.nn as nn
import torch.nn.functional as F
from fastai.learner import Learner
from fastai.optimizer import OptimWrapper
from fastai.vision.all import ImageDataLoaders, Resize
from fastai.metrics import accuracy
from fastai.layers import partial
from torch import optim


class SmallModel(nn.Module):
    def __init__(self, classes):
        super(SmallModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(0.5),
            nn.Linear(64 * 54 * 54, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self, classes):
        # describe the operation
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1)
        # self.pool = nn.MaxPool2d(2, 2)
        self.pool = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Linear(9216, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, classes)
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(0.5)
        # self.fc = nn.Linear(64 * 62 * 62, num_of_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def prepare_dataset(directory):
    dataset_name = directory.name + "_dataset"
    dataset_path = Path(dataset_name)

    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        print(f"Error: the '{dataset_path}' file already exists.")
        exit(0)
    else:
        print(f"Successfully made the '{dataset_path}' directory.")

    tmp_path = dataset_path / 'tmp'
    train_path = dataset_path / "train"
    validation_path = dataset_path / "validation"

    source: DatasetFolder = DatasetFolder(directory)

    train_dataset_exists_and_balanced = train_path.is_dir() and DatasetFolder(train_path).is_balanced()
    validation_dataset_exists_and_balanced = validation_path.is_dir() and DatasetFolder(validation_path).is_balanced()

    if train_dataset_exists_and_balanced and validation_dataset_exists_and_balanced:
        print("Train and validation datasets already exist and are balanced. Skipping recreation.")
    else:
        if train_path.is_dir():
            print(f"Removing existing unbalanced train directory: {train_path}")
            shutil.rmtree(train_path)
        if validation_path.is_dir():
            print(f"Removing existing unbalanced validation directory: {validation_path}")
            shutil.rmtree(validation_path)

        source.to_images()
        source.augment_images()
        source.balance_dataset(tmp_path)

        dataset_tmp: DatasetFolder = DatasetFolder(tmp_path)

        train, validation = random_split(dataset_tmp, 0.2)

        copy_dataset(dataset_tmp, train, train_path)
        copy_dataset(dataset_tmp, validation, validation_path)
        shutil.rmtree(tmp_path)

    print("Dataset preparation complete.")
    return dataset_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("directory", help="path to a directory with images to classify")
    args = parser.parse_args()

    directory = Path(args.directory)
    if directory.is_dir() is False:
        parser.error("Path provided doesn't exist")
        parser.print_help()

    dataset_path = prepare_dataset(directory)

    dls = ImageDataLoaders.from_folder(dataset_path, train="train", valid="validation", item_tfms=Resize(227))
    # dls.show_batch(max_n=6)

    total_items = len(dls.train_ds)
    batch_size = dls.bs
    total_batches = len(dls.train)
    print(f"number of items: {total_items}, batch_size: {batch_size}, number of batches: {total_batches}")
    print(dls.vocab)
    print(dls.vocab.o2i)
    print(dls.after_item, dls.after_batch)
    # 40k images is not optimal for training

    opt_func = partial(OptimWrapper, opt=optim.Adam)

    classes = list(dls.vocab)
    model = AlexNet(len(classes))
    # print(model)
    # print(type(model))

    # model = SmallModel()
    criterion = nn.CrossEntropyLoss()
    learn = Learner(dls, model, loss_func=criterion, opt_func=opt_func, metrics=accuracy)

    # learn = vision_learner(dls, resnet18, metrics=accuracy)

    # print(dls.device)
    print(learn.model)
    print(type(learn.model))

    epoch = 10
    suggested_learning_rate = learn.lr_find()
    optimal_lr = suggested_learning_rate.valley
    print(f"\nOptimal learning rate: {optimal_lr}\n")
    learn.fine_tune(epoch, base_lr=optimal_lr)
    # learn.fine_tune(epoch)

    results = learn.validate()
    print(f"Validation accuracy: {results[1]:.2f}")
    # print(f"Validation accuracy: {results}")
    # for image_path in dls.valid_ds.items:
        # image_path = Path(image)
        # prediction = predict_image(learn, image_path)
        # print(f"Image: {image_path}, Predicted: {prediction}")

    model_name = f"models/{model.__class__.__name__}-{dataset_path}-Epch:{epoch}-Acc:{results[1]*100:.0f}"
    print(f"model name for saving: {model_name}")

    classes_outfile = Path(f"{model_name}.json")
    if classes_outfile.is_file():
        with open(classes_outfile, 'r') as f:
            loaded_classes = json.load(f)
            if classes != loaded_classes:
                classes_outfile = Path(f"{model_name}-classes.json")
                print(f"json file differs from current classes saving in {classes_outfile}")
                with open(classes_outfile, 'w') as f:
                    json.dump(classes, f)
    else:
        with open(classes_outfile, 'w') as f:
            json.dump(classes, f)

    learn.save_model(f"{model_name}.pth", learn.model, None)
