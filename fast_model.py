from fastai.data.all import *
from fastai.vision.all import *
from fastai.learner import Learner
from fastai.optimizer import OptimWrapper
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

def predict_image(learner, image_path):
    test_dl = learner.dls.test_dl([image_path], with_labels=False)

    preds, _ = learner.get_preds(dl=test_dl)
    pred_class = preds.argmax(dim=1).item()

    return learner.dls.vocab[pred_class]

def verification(dls):
    for i in range(100):
        image, label = dls.train_ds[i]
        # print(image, label)
        label_name = dls.vocab[label]
        print(f"Decoded label: {label_name}, Encoded label: {label}")
        pil_image = PILImage.create(image)

        pred_class, pred_idx, outputs = learn.predict(pil_image)
        print(f"Predicted class: {pred_class}, Actual label: {label}")

class SmallModel(nn.Module):
    def __init__(self):
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
            nn.Linear(64 * 54 * 54, 8)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlexNet(nn.Module):
    def __init__(self):
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
        self.fc3 = nn.Linear(4096, 8)
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


if __name__ == "__main__":
    path = Path("images")

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        # item_tfms=Resize(227)
        item_tfms=Resize(224)
    ).dataloaders(path)
    # dls.show_batch(max_n=6)

    total_items = len(dls.train_ds)
    batch_size = dls.bs
    total_batches = len(dls.train)
    print(f"number of items: {total_items}, batch_size: {batch_size}, number of batches: {total_batches}")
    print(dls.vocab)
    print(dls.vocab.o2i)
    # print(dls.train.items)
    # print(dls.train_ds.items)
    # exit(0)
    # 40k images is not optimal for training


    opt_func = partial(OptimWrapper, opt=optim.Adam)

    # model = AlexNet()
    # print(model)
    # print(type(model))

    model = SmallModel()
    criterion = nn.CrossEntropyLoss()
    learn = Learner(dls, model, loss_func=criterion, opt_func=opt_func, metrics=accuracy)

    # learn = vision_learner(dls, resnet18, metrics=accuracy)

    # print(dls.device)
    print(learn.model)
    print(type(learn.model))
    exit(0)

    epoch = 20
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

    model_name = f"{model.__class__.__name__}-Epch:{epoch}-Acc:{results[1]*100:.0f}"
    print(f"model name for saving: {model_name}")

    learn.save(model_name)
    # learn.export("resnet18_finetuned.pkl")
