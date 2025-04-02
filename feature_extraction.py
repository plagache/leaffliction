import os
import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from utils import DatasetFolder, Dataloader
import argparse


# prepare data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train our model on a dataset from a train and test directory")
    parser.add_argument("train_directory", help="the train set directory to parse")
    parser.add_argument("test_directory", help="the validation set directory to parse")
    args = parser.parse_args()

    train_folder: DatasetFolder = DatasetFolder(args.train_directory)
    test_folder: DatasetFolder = DatasetFolder(args.test_directory)

    train_loader: Dataloader = Dataloader(train_folder, 12)
    test_loader: Dataloader = Dataloader(test_folder, 12)

    test_images, Y_test = test_loader.get_data()
    train_images, Y_train = train_loader.get_data()

    # print(train_images, test_images)
    # print(Y_train, Y_test)
    img2vec = Img2Vec(cuda=True)

    X_train = []
    for image in train_images:
        X_train.append(img2vec.get_vec(image))

    X_test = []
    for image in test_images:
        X_test.append(img2vec.get_vec(image))

    # print(X_train, Y_train)
    # print(X_test, Y_test)

    # train model
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, Y_train)

    # test performance
    y_pred = model.predict(X_test)
    score = accuracy_score(y_pred, Y_test)

    print(score)

    # save the model
    with open('./model.p', 'wb') as f:
        pickle.dump(model, f)
        f.close()
