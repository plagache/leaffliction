import torch
from train import SmallModel, AlexNet
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse
from Transformation import transform_image
from Augmentation import display_images
from tqdm import tqdm
import json

def get_model_from_path(model_path, classes_count):
    if AlexNet.__name__ in str(model_path):
        return AlexNet(classes_count), T.Compose([
            T.Resize(227),
            T.ToTensor(),
        ])
    if SmallModel.__name__ in str(model_path):
        return SmallModel(classes_count), T.Compose([
            T.Resize(224),
            T.ToTensor(),
        ])
    return "no model found"

def predict_image(image_path, model_path):
    model, transform, classes = prepare_model(model_path)

    # load pil image
    image = Image.open(image_path)
    trans_image = transform(image)
    image_tensor = trans_image.unsqueeze(0)

    # apply model
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=-1)
    predicted_class = torch.argmax(probabilities).item()

    return f"{classes[predicted_class]}"


def predict_dataset(model_path):
    model, transform, classes = prepare_model(model_path)

    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels = []
    predictions = []

    with torch.no_grad():
        for image_batch, label_batch in tqdm(data_loader):
            outputs = model(image_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            labels.append(label_batch)
            predictions.append(predicted_class)

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    return labels, predictions, dataset.classes

def get_classes_from_path(model_path):
    classes_file = Path(f"{model_path.parent}/{model_path.stem}.json")
    if classes_file.is_file():
        with open(classes_file, 'r') as f:
            classes = json.load(f)
    else:
        print(f"{classes_file} not detected")
        exit(0)
    return classes


def prepare_model(model_path):
    model_path = Path(model_path)
    classes = get_classes_from_path(model_path)
    model, transform = get_model_from_path(model_path, len(classes))

    state_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(state_dict)

    model.eval()

    return model, transform, classes


def model_confusion(y_true, y_prediction, classes, show=False):
    model_cm = confusion_matrix(y_true, y_prediction)
    print(f"model confusion matrice:\n{model_cm}")
    display = ConfusionMatrixDisplay(model_cm, display_labels=classes).plot(cmap="Blues", xticks_rotation=45)
    if show is True:
        plt.show()
    return display.figure_


def get_accuracy(y_true, y_prediction):
    return (y_prediction == y_true).sum() / len(y_true)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict")
    parser.add_argument("filename", help="a single image to predict", nargs="?", default=None)
    args = parser.parse_args()

    model_path = "models/AlexNet-Apple_dataset-Epch:10-Acc:91.pth"
    if args.filename is None:
        labels, predictions, classes = predict_dataset(model_path)
        la_retourne_a_tourner = get_accuracy(labels, predictions)
        print(f"accuracy of {la_retourne_a_tourner * 100:.2f}% on {len(labels)} items")
        model_confusion(labels, predictions, classes, show=True)
        exit(0)

    file = Path(args.filename)
    if file.is_file() is True:
        predicted_class = predict_image(file, model_path)
        transformed_images = transform_image(file)
        transformed_images = [item for item in transformed_images if "Masked" in item[0] or "Original" in item[0]]
        display_images(f"Class predicted: {predicted_class}", transformed_images)
    else:
        parser.error("file provided doesn't exist")
        parser.print_help()
