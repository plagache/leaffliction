import torch
from fast_model import SmallModel, AlexNet
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_model_from_path(model_path):
    if AlexNet.__name__ in model_path:
        return AlexNet(), T.Compose([
            T.Resize(227),
            T.ToTensor(),
        ])
    if SmallModel.__name__ in model_path:
        return SmallModel(), T.Compose([
            T.Resize(224),
            T.ToTensor(),
        ])
    return "no model found"

def predict_image(image_path, model_path):
    model, dataset, transform = prepare_model(model_path)

    # load pil image
    image = Image.open(image_path)
    trans_image = transform(image)
    image_tensor = trans_image.unsqueeze(0)

    # apply model
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=-1)
    predicted_class = torch.argmax(probabilities).item()

    return f"{dataset.classes[predicted_class]}"


def predict_dataset(model_path):
    model, dataset, transform = prepare_model(model_path)

    batch_size = 64
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    labels = []
    predictions = []

    with torch.no_grad():
        for image_batch, label_batch in data_loader:
            outputs = model(image_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1)
            labels.append(label_batch)
            predictions.append(predicted_class)

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    return labels, predictions, dataset.classes


def prepare_model(model_path):

    state_dict = torch.load(model_path, weights_only=False)

    model, transform = get_model_from_path(model_path)

    model.load_state_dict(state_dict["model"])
    model.eval()

    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder, transform=transform)

    return model, dataset, transform


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
    model_path = "models/AlexNet-Epch:20-Acc:91.pth"

    labels, predictions, classes = predict_dataset(model_path)
    la_retourne_a_tourner = get_accuracy(labels, predictions)
    print(f"accuracy of {la_retourne_a_tourner * 100:.2f}% on {len(labels)} items")
    model_confusion(labels, predictions, classes, show=True)
