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
    model, classes, transform = prepare_model(model_path)

    # load pil image
    image = Image.open(image_path)
    trans_image = transform(image)
    image_tensor = trans_image.unsqueeze(0)

    # apply model
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return f"{classes[predicted_class]}"


def predict_dataset(data_loader, model_path):
    model, classes, transform = prepare_model(model_path)

    labels = []
    predictions = []

    with torch.no_grad():
        # image_batch, label_batch = next(iter(data_loader))
        for image_batch, label_batch in data_loader:
            outputs = model(image_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=0)
            predicted_class = torch.argmax(probabilities, dim=1)
            labels.append(label_batch)
            predictions.append(predicted_class)

    labels = torch.cat(labels)
    predictions = torch.cat(predictions)
    return labels, predictions


def prepare_model(model_path):
    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder)

    state_dict = torch.load(model_path, weights_only=False)

    model, transform = get_model_from_path(model_path)

    model.load_state_dict(state_dict["model"])
    model.eval()

    return model, dataset.classes, transform


def model_confusion(y_true, y_prediction, classes):
    model_cm = confusion_matrix(y_true, y_prediction)
    print(classes)
    print(f"model confusion matrice:\n{model_cm}")
    display = ConfusionMatrixDisplay(model_cm, display_labels=classes).plot(cmap="Blues", xticks_rotation=45)
    plt.show()
    return display.figure_


def get_accuracy(y_true, y_prediction):
    accuracy = (y_prediction == y_true).sum()
    return f"valid: {accuracy * 100 / len(y_true):.2f}%\ndataset len: {len(y_true)}"


if __name__ == "__main__":
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])
    batch_size = 300
    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # print(dataset.classes)
    # print(dataset.class_to_idx)

    model = SmallModel()

    model_path = "models/SmallModel-Epch:5-Acc:86.pth"
    state_dict = torch.load("models/SmallModel-Epch:5-Acc:86.pth", weights_only=False)
    # print(f"torch load state_dict: {state_dict}")
    # exit(0)

    model.load_state_dict(state_dict["model"])
    model.eval()

    labels, predictions = predict_dataset(data_loader, model_path)
    la_retourne_a_tourner = get_accuracy(labels, predictions)
    print(la_retourne_a_tourner)
    model_confusion(labels, predictions, dataset.classes)
