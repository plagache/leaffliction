import torch
from fast_model import SmallModel, AlexNet
from torchvision import datasets
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

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

def prepare_model(model_path):
    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder)

    state_dict = torch.load(model_path, weights_only=False)
    
    model, transform = get_model_from_path(model_path)

    model.load_state_dict(state_dict["model"])
    model.eval()

    return model, dataset.classes, transform


if __name__ == "__main__":
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])
    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    print(dataset.class_to_idx)

    model = SmallModel()

    state_dict = torch.load("models/SmallModel-Epch:5-Acc:86.pth", weights_only=False)
    # print(f"torch load state_dict: {state_dict}")
    # exit(0)

    model.load_state_dict(state_dict["model"])
    model.eval()

    for element in dataset:
        image, label = element
        resized_image = image.unsqueeze(0)
        # print(image.shape)
        # print(resized_image.shape)
        with torch.no_grad():
            output = model(resized_image)
        # print(output)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)
        predicted_class = torch.argmax(probabilities).item()

        print(f"Predicted class: {dataset.classes[predicted_class]}, True label: {dataset.classes[label]}")
