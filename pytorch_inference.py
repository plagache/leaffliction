import torch
from fast_model import SmallModel, AlexNet
from torchvision import datasets
import torchvision.transforms as T
from pathlib import Path
from PIL import Image

def predict_image(image_path, model, classes):
    # load pil image
    image = Image.open(image_path)
    # pre transform image to tensor
    transform = T.Compose([
        T.Resize(227),
        T.ToTensor(),
    ])
    trans_image = transform(image)
    image_tensor = trans_image.unsqueeze(0)

    # apply model
    with torch.no_grad():
        output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    return f"Predicted class: {classes[predicted_class]}"

def prepare_model(model_path):
    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder)

    state_dict = torch.load(model_path, weights_only=False)
    
    # USE MODEL PATH TO COMPUTE WHICH MODEL TO USE
    model = AlexNet()
    model.load_state_dict(state_dict["model"])
    model.eval()

    return model, dataset.classes 


if __name__ == "__main__":
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])
    data_folder = Path("validation")
    dataset = datasets.ImageFolder(data_folder, transform=transform)
    print(dataset.class_to_idx)

    model = SmallModel(len(dataset.class_to_idx))

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
