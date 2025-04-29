import torch
from fast_model import SmallModel
from torchvision import datasets
import torchvision.transforms as T
from pathlib import Path

if __name__ == "__main__":
    data_folder = Path("validation")
    transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
    ])
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
