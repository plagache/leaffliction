from torch import load
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.vision.all import CrossEntropyLossFlat, ImageDataLoaders, Resize

from train import AlexNet
from pytorch_inference import get_accuracy, model_confusion

if __name__ == "__main__":
    dls = ImageDataLoaders.from_folder(".", train="train", valid="validation", item_tfms=Resize(227))

    model_path = "models/AlexNet-Epch:20-Acc:91.pth"
    state_dict = load(model_path, weights_only=False)
    model = AlexNet()
    # print(model)
    model.load_state_dict(state_dict["model"])
    # print(model)
    learner = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)

    # learner.dls.to(device='cuda')
    # learner.model.to(device='cuda')

    classes = dls.vocab

    # Predict an image from path
    # image_path = "images/Apple_rust/image (12).JPG"
    # predicted_label, predicted_index, probabilities = learner.predict(image_path)
    # print(f"label: {predicted_label}, index: {predicted_index}\nprobabilities: {probabilities}")

    # Predict the dataset
    # we can also use ds_idx = 0: train, 1: valid
    # probabilities, labels, decoded = learner.get_preds(ds_idx=1, with_decoded=True)
    probabilities, labels, decoded = learner.get_preds(dl=dls.valid, with_decoded=True)

    la_retourne_a_tourner = get_accuracy(labels, decoded)
    print(f"accuracy of {la_retourne_a_tourner * 100:.2f}% on {len(labels)} items")
    model_confusion(labels, decoded, classes, show=True)
