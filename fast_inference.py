# from fastai.data.all import *
from fastai.vision.all import *
import torch
# import fastai
# print(fastai)
from fastai.learner import load_learner
from fast_model import AlexNet

def predict_image(learner, image_path):
    test_dl = learner.dls.test_dl([image_path], with_labels=False)

    preds, _ = learner.get_preds(dl=test_dl)
    pred_class = preds.argmax(dim=1).item()

    return learner.dls.vocab[pred_class]


if __name__ == "__main__":
    # load dataloaders with the same seed than training ensuring valid and train set are identical
    path = Path("validation")
    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(227))

    # learner = load_learner("resnet18_finetuned.pkl")
    # print(f"eval model: {learner.model.eval()}")
    # print(f"model: {learner.model}")
    # model = learner.model.eval()
    # torch.save(model.state_dict(), 'weights_only.pth')

    # from a pytorch model
    # learner = vision_learner(dls, resnet18, metrics=accuracy).load('resnet18_finetuned')
    # learner.export()
    model_path = "models/AlexNet-Epch:10-Acc:94.pth"
    state_dict = torch.load(model_path, weights_only=False)
    # state_dict = torch.load('models/resnet18_finetuned.pth', weights_only=False)
    # print(state_dict)
    # model = resnet18()  # Use the same architecture as training
    model = AlexNet()  # Use the same architecture as training
    print(model)
    model.load_state_dict(state_dict["model"])
    print(model)
    learner = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    # learner = vision_learner(dls, resnet18, metrics=accuracy)
    # learner.load('resnet18_finetuned')
    # safer version
    # learner = Learner.load('resnet18_finetuned.pkl')
    # unsafe version
    # learn = load_learner('resnet18_finetuned.pkl')
    # print(learner.model)

    # fnames = get_image_files("images")
    # dblock = DataBlock()
    # dsets = dblock.datasets(fnames)
    # first_element = dsets.train[0]
    # print(first_element)

    # images = [
    #     "images/Apple_healthy/image (9).JPG",
    #     "images/Apple_rust/image (12).JPG",
    #     "images/Grape_Esca/image (601).JPG",
    #     "images/Grape_spot/image (203).JPG",
    #     "images/Grape_healthy/image (402).JPG",
    #     "images/Apple_Black_rot/image (115).JPG"
    # ]
    # print(dls.valid_ds.items)
    # exit(0)

    # acc2 = learner.validate(dl=dls)
    # print(acc2)
    # exit(0)

    truesque = 0
    lenny = 0
    for image_path in dls.valid_ds.items:
        # image_path = Path(image)
        lenny += 1
        prediction = predict_image(learner, image_path)
        if prediction in str(image_path):
            truesque += 1
        print(f"Image: {image_path}, Predicted: {prediction}")

    acc = truesque / lenny * 100
    print(f"acc = {acc}")

    # validation_set = get_image_files(path)
    # test_loader = learner.dls.test_dl(validation_set, with_labels=False)
    # preds, _ = learner.get_preds(dl=test_loader)
    # pred_classes = preds.argmax(dim=1)
    # labels = [learner.dls.vocab[i] for i in pred_classes]
    # for file, label in zip(validation_set, labels):
    #     print(f"Image: {file.name}, Predicted: {label}")

    # learn = vision_learner(dls, resnet18, metrics=accuracy)
    # learn.fine_tune(3)
    # optimal_learning_rate = learn.lr_find()
    # learn.fine_tune(3, base_lr=optimal_learning_rate)
    # results = learn.validate()
    # print(results)
    # learn.save('resnet18_finetuned')
