from fastai.data.all import *
from fastai.vision.all import *


def verification(dls):
    for i in range(100):
        image, label = dls.train_ds[i]
        # print(image, label)
        label_name = dls.vocab[label]
        print(f"Decoded label: {label_name}, Encoded label: {label}")
        pil_image = PILImage.create(image)

        pred_class, pred_idx, outputs = learn.predict(pil_image)
        print(f"Predicted class: {pred_class}, Actual label: {label}")


if __name__ == "__main__":
    path = Path("images")

    dls = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
    ).dataloaders(path)
    # dls.show_batch(max_n=6)

    total_items = len(dls.train_ds)
    batch_size = dls.bs
    total_batches = len(dls.train)
    print(f"number of items: {total_items}, batch_size: {batch_size}, number of batches: {total_batches}")
    # 40k images is not optimal for training

    learn = vision_learner(dls, resnet18, metrics=accuracy)
    # results = learn.validate()
    # print(f"Validation accuracy: {results[1]:.2f}")

    # optimal_learning_rate = learn.lr_find()
    # print(f"Optimal learning rate: {optimal_learning_rate}")
    # learn.fine_tune(2, base_lr=optimal_learning_rate)
    learn.fine_tune(3, base_lr=0.002)

    results = learn.validate()
    print(f"Validation accuracy: {results[1]:.2f}")

    learn.save("resnet18_finetuned")
    learn.export("resnet18_finetuned.pkl")
    print(learn.model)
