from fastai.data.all import *
from fastai.vision.all import *
import matplotlib.pyplot as plt

path = Path("images")

dls = DataBlock(

    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2),
    get_y=parent_label

).dataloaders(path)
# dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=accuracy)
results = learn.validate()
print(f"Validation accuracy: {results[1]:.2f}")

# optimal_learning_rate = learn.lr_find()
# print(f"Optimal learning rate: {optimal_learning_rate}")
# learn.fine_tune(2, base_lr=optimal_learning_rate)
learn.fine_tune(3)

for i in range(100):
    image, label = dls.train_ds[i]
    # print(image, label)
    label_name = dls.vocab[label]
    print(f"Decoded label: {label_name}, Encoded label: {label}")
    pil_image = PILImage.create(image)


    pred_class, pred_idx, outputs = learn.predict(pil_image)  # Predict on a single image
    print(f"Predicted class: {pred_class}, Actual label: {label}")

results = learn.validate()
print(f"Validation accuracy: {results[1]:.2f}")

plt.show()
exit()
learn.save('resnet18_finetuned')
learn.export('resnet18_finetuned.pkl')
