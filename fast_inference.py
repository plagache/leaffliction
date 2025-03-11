# from fastai.data.all import *
from fastai.vision.all import *
# import fastai
# print(fastai)

from fastai.learner import load_learner
learn = load_learner('resnet18_finetuned.pkl')

# fnames = get_image_files("images")
# path = Path("images")
# dblock = DataBlock()
# dsets = dblock.datasets(fnames)
# first_element = dsets.train[0]
# print(first_element)

dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, item_tfms=Resize(224), batch_tfms=aug_transforms())
# learn = vision_learner(dls, resnet18, metrics=accuracy)
# learn.fine_tune(3)
# optimal_learning_rate = learn.lr_find()
# learn.fine_tune(3, base_lr=optimal_learning_rate)
results = learn.validate()
print(results)
# learn.save('resnet18_finetuned')
