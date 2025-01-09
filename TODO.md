# TODO
download only if nescessary / not exist
inputs url, pathlib, subdir, gzip:Optional[method], return path with data

# Plan

- get data
<!--    - check data integrety-->
<!--    - normalization / resize-->
<!--    - distribution-->
<!--        - load data in class-->
<!--            - (labels|classes) / path / elements / number of elements / batches-->
<!--- augment data to balance dataset-->
- create data validation / training batches
- setup labels to detecte categories
- transform dataset to detect features from categories
    <!--- What count as data transformation: Convolution/-->
- train model
- classify new inputs from data validation

# Question

what type of augmentation is interesting ?
it should modify our data, but not create things that nature cannot produce
a tree can be rotate left to right, but the leaf cannot be on the soil for example

Datasets class implements how to get an items and the number of items in the Datasets
DatasetFolders subclass implements how to get items and categories base on folders structures

DataLoaders class takes a Datasets class as parameters and 

testing valid inputs

transform is a function of Datasets

dataloaders:
get_images=path_to_images where to get inputs
get_y=parents_labels is fetch from the directory name
function to create data validation set randomly
path to data
type of outputs data / models that is going to process this
