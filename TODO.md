# TODO
download only if nescessary / not exist
inputs url, pathlib, subdir, gzip:Optional[method], return path with data

# Plan

- get data
<!--- add shuffle to dataloader-->
- add show batch
- show one item of each category
<!--    - check data integrety-->
<!--    - normalization / resize-->
<!--    - distribution-->
<!--        - load data in class-->
<!--            - (labels|classes) / path / elements / number of elements / batches-->
<!--- augment data to balance dataset-->
- create data validation / training batches
- sampling test and training on augmented dataset
- setup labels to detect categories

- transform dataset to detect features from categories
    - test pixel intensity to determine threshold
    <!--- What count as data transformation: Convolution/-->

- train model
    - test cnn from tinygrad mnist example to train
    <!--- function in dataloader which returns the X_train, Y_train as tinygrad.Tensor-->
    <!--- reduce ndtype on tensor-->
    <!--- no grad on input tensor-->
    - normalize input tensor (/255)
    - simplify training by default (set debug / context / jit as optional with env variable)
    <!--- modify batch size-->
- classify new inputs from data validation

<!--- clean Utils-->

- Test
    - Test distribution with different structure of directories
- transformation
    - usage with argparse handle single file and directory

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
