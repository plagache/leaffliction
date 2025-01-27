from __future__ import annotations

import argparse

# import glob
import os
from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image


class DatasetFolder:
    def __init__(self, root: Union[str, Path]):
        if isinstance(root, Path):
            self.root = root
        else:
            self.root = Path(root)
        self.samples = self.make_dataset(self.root)
        self.items = self.samples
        self.classes, self.mapped_dictonnary, self.count_dictionnary, self.root_dictionnary, self.indexes_dictionnary = self.find_classes(self.root)
        self.images: Optional[list] = None
        self.numpy: Optional[list] = None

    def find_classes(self, directory):
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── x_sample_1.jpg
            │   ├── x_sample_2.jpg
            │   └── sub_class_x
            │       └── sub_class_x_sample_1.jpg
            └── class_y
                ├── y_sample_2.jpg
                ├── y_sample_2.jpg
                └── ...

        Inputs:
            directory(str): Root directory path, corresponding to ``self.root``

        Outputs:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.

            list["dog", "cat", "monkey"]

            dict{'dog': 0,
            'cat': 1,
            'monkey': 2
            }
        """
        mapped_dic = {}
        count_dic = {}
        root_dic = {}
        index_dic = {}
        categories = []
        category_index = 0
        for root, dirs, files in os.walk(directory, topdown=False):
            category = root if len(root.split("/")) < 1 else root.split("/")[-1]
            if len(files) != 0:
                categories.append(category)
                if len(dirs) == 0:
                    # print(root)
                    mapped_dic[category] = category_index
                    root_dic[category] = root
                    count_dic[category] = len(files)

                    # iterate over files to find the index of "root/file" in the samples
                    indexes = []
                    for file in files:
                        try:
                            index = self.samples.index(Path(f"{root}/{file}"))
                            indexes.append(index)
                        except ValueError:
                            print(f"{root}/{file} was not found in the samples this should never happened")
                            exit(-1)
                    index_dic[category] = indexes
                category_index += 1
            # How to test if directory/category is relevant?
            # Has at least one file that is not a directory
            # for example images is not a category
            # but a subdirectories with one image is
            # count = 0
            # for category in categories:
            #     if len(dirs) == 0:
            #         data_dic[category] = count
            #     count += 1
            # if len(dirs) == 0:
            #     data_dic[category] = len(files)
        return categories, mapped_dic, count_dic, root_dic, index_dic

    def make_dataset(self, directory):
        """
        Inputs:
            directory(str): Root directory path, corresponding to ``self.root``
        Outputs:
            samples: (lst) of sample
        """
        # pathname = directory + "/**"
        # files = glob.glob(pathname, recursive=True)
        pattern = "*"
        files = list(directory.rglob(pattern))
        samples = [sample for sample in files if os.path.isfile(sample) is True]
        return samples

    def to_path(self):
        self.items = self.samples
        return self

    def to_numpy(self):
        if self.images is None:
            self.to_images()
        if self.numpy is None:
            self.numpy = np.asarray(self.images)
        self.items = self.numpy
        return self

    def to_images(self):
        if self.images is None:
            self.images = []
            for path in self.samples:
                image = Image.open(path).copy()
                # image = image.load()
                self.images.append(image)
        # self.images = Image.open(self.samples)
        self.items = self.images
        return self

    def __getitem__(self, index: int):
        # def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Inputs:
            index: int

        Outputs:
            tuple: (sample, target) where target is class_index of the target class.
        """
        # sample = self.samples[index]

        class_index = None
        for key, value in self.mapped_dictonnary.items():
            if key in str(self.samples[index]):
                class_index = value

        return (self.items[index], class_index)

    def __len__(self) -> int:
        """
        needed by the Dataloaders to know how many samples there is and how to shuffle/separate each batch
        """
        lenght = len(self.items)
        return lenght


class Dataloaders:
    """
    we give a type(Dataset) to this Loaders a batch size, and a bunch of other parameters to this class and it fetch the data for us
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Dataset class")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    one_image = "images/Apple_rust/image (9).JPG"
    # dataset = DatasetFolder(Path(args.directory))
    dataset = DatasetFolder(args.directory)

    # test_bad_path = Path("this is not a folder")
    # if test_bad_path.exists():
    #     test_bad_path.open()
    # else:
    #     print("does not exist")

    # file_path, y = dataset[12]
    # file_path.read_bytes()
    # print(file_path.read_bytes())

    print(dataset.classes)
    print(dataset.mapped_dictonnary)

    # print(dataset.images)
    # dataset.to_images()
    # images = dataset.images
    # image = images[28]
    # print(image.format)
    # print(image.size)
    # print(image.mode)
    # image.show()
    #
    # dataset.to_numpy()
    # np_arrays = dataset.numpy
    # an_array = np_arrays[28]
    # print(an_array)
    # an_array = np_arrays[12]
    # print(an_array)
    # an_array = np_arrays[2]
    # print(an_array)

    # sample, label = dataset[28]
    # another_array = np.load(sample, allow_pickle=True)
    # print(another_array)

    print(dataset[0])
    print(dataset[1])
    dataset.to_images()
    print(dataset[15])
    print(dataset[300])
    dataset.to_numpy()
    print(dataset[2000])
    print(dataset[4000])
    print(dataset[7000])
    # print(dataset.samples)
    print(dataset.__len__())
    print(len(dataset))
    # print(dataset[20000])
    # print(dataset.__len__())

    # dataset.to_numpy()
    # print(dataset[12])
