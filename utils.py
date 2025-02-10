from __future__ import annotations

import argparse

from random import sample
from shutil import copy
from pathlib import Path
from typing import Union, Optional
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort

import numpy as np
from PIL import Image


class DatasetFolder:
    def __init__(self, root: Union[str, Path]):
        if isinstance(root, Path):
            self.root = root
        else:
            self.root = Path(root)
        if self.root.is_dir() is False:
            print("argument provided is not a valid directory")
            exit(0)
        self.samples = self.make_dataset(self.root)
        self.items = self.samples

        self.classes: Optional[list[str]] = None
        self.mapped_dictionnary: Optional[dict[str, int]] = None
        self.count_dictionnary: dict[str, int] = {}
        self.root_dictionnary: dict[str, str] = {}
        self.indexes_dictionnary: dict[str, list[int]] = {}
        self.__find_classes(self.root)
        self.images: Optional[list] = None
        self.numpy: Optional[list] = None
        self.augmented_images = {}
        self.max_count = max(self.count_dictionnary.values())

    def __find_classes(self, directory: Path):
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
        if self.classes is None or self.mapped_dictionnary is None:
            self.classes = []
            self.mapped_dictionnary = {}
            category_index = 0
            for root, dirs, files in directory.walk(top_down=False):
                root = str(root)
                category = root if len(root.split("/")) < 1 else root.split("/")[-1]
                if len(files) != 0:
                    self.classes.append(category)
                    if len(dirs) == 0:
                        # print(root)
                        self.mapped_dictionnary[category] = category_index
                        self.root_dictionnary[category] = root
                        self.count_dictionnary[category] = len(files)

                        # iterate over files to find the index of "root/file" in the samples
                        indexes = []
                        for file in files:
                            try:
                                index = self.samples.index(Path(f"{root}/{file}"))
                                indexes.append(index)
                            except ValueError:
                                print(f"{root}/{file} was not found in the samples this should never happened")
                                exit(-1)
                        self.indexes_dictionnary[category] = indexes
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
        return self.classes, self.mapped_dictionnary

    def make_dataset(self, directory: Path):
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
        samples = [sample for sample in files if Path.is_file(sample) is True]
        return samples

    def __get_modified_images(self, image):
        modified_images = {
            "Rotate": Rotate(probability=1, rotation=90).perform_operation([image]),
            "Flip": Flip(probability=1, top_bottom_left_right="RANDOM").perform_operation([image]),
            "Skew": Skew(probability=1, skew_type="TILT", magnitude=1).perform_operation([image]),
            "Shear": Shear(probability=1, max_shear_left=20, max_shear_right=20).perform_operation([image]),
            "Crop": CropRandom(probability=1, percentage_area=0.8).perform_operation([image]),
            "Distortion": Distort(probability=1, grid_width=2, grid_height=2, magnitude=9).perform_operation([image]),
        }
        return modified_images.items()

    def __get_modified_image_name(self, modification: str, image_path) -> str:
        split_image_name = str(image_path).split(".")
        split_image_name[0] = f"{split_image_name[0]}_{modification}"
        return ".".join(split_image_name)

    def augment_images(self):
        for name, indexes in self.indexes_dictionnary.items():
            print(f"augmenting images for category: {name}")

            augmented_images = []
            for index in indexes:
                file_pathname = self.samples[index]
                image = self.images[index]

                for modification, images in self.__get_modified_images(image):
                    output_path = self.__get_modified_image_name(modification, file_pathname)
                    augmented_images.append(output_path)
                    images[0].save(output_path)
            self.augmented_images[name] = augmented_images
        return self

    def get_items_from_categories(self, items: list, category: str):
        return list(map(lambda i: items[i], self.indexes_dictionnary[category]))

    def balance_dataset(self, output_directory: str):
        # Balance dataset
        # Create output directories (only relevant for balancing dataset)
        for category_name in self.classes:
            print(f"balancing category: {category_name}")
            new_path: str = f"{output_directory}/{self.root_dictionnary[category_name]}"
            Path(new_path).mkdir(parents=True, exist_ok=True)

            # 1 - copy all original images in new_path
            for file in self.get_items_from_categories(self.samples, category_name):
                copy(file, new_path)

            # when dir is created we can then balance this category around the biggest known
            # 2 - copy max_count - category.count images in new_path
            category_count = self.count_dictionnary[category_name]
            augmented_images = self.augmented_images[category_name]
            if self.max_count > category_count:
                # todo add a shuffle of augmented_images to select them at random
                for file in sample(augmented_images, self.max_count - category_count):
                    copy(file, new_path)
        return self

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

    # def __getitem__(self, index: int):
    #     # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     """
    #     Inputs:
    #         index: int
    #
    #     Outputs:
    #         tuple: (sample, target) where target is class_index of the target class.
    #     """
    #     # sample = self.samples[index]
    #
    #     class_index = None
    #     for key, value in self.mapped_dictonnary.items():
    #         if key in str(self.samples[index]):
    #             class_index = value
    #
    #     return (self.items[index], class_index)

    def __getitem__(self, index: int | slice):
        # def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Inputs:
            index: int

        Outputs:
            tuple: (sample, target) where target is class_index of the target class.
        """
        start = index
        stop, step = None, None

        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))

        if stop is None:
            stop = start + 1
        if step is None:
            step = 1

        elements = []
        for index in range(start, stop, step):
            class_index = None
            for key, value in self.mapped_dictionnary.items():
                if key in str(self.samples[index]):
                    class_index = value
            elements.append((self.items[index], class_index))
        if len(elements) == 1:
            return elements[0]
        return elements

        # return (self.items[index:end:stride], class_index)

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
    def __init__(self, dataset: DatasetFolder, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        # self.index = 0

    """
    object on wich we want to iterate (must be iterable)
    e. g. our dataset
    """
    def __iter__(self):
        self.index = 0
        while self.index + self.batch_size < len(self.dataset):
            yield tuple(self.dataset[self.index : self.index + self.batch_size])
            self.index += self.batch_size
        # return self
        # if self.dataset.numpy is None:
        #     self.dataset.to_numpy()
        # return self.dataset.numpy
        # rsize = range(self.batch_size)
        # elements = list()
        # for element in rsize:
        #     elements.append(self.dataset[element + self.index])
        # self.batch = tuple(elements)
        # return self.batch

    # """
    # how to fetch the next __iter__(object)
    # """
    # def __next__(self):
    #     if self.index <= len(self.dataset):
    #         rsize = range(self.batch_size)
    #         elements = list()
    #         for element in rsize:
    #             if element + self.index < len(self.dataset):
    #                 elements.append(self.dataset[element + self.index])
    #         self.batch = tuple(elements)
    #         # next_batch = self.batch
    #         # print(self.index)
    #         self.index += self.batch_size
    #         return self.batch
    #     else:
    #         raise StopIteration


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

    # print(dataset.classes)
    # print(dataset.mapped_dictionnary)
    print(dataset.classes)
    print(dataset.mapped_dictionnary)

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

    # print(dataset[0])
    # print(dataset[1])
    # dataset.to_images()
    # print(dataset[15])
    # print(dataset[300])
    # dataset.to_numpy()
    # print(dataset[2000])
    # print(dataset[4000])
    # print(dataset[7000])
    # print(dataset.samples)
    # print(dataset.__len__())
    # print(len(dataset))
    # print(dataset[20000])
    # print(dataset.__len__())

    # dataset.to_numpy()
    # print(dataset[12])

    dataloader = Dataloaders(dataset, batch_size=5)
    for batch in dataloader:
        print("______________")
        print(batch)
