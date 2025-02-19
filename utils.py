from __future__ import annotations

from os import strerror
import errno
from random import sample, shuffle
from shutil import copy
from pathlib import Path
from typing import Union, Optional, Self
from Augmentor.Operations import Flip, Rotate, Skew, Shear, CropRandom, Distort
from tinygrad import Tensor

import numpy as np
from PIL import Image


class DatasetFolder:
    def __init__(self, root: Union[str, Path]):
        if isinstance(root, Path):
            self.root = root
        else:
            self.root = Path(root)
        if self.root.is_dir() is False:
            raise FileNotFoundError(errno.ENOENT, strerror(errno.ENOENT), root)
        self.samples: list(Path) = self.make_dataset(self.root)
        self.items: list = self.samples

        self.classes: Optional[list[str]] = None
        self.mapped_dictionnary: Optional[dict[str, int]] = None
        self.count_dictionnary: dict[str, int] = {}
        self.root_dictionnary: dict[str, str] = {}
        self.indices_dictionnary: dict[str, list[int]] = {}
        self.__find_classes(self.root)
        self.images: Optional[list] = None
        self.numpy_arrays: Optional[list[np.ndarray]] = None
        self.augmented_images = {}
        self.max_count = max(self.count_dictionnary.values())

    def __find_classes(self, directory: Path) -> tuple(list(str), dict(str, int)):
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
                        self.mapped_dictionnary[category] = category_index
                        self.root_dictionnary[category] = root
                        self.count_dictionnary[category] = len(files)

                        # iterate over files to find the index of "root/file" in the samples
                        indices = []
                        for file in files:
                            try:
                                index = self.samples.index(Path(f"{root}/{file}"))
                                indices.append(index)
                            except ValueError:
                                print(f"{root}/{file} was not found in the samples this should never happened")
                                exit(-1)
                        self.indices_dictionnary[category] = indices
                    category_index += 1
        return self.classes, self.mapped_dictionnary

    def make_dataset(self, directory: Path) -> list(Path):
        """
        Inputs:
            directory(str): Root directory path, corresponding to ``self.root``
        Outputs:
            samples: (list) of sample
        """
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

    def augment_images(self) -> Self:
        """
        Augment all images found
        """
        for name, indices in self.indices_dictionnary.items():
            print(f"augmenting images for category: {name}")

            augmented_images = []
            for index in indices:
                file_pathname = self.samples[index]
                image = self.images[index]

                for modification, images in self.__get_modified_images(image):
                    output_path = self.__get_modified_image_name(modification, file_pathname)
                    augmented_images.append(output_path)
                    images[0].save(output_path)
            self.augmented_images[name] = augmented_images
        return self

    def get_items_from_categories(self, items: list, category: str) -> list:
        """
        Inputs:
            items: A list of items of different categories
            category: The category of items that will be returned
        Outputs:
            list: list of items of given category
        """
        return list(map(lambda i: items[i], self.indices_dictionnary[category]))

    def balance_dataset(self, output_directory: str) -> Self:
        """
        Inputs:
            output_directory: the directory where the balanced dataset will be saved
        """
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
                for file in sample(augmented_images, self.max_count - category_count):
                    copy(file, new_path)
        return self

    def to_path(self) -> Self:
        self.items = self.samples
        return self

    def to_numpy(self) -> Self:
        if self.images is None:
            self.to_images()
        if self.numpy_arrays is None:
            self.numpy_arrays = np.asarray(self.images)
        self.items = self.numpy_arrays
        return self

    def to_images(self) -> Self:
        if self.images is None:
            self.images = []
            for path in self.samples:
                image = Image.open(path).copy()
                # image = image.load()
                self.images.append(image)
        # self.images = Image.open(self.samples)
        self.items = self.images
        return self

    def __getitem__(self, index: int | slice) -> tuple | list(tuple):
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

        elements: list(tuple) = []
        for index in range(start, stop, step):
            class_index = None
            for key, value in self.mapped_dictionnary.items():
                if key in str(self.samples[index]):
                    class_index = value
            elements.append((self.items[index], class_index))
        if len(elements) == 1:
            return elements[0]
        return elements

    def __len__(self) -> int:
        """
        needed by the Dataloaders to know how many samples there is and how to shuffle/separate each batch
        """
        lenght = len(self.items)
        return lenght


class Dataloader:
    """
    we give a type(Dataset) to this Loaders a batch size, and a bunch of other parameters to this class and it fetch the data for us
    """

    def __init__(self, dataset: DatasetFolder, batch_size: int, shuffle: bool = False):
        self.dataset: DatasetFolder = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.indices: list(int) = list(range(len(self.dataset)))
        self.x_tensor: Tensor = None

    def _reset_indices(self):
        self.indices: list(int) = list(range(len(self.dataset)))
        return self

    def __iter__(self) -> tuple:
        """
        object on wich we want to iterate (must be iterable)
        e. g. our dataset
        """
        self.index = 0
        if self.shuffle is True:
            shuffle(self.indices)
        else:
            self._reset_indices()

        while self.index + self.batch_size < len(self.dataset):
            # batch_size number of indices
            batch_indices = self.indices[self.index : self.index + self.batch_size]
            batch = [self.dataset[index] for index in batch_indices]
            yield tuple(batch)
            self.index += self.batch_size

    def show_batch(self):
        # Find a batch
        # Match class_index from batch with category_name
        # use display images from Augmentation
        batch = next(iter(self))
        print(batch)
        pass

    def get_tensor(self) -> tuple(Tensor, Tensor):
        """
        Return the X_train, Y_train as tinygrad.Tensor from the dataset
        from images numpy arrays create a unique Tensor containing all the dataset => X_train
        see getitem to retrieve class index => Y_train
        """
        if self.dataset.numpy_arrays is None:
            self.dataset.to_numpy()

        simple_array = np.concatenate(self.dataset.numpy_arrays)

        labels_array = np.zeros(len(self.dataset))
        for label, indices in self.dataset.indices_dictionnary.items():
            np.put(labels_array, indices, self.dataset.mapped_dictionnary[label])
        self.y_tensor = Tensor(labels_array)
        self.x_tensor = Tensor(simple_array)
        self.x_tensor = self.x_tensor.reshape(-1, 3, 256, 256)
        return self.x_tensor, self.y_tensor, self.x_tensor[:10], self.y_tensor[:10]
