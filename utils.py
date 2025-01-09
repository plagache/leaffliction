from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Union


class DatasetFolder:

    def __init__(self, root: Union[str, Path]):
        self.root = root
        self.classes = self.find_classes(self.root)

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
        data_dic = {}
        data = []
        categories = set()
        for root, dirs, files in os.walk(self.root, topdown=False):
            category = root if len(root.split("/")) < 1 else root.split("/")[-1]
            if len(files) != 0:
                categories.add(category)
            # How to test if directory/category is relevant?
            # Has at least one file that is not a directory
            # for example images is not a category
            # but a subdirectories with one image is
            if len(dirs) == 0:
                data_dic[category] = len(files)
        return data_dic, categories

    def __getitem__(self, index: int):
        # def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Inputs:
            index: int

        Outputs:
            tuple: (sample, target) where target is class_index of the target class.
        """
        return

    def __len__(self) -> int:
        """
        needed by the Dataloaders to know how many samples there is and how to shuffle/separate each batch
        """
        return 0


class Dataloaders:
    """
    we give a type(Dataset) to this Loaders a batch size, and a bunch of other parameters to this class and it fetch the data for us
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="testing the Dataset class")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    dataset = DatasetFolder(args.directory)
    print(dataset.classes)
    # element, labels = dataset.__get_item__(5)
    # print(dataset.__len__())
