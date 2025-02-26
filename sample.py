import argparse
from utils import DatasetFolder
from random import shuffle
from shutil import copy
from pathlib import Path


def copy_dataset(dataset: DatasetFolder, indices: list, output_directory: str):
    for category_name in dataset.classes:
        new_path: str = f"{output_directory}/{category_name}"
        Path(new_path).mkdir(parents=True, exist_ok=True)

    for indice in indices:
        old_path, index = dataset[indice]
        for key, value in dataset.mapped_dictionnary.items():
            if index == value:
                category_name = key
        # category_name = next(key for key, value in dataset.mapped_dictionnary.items() if index == value)
        # print(category_name, index)
        new_path: str = f"{output_directory}/{category_name}"
        copy(old_path, new_path)
    return


def random_split(dataset: DatasetFolder, percentage: float):
    train_indices = []
    validation_indices = []
    for category, indices in dataset.indices_dictionnary.items():
        shuffle(indices)

        number_to_slice_for_validation = len(indices) * percentage

        train_category_indices, validation_category_indices = (
            indices[int(number_to_slice_for_validation) :],
            indices[: int(number_to_slice_for_validation)],
        )

        train_indices += train_category_indices
        validation_indices += validation_category_indices

    # print(len(train_indices), len(validation_indices))
    # print(len(dataset) * (1 - percentage), len(dataset) * percentage)

    return train_indices, validation_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a directory into a training and validation set")
    parser.add_argument("path", help="the directory to sample")
    args = parser.parse_args()

    print(f"sampling: {args.path}")

    dataset = DatasetFolder(args.path)

    train, validation = random_split(dataset, 0.2)

    copy_dataset(dataset, train, "train")
    copy_dataset(dataset, validation, "validation")
