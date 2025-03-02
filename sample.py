import argparse
from utils import DatasetFolder
from random import shuffle
from shutil import copy
from pathlib import Path


def copy_dataset(dataset: DatasetFolder, indices: list, output_directory: str):
    paths: dict[str, Path] = {}

    for index in indices:
        file, category_index = dataset[index]

        for class_name, class_index in dataset.mapped_dictionnary.items():
            if category_index == class_index:
                category_name = class_name

        if category_name in paths:
            destination = paths[category_name]
        else:
            destination: Path = Path(f"{output_directory}/{category_name}")

            if destination.exists() is True:
                raise FileExistsError(f"The directory [{destination}] already exist")

            paths[category_name] = destination
            destination.mkdir(parents=True, exist_ok=True)

        copy(file, destination)
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
