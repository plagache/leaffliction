import argparse
from  utils import DatasetFolder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a directory into a training and validation set")
    parser.add_argument("path", help="the directory to sample")
    args = parser.parse_args()

    print(f"sampling: {args.path}")

    # DatasetFolder(path)

    # mkdir(train_directory)
    # mkdir(validation_directory)

    # for indices in indices_category
        # shuffle(indices)
        # train, validation = indices[80%], indices[20%]

        # for train_indices in train:
        # copy(sample[train_indices], train_directory)

        # for validation_indices in validation:
        # copy(sample[validation_indices], validation_directory)

