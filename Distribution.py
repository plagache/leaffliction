import os
import argparse
import matplotlib.pyplot as plt
from  dataclasses import dataclass
# plt.switch_backend("qtagg")


@dataclass
class Category:
    name: str
    path: str
    files: list[str]
    count: int
    new_path: str = None
    modified_images: list[str] = None


def get_categories(directory) -> list[Category]:
    categories: list[Category] = []

    for root, dirs, files in os.walk(directory, topdown=False):
        # get the last part of the dirpath
        category = root if len(root.split("/")) < 1 else root.split("/")[-1]
        # How to test if directory/category is relevant?
        # has no files in it and has subdirectories/categories => is irrelevant
        if len(dirs) == 0:
            categories.append(Category(category, root, files, len(files)))
        # print(f"category {category}")
        # print(f"list of directory in {root}: {dirs}")
        # print(f"number of files in {root}: {len(files)}")
        # print("--------------------------------")
    # print(categories_dic)
    return categories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    categories: list[Category] = get_categories(args.directory)
    labels = [value.name for value in categories]
    sizes = [value.count for value in categories]

    if len(labels) != 0:
        figure, axis = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)

        axis[0].pie(sizes, autopct="%1.1f%%", labels=labels)

        y_pos = range(len(sizes))
        axis[1].bar(y_pos, sizes, align="center", alpha=0.5)
        axis[1].set_xticks(y_pos, labels, rotation=45)
        axis[1].set_ylabel("counts")
        plt.title("class distribution")

        figure.savefig(f"{args.directory.replace('/', '-')}.jpg")
        plt.show()
