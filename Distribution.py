import os
import argparse
import matplotlib.pyplot as plt
# plt.switch_backend("qtagg")


def is_an_image():
    return


def get_categories_dic(directory):
    data_dic = {}

    for root, dirs, files in os.walk(directory, topdown=False):
        # get the last part of the dirpath
        root = root if len(root.split("/")) < 1 else root.split("/")[-1]
        # How to test if directory/category is relevant?
        # has no files in it and has subdirectories/categories => is irrelevant
        if len(dirs) == 0:
            data_dic[root] = len(files)
        print(f"root {root}")
        print(f"list of directory in {root}: {dirs}")
        print(f"number of files in {root}: {len(files)}")
        print("--------------------------------")
    print(data_dic)
    return data_dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="analyse a dataset from a given directory"
    )
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    categories_dic = get_categories_dic(args.directory)
    labels = categories_dic.keys()
    sizes = categories_dic.values()

    if len(labels) == 0:
        figure, axis = plt.subplots(1, 2)
        axis[0].pie(sizes, autopct="%1.1f%%", labels=labels)

        y_pos = [i for i in range(0, len(sizes))]
        axis[1].bar(y_pos, sizes, align="center", alpha=0.5)
        axis[1].set_xticks(y_pos, labels)
        axis[1].set_ylabel("counts")
        plt.title("class distribution")

        plt.show()
