import argparse
import matplotlib.pyplot as plt
from utils import DatasetFolder
# plt.switch_backend("qtagg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    folder: DatasetFolder = DatasetFolder(args.directory)
    labels = folder.classes
    sizes = folder.count_dictionnary.values()

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
