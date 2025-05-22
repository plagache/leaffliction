import argparse
import matplotlib.pyplot as plt
from utils import DatasetFolder
# plt.switch_backend("qtagg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    folder: DatasetFolder = DatasetFolder(args.directory)
    sorted_count = dict(sorted(folder.count_dictionnary.items(), key=lambda item: item[1], reverse=True))
    labels = sorted_count.keys()
    sizes = sorted_count.values()
    num_classes = len(labels)

    if num_classes != 0:
        figure, axis = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)

        axis[0].pie(sizes, autopct="%1.1f%%", labels=labels)

        x_pos = range(num_classes)
        axis[1].bar(x_pos, sizes, align="center", alpha=0.5)
        axis[1].set_xticks(x_pos, labels, rotation=45)
        axis[1].set_ylabel("# of items")
        plt.title("classes distribution")

        # figure.savefig(f"{args.directory.replace('/', '-')}.jpg")
        plt.show()
