import os
import argparse
import matplotlib.pyplot as plt
# plt.switch_backend("qtagg")


def is_an_image():
    return


def list_files(directory):
    file_list = []
    dir_list = []

    pie_chart_dict = {}
    for root, dirs, files in os.walk(directory):
        root = root if len(root.split("/")) <= 1 else root.split("/")[1]
        for file in files:
            file_list.append(os.path.join(root, file))
        for dir in dirs:
            # dir_list.append(os.path.join(root, dir))
            dir_list.append(dir)
        pie_chart_dict[root] = len(files)
        print(f"root {root}")
        print(f"list of directory in {root}: {dirs}")
        print(f"number of files in {root}: {len(files)}")
        print("--------------------------------")
    print(pie_chart_dict)
    print(dir_list)
    return pie_chart_dict, dir_list, file_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="analyse a dataset from a given directory"
    )
    parser.add_argument("directory", help="the directory to parse")
    args = parser.parse_args()

    pie_dictionaries, directories, files = list_files(args.directory)
    # print(directories)
    # print(files)
    labels = []
    sizes = []

    for label, size in pie_dictionaries.items():
        labels.append(label)
        sizes.append(size)

    figure, axis = plt.subplots(1, 2)
    axis[0].pie(sizes, autopct="%1.1f%%", labels=labels)

    y_pos = [i for i in range(0, len(sizes))]
    axis[1].bar(y_pos, sizes, align="center", alpha=0.5)
    axis[1].set_xticks(y_pos, labels)
    axis[1].set_ylabel("counts")
    plt.title("class distribution")

    plt.show()
