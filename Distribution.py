import os
import argparse
import matplotlib.pyplot as plt

def is_an_image():
    return

def list_files(directory):
    file_list = []
    dir_list = []

    pie_chart_dict = {}
    for (root, dirs, files) in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
        for dir in dirs:
            # dir_list.append(os.path.join(root, dir))
            dir_list.append(dir)
        pie_chart_dict[root] = len(files)
        print(f"root {root}")
        print(f"list of directory in {root}: {dirs}")
        print(f"number of files in {root}: {len(files)}")
        print ('--------------------------------')
    print(pie_chart_dict)
    print(dir_list)
    return pie_chart_dict, dir_list, file_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="analyse a dataset from a given directory")
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

    plt.pie(sizes, labels=labels)
    plt.show()
