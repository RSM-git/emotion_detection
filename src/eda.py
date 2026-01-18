import matplotlib.pyplot as plt

from pathlib import Path
from data import download_dataset


def plot_training_samples():
    data_path = download_dataset()

    fig, ax = plt.subplots(3, 5)
    
    directories = data_path.glob("*")
    for directory in directories:
        for sub_dir in directory.glob("*"):
            print(sub_dir)


if __name__ == '__main__':
    plot_training_samples()
