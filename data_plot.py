import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def func_data_plot(path_csv):
    # path_csv = r'checkpoints/10202225/acc_epoch.csv'
    df = pd.read_csv(path_csv)
    print(df)
    print(df.columns)
    print(df.max(), df.min())
    x = list(df.iloc[:, 0])
    y = list(df.iloc[:, 1])
    plt.plot(x, y)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title(path_csv.split('/')[-1][:-4])
    plt.show()

if __name__ == '__main__':
    func_data_plot(r'checkpoints/10211809/acc_val_epoch.csv')
    func_data_plot(r'checkpoints/10211809/loss_epoch.csv')