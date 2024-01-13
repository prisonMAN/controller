import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_csv_waveform(file_path):
    x = []  # 存储X轴数据
    y = []  # 存储Y轴数据
    z = []  # 存储Z轴数据

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行

        for row in csv_reader:
            x.append(float(row[0]))  # 将第一列数据添加到X轴列表
            y.append(float(row[1]))  # 将第二列数据添加到Y轴列表
            z.append(float(row[2]))  # 将第三列数据添加到Z轴列表

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('column_index')
    plt.show()


def plot_csv_waveform_single(file_path, column_index, flag):
    x = []  # 存储行数
    y = []  # 存储数据

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # 跳过标题行

        for i, row in enumerate(csv_reader):
            x.append(i + 1)  # 行数从1开始
            y.append(float(row[column_index]))

    plt.plot(x, y)
    plt.xlabel('Row')
    plt.ylabel('Value')
    if flag == 0:
        plt.title(column_index)
    elif flag == 1:
        plt.title(column_index + 10)
    elif flag == 2:
        plt.title(column_index + 20)
    plt.grid(True)
    plt.show()


def main():
    # plot_csv_waveform('output.csv')
    plot_csv_waveform_single('output.csv', 0, 0)
    plot_csv_waveform_single('output.csv', 1, 0)
    plot_csv_waveform_single('output.csv', 2, 0)

    # plot_csv_waveform('flitered.csv')
    plot_csv_waveform_single('flitered_output.csv', 0, 1)
    plot_csv_waveform_single('flitered_output.csv', 1, 1)
    plot_csv_waveform_single('flitered_output.csv', 2, 1)

    # plot_csv_waveform_single('all.csv', 0, 2)
    # plot_csv_waveform_single('all.csv', 1, 2)
    # plot_csv_waveform_single('all.csv', 2, 2)


# 检查脚本是否直接运行
if __name__ == "__main__":
    main()
