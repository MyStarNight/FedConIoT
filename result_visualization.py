import matplotlib.pyplot as plt
import os
import glob
import pandas as pd
import numpy as np


def exp1_visualization_single(group_name: str):
    cfl_file_path = rf"result\{group_name}-cfl"
    dfl_file_path = rf"result\{group_name}-dfl"

    cfl_accuracy_path = glob.glob(os.path.join(cfl_file_path, 'accuracy*.csv'))[-1]
    cfl_time_path = glob.glob(os.path.join(cfl_file_path, 'time*.csv'))[-1]

    dfl_accuracy_path = glob.glob(os.path.join(dfl_file_path, 'accuracy*.csv'))[-1]
    dfl_time_path = glob.glob(os.path.join(dfl_file_path, 'time*.csv'))[-1]

    cfl_accuracy = pd.read_csv(cfl_accuracy_path, index_col=0)
    cfl_time = pd.read_csv(cfl_time_path, index_col=0).iloc[:100]

    dfl_accuracy = pd.read_csv(dfl_accuracy_path, index_col=0)
    dfl_time = pd.read_csv(dfl_time_path, index_col=0).iloc[:100]

    dfl_time_list = []
    cfl_time_list = []
    for i in range(20):
        start_index = 0
        end_index = 5 * (i+1)
        dfl_time_sum = np.sum(dfl_time.iloc[start_index: end_index].values)
        cfl_time_sum = np.sum(cfl_time.iloc[start_index: end_index].values)

        dfl_time_list.append(dfl_time_sum)
        cfl_time_list.append(cfl_time_sum)

    # 横坐标round，纵坐标accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(dfl_accuracy, label='DFL Accuracy', color='royalblue', linestyle='-', marker='o')
    plt.plot(cfl_accuracy, label='CFL Accuracy', color='darkorange', linestyle='--', marker='x')
    plt.title('Model Accuracy over Rounds', fontsize=16)
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0.3, 0.9)
    plt.xlim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()

    # 横坐标time，纵坐标accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(dfl_time_list, dfl_accuracy, label='DFL Accuracy', color='royalblue', linestyle='-', marker='o')
    plt.plot(cfl_time_list, cfl_accuracy, label='CFL Accuracy', color='darkorange', linestyle='--', marker='x')
    plt.title('Model Accuracy over Time', fontsize=16)
    plt.xlabel('Time (seconds)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0.3, 0.9)
    plt.xlim(0, max(dfl_time_list[-1], cfl_time_list[-1])+50)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()

    return cfl_accuracy, cfl_time, dfl_accuracy, dfl_time, dfl_time_list, cfl_time_list


def read_csv_file(file_path):
    accuracy_path = glob.glob(os.path.join(file_path, 'accuracy*.csv'))[-1]
    time_path = glob.glob(os.path.join(file_path, 'time*.csv'))[-1]

    accuracy_df = pd.read_csv(accuracy_path, index_col=0)
    time_df = pd.read_csv(time_path, index_col=0).iloc[:100]

    return accuracy_df, time_df


def calculate_5_rounds_time(time_df):
    time_list = []
    for i in range(20):
        start_index = 0
        end_index = (i+1) * 5
        time_sum = np.sum(time_df.iloc[start_index: end_index].values)
        time_list.append(time_sum)

    return time_list


def exp1_visualization_all():
    group_name_list = ['5-5J',  '7-3J_4R', '10-10R', '15-5J_10R']
    color_list = ['darkorange', 'royalblue', 'forestgreen', 'crimson', 'tomato', 'dodgerblue', 'limegreen', 'firebrick']

    cfl_accuracy_df_list = []
    dfl_accuracy_df_list = []
    cfl_time_list_list = []
    dfl_time_list_list = []
    for group_name in group_name_list:
        cfl_file_path = rf"result\{group_name}-cfl"
        dfl_file_path = rf"result\{group_name}-dfl"

        cfl_accuracy_df, cfl_time_df = read_csv_file(cfl_file_path)
        dfl_accuracy_df, dfl_time_df = read_csv_file(dfl_file_path)

        cfl_accuracy_df_list.append(cfl_accuracy_df)
        dfl_accuracy_df_list.append(dfl_accuracy_df)
        cfl_time_list_list.append(calculate_5_rounds_time(cfl_time_df))
        dfl_time_list_list.append(calculate_5_rounds_time(dfl_time_df))

    # 整图：横坐标round，纵坐标accuracy
    plt.figure(figsize=(10, 6))
    for i, group_name in enumerate(group_name_list):
        plt.plot(cfl_accuracy_df_list[i], label=f'{group_name} CFL Accuracy', color=color_list[i], linestyle='--', marker='x')
        plt.plot(dfl_accuracy_df_list[i], label=f'{group_name} DFL Accuracy', color=color_list[i], linestyle='-', marker='o')

    plt.title('Model Accuracy over Rounds', fontsize=16)
    plt.xlabel('Rounds', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0.4, 0.9)
    plt.xlim(0, 105)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()

    # 分图：横坐标round，纵坐标accuracy
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axs.flat):
        ax.plot(cfl_accuracy_df_list[i], label=f'{group_name_list[i]} CFL', color=color_list[i], linestyle='--',
                marker='x')
        ax.plot(dfl_accuracy_df_list[i], label=f'{group_name_list[i]} DFL', color=color_list[4+i], linestyle='-',
                marker='o')
        ax.set_title(f'{group_name_list[i]} Model Accuracy over Rounds')
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 0.9)
        ax.set_xlim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # 整图：横坐标time，纵坐标accuracy
    plt.figure(figsize=(10, 6))
    for i, group_name in enumerate(group_name_list):
        plt.plot(cfl_time_list_list[i], cfl_accuracy_df_list[i], label=f'{group_name} CFL Accuracy', color=color_list[i], linestyle='--', marker='x')
        plt.plot(dfl_time_list_list[i], dfl_accuracy_df_list[i], label=f'{group_name} DFL Accuracy', color=color_list[i], linestyle='-', marker='o')

    plt.title('Model Accuracy over Time', fontsize=16)
    plt.xlabel('Time(seconds)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.ylim(0.4, 0.9)
    # plt.xlim()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.show()

    # 分图：横坐标round，纵坐标accuracy
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axs.flat):
        ax.plot(cfl_time_list_list[i], cfl_accuracy_df_list[i], label=f'{group_name_list[i]} CFL', color=color_list[i], linestyle='--',
                marker='x')
        ax.plot(dfl_time_list_list[i], dfl_accuracy_df_list[i], label=f'{group_name_list[i]} DFL', color=color_list[4+i], linestyle='-',
                marker='o')
        ax.set_title(f'{group_name_list[i]} Model Accuracy over Time')
        ax.set_xlabel('Time(seconds)')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0.4, 0.9)
        # ax.set_xlim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    return cfl_accuracy_df_list, dfl_accuracy_df_list, cfl_time_list_list, dfl_time_list_list


def exp2_visualization():
    data = pd.read_excel("result/federated_learning_time.xlsx", sheet_name="Sheet1", index_col=2).iloc[:10, 2:]
    data['CFL Communication Time'] = data['CFL Pull Time'] + data['CFL Push Time']
    data['DFL Communication Time'] = data['DFL Pull Time'] + data['DFL Push Time']

    new_index = ['5J', '5R', '5J+2R', '3J+4R', '7R', '5J+5R', '10R', '5J+8R', '3J+10R', '5J+10R']
    data = data.reindex(new_index)

    groups = list(data.index)
    index = np.arange(len(groups))

    fig, ax1 = plt.subplots(figsize=(14, 8))
    bar_width = 0.35
    bar1 = ax1.bar(index, data['CFL Communication Time'], bar_width, label='CFL Communication Time', color='royalblue')
    bar2 = ax1.bar(index + bar_width, data['DFL Communication Time'], bar_width, label='DFL Communication Time',
                   color='darkorange')

    ax1.set_title('CFL and DFL Communication Time for Different Clusters', fontsize=16)
    ax1.set_xlabel('Clusters', fontsize=14)
    ax1.set_ylabel('Communication Time (seconds)', fontsize=14)
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(data.index, fontsize=12)

    # ax1.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')

    ax2 = ax1.twinx()
    line = ax2.plot(index + bar_width / 2, data['Communication Improvement']*100, label='Communication Improvement',
                    color='darkred', marker='o', linestyle='-', linewidth=2)

    ax2.set_ylabel('Communication Improvement (%)', fontsize=14)
    ax2.set_ylim(0, 50)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.show()


if __name__ == '__main__':
    # result = exp1_visualization_single('10-10R')
    # result = exp1_visualization_all()
    exp2_visualization()