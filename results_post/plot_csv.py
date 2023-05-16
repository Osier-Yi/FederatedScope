import pandas as pd
from results_post.results_collection import check_dir
from results_post.results_collection import split_dir_name
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


COLORS=['red', 'blue', 'green', 'black']
MAXROUND = -1
Linewidth = 3
FontSize = 25
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

MAX2RUN=200


def plot(alpha_all_dict, plot_key_list, plot_sav_dir, min_rep=5):
    re_run_file = []
    check_dir(plot_sav_dir)
    for rep in alpha_all_dict.keys():
        rep_times = len(alpha_all_dict[rep])
        if rep_times < min_rep:
            print("{} repeats less then {} times".format(rep, min_rep))
            continue

        client_num = alpha_all_dict[rep][0]['client'].unique().shape[0]
        if client_num == 0:
            re_run_file.append(rep)
            continue

        print('working: {}'.format(rep))

        for plot_key in plot_key_list:
            plot_cnt = 0
            for client_id in range(1, client_num + 1):
                acc_mean_list = []
                acc_std_list = []
                round_list = []
                data_collect = []
                len_col = []
                for rep_id in range(rep_times):
                    tmp_data = acc_all_dict[rep][rep_id]
                    data = tmp_data[tmp_data['client'] == client_id][['round', plot_key]].to_numpy()
                    len_col.append(data.shape[0])
                    data_collect.append(data)
                max_plot_round = min(len_col)
                print('file name: {}, min round: {}'.format(rep, data_collect[0][max_plot_round-1, 0]))
                if data_collect[0][max_plot_round-1, 0] < MAX2RUN:
                    if rep not in re_run_file:
                        re_run_file.append(rep)
                    continue

                for i in range(max_plot_round):
                    round_list.append(data_collect[0][i, 0])
                    acc_mean_list.append(np.mean([rep[i, 1] for rep in data_collect]))
                    acc_std_list.append(np.std([rep[i, 1] for rep in data_collect]))
                plt.plot(round_list, acc_mean_list, label='client: {}'.format(client_id), color=COLORS[client_id - 1])
                plt.fill_between(round_list,
                                 np.array(acc_mean_list) - np.array(acc_std_list),
                                 np.array(acc_mean_list) + np.array(acc_std_list),
                                 color=COLORS[client_id - 1], alpha=0.2)
                plot_cnt+=1
            if plot_cnt == client_num:
                plt.legend()
                plt.xlabel('Round')
                plt.ylabel(plot_key)
                # plt.title(plot_key)
                plt.grid()

                if client_num == 3:
                    plt.ylim([0.5, 0.9])
                else:
                    plt.ylim([0.2, 0.95])
                plt.tight_layout()
                plt.savefig(os.path.join(plot_sav_dir, rep + '_' + plot_key + '.pdf'))
                plt.close()
            else:
                if rep not in re_run_file:
                    re_run_file.append(rep)






            # if plot_key.split('_')[-1] == 'acc':
            #     if plot_key.split('_')[0] == 'train':
            #         plt.ylim([0.5, 1])
            #     elif client_num == 2:
            #         plt.ylim([0.3, 1])
            #     else:
            #         plt.ylim([0.6, 0.9])
            # if plot_key.split('_')[-1] == 'loss':
            #     plt.ylim([0, 2])

    print("rerun: ")
    print(re_run_file)
    return re_run_file


def collect_csv(root_dir):
    acc_all_dict = {}
    alpha_all_dict = {}
    for file_name in os.listdir(root_dir):
        if file_name[0] == '.':
            continue
        file_info = split_dir_name(file_name, prefix='')
        tmp_df = pd.read_csv(os.path.join(root_dir, file_name))
        tmp_key = ':'.join('{}'.format(item) for item in file_info['client_bid']) + '_' + '{}'.format(file_info['alpha_tau'])
        if file_name.split('.')[0].split('_')[-1] == 'alpha':
            print('**** handling alpha csv')
            tmp_dict = alpha_all_dict
        elif file_name.split('_')[-3] == 'test':
            tmp_dict = acc_all_dict
        else:
            print('**** skipping: {}'.format(file_name))
            continue
        if tmp_key in tmp_dict.keys():
            tmp_dict[tmp_key].append(tmp_df)
        else:
            tmp_dict[tmp_key] = [tmp_df]
    return acc_all_dict, alpha_all_dict

def plot_alpha(alpha_all_dict, plot_sav_dir, min_rep=5):
    check_dir(plot_sav_dir)
    for rep in alpha_all_dict.keys():
        # print('**** plotting: {}'.format(rep))
        rep_times = len(alpha_all_dict[rep])
        if rep_times < min_rep:
            print("{} repeats less then {} times".format(rep, min_rep))
            continue
        # client_num = alpha_all_dict[rep][0]['client'].unique().shape[0]

        tmp = []
        for item in alpha_all_dict[rep][0].keys():
            if item.split('_')[0] == 'client':
                tmp.append(item)

        client_num = len(tmp)

        for client_id in range(1, client_num + 1):
            alpha_mean_list = []
            alpha_std_list = []
            round_list = []
            data_collect = []
            len_col = []
            for rep_id in range(rep_times):
                tmp_data = alpha_all_dict[rep][rep_id]
                data = tmp_data['client_{}'.format(client_id)]
                len_col.append(data.shape[0])
                data_collect.append(data)
            max_plot_round = min(len_col)
            if max_plot_round < MAX2RUN:
                continue
            for i in range(max_plot_round):
                round_list.append(alpha_all_dict[rep][0]['round'][i])
                alpha_mean_list.append(np.mean([rep[i] for rep in data_collect]))
                alpha_std_list.append(np.std([rep[i] for rep in data_collect]))
            plt.plot(round_list, alpha_mean_list, label='client: {}'.format(client_id), color=COLORS[client_id - 1])
            plt.fill_between(round_list,
                             np.array(alpha_mean_list) - np.array(alpha_std_list),
                             np.array(alpha_mean_list) + np.array(alpha_std_list),
                             color=COLORS[client_id - 1], alpha=0.2)

        plt.legend()
        plt.xlabel('Round')
        plt.ylabel('Alpha')
        # plt.title(plot_key)
        plt.grid()
        # if client_num == 3:
        #     plt.ylim([0.5, 0.9])
        # else:
        #     plt.ylim([0.2,0.95])
        plt.ylim([0, 1])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_sav_dir, rep + '_' + 'alpha' + '.pdf'))
        plt.close()

if __name__ == '__main__':
    root_dir = 'results_rep_summary'
    plot_key_list = ['test_acc', 'val_acc']
    plot_sav_dir = 'results_rep_plot'
    acc_all_dict, alpha_all_dict = collect_csv(root_dir)
    plot(acc_all_dict, plot_key_list, plot_sav_dir, min_rep=5)
    plot_alpha(alpha_all_dict, plot_sav_dir, min_rep=5)

