import numpy as np
import pandas as pd
import os

Round = 395


def table_data_collect(dir_pth, dir_mapping=None):
    table_txt = 'client id & price & alpha & Test Acc & Val Acc & Train Acc\\\ \n'
    for dir_name in os.listdir(dir_pth):
        print(dir_name)
        if dir_name[0] != '.':
            if 'valid_metric' not in os.listdir(os.path.join(
                    dir_pth, dir_name)):
                if dir_mapping is not None:
                    pass
                else:
                    table_txt += '{} & '.format(dir_name)
                tmp = pd.read_csv(
                    os.path.join(dir_pth, dir_name, 'test_valid_metric'))
                tmp_train = pd.read_csv(
                    os.path.join(dir_pth, dir_name, 'train_valid_metric'))
                alpha_pd = pd.read_csv(os.path.join(dir_pth, dir_name,
                                                    'alpha'))
                # data = {}
                client_num = len(tmp['client'].unique())
                # print(client_num)
                for client_id in range(1, client_num + 1):
                    data = []
                    # print(tmp[(tmp['round']==Round) & (tmp['client']==client_id)]['test_acc'].iloc[0])

                    data.append(
                        alpha_pd[(alpha_pd['round'] == Round
                                  )]['client_{}'.format(client_id)].iloc[0])
                    data.append(tmp[(tmp['round'] == Round) & (
                        tmp['client'] == client_id)]['test_acc'].iloc[0])
                    data.append(
                        tmp['val_acc'][(tmp['round'] == Round)
                                       & (tmp['client'] == client_id)].iloc[0])
                    data.append(tmp_train['train_acc']
                                [(tmp_train['round'] == Round)
                                 & (tmp_train['client'] == client_id)].iloc[0])
                    # print(data)
                    if dir_mapping is not None:
                        table_txt += '{} & {} & '.format(
                            client_id, dir_mapping[dir_name][client_id - 1])
                        table_txt += ' & '.join('{:.4f}'.format(item)
                                                for item in data)
                    else:
                        table_txt += ' & '.join('{:.4f}'.format(item)
                                                for item in data)
                    table_txt += '\\\ \n'
                    if client_id != client_num and dir_mapping is None:
                        table_txt += '&'
                table_txt += '\midrule \n'

    print(table_txt)


if __name__ == '__main__':
    dir = 'Alpha_tune_new_version_debug_results_consistent_label'
    dir_mapping = {
        'results_exp_ls_epoch_alpha_tune_3_clients_extrem_entropy_tau_alpha_consistent_label': [
            'E10', 'E20', 'E1000'
        ],
        'results_exp_ls_epoch_alpha_tune_2_clients_1000_0_entropy_tau_alpha_consistent_label': [
            'E1000', 'E0'
        ],
        'results_exp_ls_epoch_alpha_tune_2_clients_1000_0_consistent_label': [
            1000, 0
        ],
        'results_exp_ls_epoch_alpha_tune_2_clients_consist_label': [0, 1000],
        'results_exp_ls_epoch_alpha_tune_3_clients_1000_20_10_consistent_label': [
            1000, 20, 10
        ],
        'results_exp_ls_epoch_alpha_tune_3_clients_consistent_label': [
            10, 20, 50
        ],
        'results_exp_ls_epoch_alpha_tune_3_clients_extrem_consistent_label': [
            10, 20, 1000
        ],
        'results_exp_ls_epoch_alpha_tune_3_clients_zeros_consistent_label': [
            0, 0, 0
        ],
        'results_exp_ls_epoch_alpha_tune_3_clients_50_20_10_consistent_label': [
            50, 20, 10
        ],
        'results_exp_ls_epoch_alpha_tune_3_clients_50_20_10_consistent_label_entropy_tau_alpha': [
            'E50', 'E20', 'E10'
        ],
        'results_exp_ls_epoch_alpha_tune_2_clients_zeros_consistent_label': [
            0, 0
        ],
        'results_exp_ls_epoch_alpha_tune_3_clients_1000_20_10_consistent_label_entropy_tau_alpha': [
            'E1000', 'E20', 'E10'
        ]
    }
    table_data_collect(dir, dir_mapping)
