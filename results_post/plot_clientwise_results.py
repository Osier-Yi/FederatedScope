import pandas as pd
import os
import matplotlib.pyplot as plt
import gzip
import pickle

COLORS=['red', 'blue', 'green', 'black']
MAXROUND = -1


train_valid_metric_name = ['train_acc', 'train_avg_loss', 'val_avg_loss_before', 'val_avg_loss_after']
test_valid_metric_name = ['test_acc', 'test_avg_loss', 'val_acc', 'val_avg_loss']

def extract_eval_from_log(file_pth, client_num, plot_sav_dir):
    test_valid_metric = {'round': [], 'client': [], 'test_acc': [], 'test_avg_loss': [], 'val_acc': [],
                         'val_avg_loss': []}
    train_valid_metric = {'round': [], 'client': [], 'train_acc': [], 'train_avg_loss': [], 'val_avg_loss_before': [],
                          'val_avg_loss_after': []}



    if file_pth.split('.')[-1] == 'gz':
        open_func = gzip.open
    else:
        open_func = open

    with open_func(file_pth, 'rb') as f:
        F = f.readlines()
        for idx, line in enumerate(F):
            # Time
            # Statistics
            try:
                results = eval(line)
            except:
                continue
            print(results)
            if results['Role'][-1] == str(client_num):
                client_id = results['Role'].split(' ')[1]
                client_id = int(client_id[1:len(client_id)])
                round = results['Round']

                if 'train_acc' in results['Results_raw'].keys():

                    train_valid_metric['round'].append(round)
                    train_valid_metric['client'].append(client_id)
                    for key in train_valid_metric_name:
                        train_valid_metric[key].append(results['Results_raw'][key])
                if 'test_acc' in results['Results_raw'].keys():
                    test_valid_metric['round'].append(round)
                    test_valid_metric['client'].append(client_id)
                    for key in test_valid_metric_name:
                        test_valid_metric[key].append(results['Results_raw'][key])

    train_valid_metric = pd.DataFrame(train_valid_metric)
    test_valid_metric = pd.DataFrame(test_valid_metric)
    train_valid_metric.to_csv(os.path.join(plot_sav_dir, 'train_valid_metric'), index=False)
    test_valid_metric.to_csv(os.path.join(plot_sav_dir, 'test_valid_metric'), index=False)
    return train_valid_metric, test_valid_metric

def plot_clientwise(file_pth, client_num, plot_sav_dir, client_vals_all):
    # train_valid_metric, test_valid_metric = extract_eval_from_log(file_pth, client_num, plot_sav_dir)



    for plot_key in train_valid_metric_name:
        for client_id in range(1, client_num + 1):
            train_valid_metric
            data = train_valid_metric[train_valid_metric['client'] == client_id]
            if MAXROUND > 0:
                plt.plot(data[data['round']<=MAXROUND]['round'], data[data['round']<=MAXROUND][plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1])
            else:
                plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1])
        plt.legend()
        plt.xlabel('Round')
        plt.ylabel(plot_key)
        if plot_key.split('_')[-1] == 'acc':
            plt.ylim([0,0.9])
        if plot_key.split('_')[-1] == 'loss':
            plt.ylim([0, 2])
        plt.title(plot_key)
        plt.grid()
        plt.savefig(os.path.join(plot_sav_dir, plot_key))
        plt.close()
if __name__ == '__main__':
    method_name_set=['alpha', 'alpha_extrem', 'plain_fedex']
    split_name_set = ['train_valid_metric', 'test_valid_metric']
    method_results_pth = {'alpha': 'Alpha_tune_results/results_3_clients_alpha',
                          'alpha_extrem': 'Alpha_tune_results/results_3_clients_alpha_extreme',
                          'plain_fedex': 'Alpha_tune_results/results_3_clients_fedex'}

    train_valid_metric_name = ['train_acc', 'train_avg_loss', 'val_avg_loss_before', 'val_avg_loss_after']
    test_valid_metric_name = ['test_acc', 'test_avg_loss', 'val_acc', 'val_avg_loss']
    client_num = 3
    plot_sav_dir = 'Alpha_tune_results/3_clients_summary'

    fedex_split_name_set = ['train_valid_metric', 'valid_metric', 'test_valid_metric']

    fedex_train_valid_metric_name = ['train_acc', 'train_avg_loss', 'val_avg_loss_before', 'val_avg_loss_after']
    fedex_test_metric_name = ['test_acc', 'test_avg_loss']
    fedex_valid_metric_name = ['val_acc', 'val_avg_loss']
    fedex_key_list = [fedex_train_valid_metric_name, fedex_valid_metric_name, fedex_test_metric_name]


    method_eval_data = {}
    for method_name in method_name_set:
        method_eval_data[method_name] = {}
        if method_name == 'plain_fedex':
            tmp = fedex_split_name_set
        else:
            tmp = split_name_set
        for split_name in tmp:
            method_eval_data[method_name][split_name] = \
                pd.read_csv(os.path.join(method_results_pth[method_name], split_name))


    split_set_key_list = [train_valid_metric_name, test_valid_metric_name]
    for idx in range(len(split_set_key_list)):
        key_list = split_set_key_list[idx]
        for plot_key in key_list:
            for client_id in range(1, client_num + 1):
                for method_name in method_name_set:
                    print('work on {}_{}_{}'.format(plot_key, client_id, method_name))
                    if method_name == 'plain_fedex':
                        for id in range(len(fedex_split_name_set)):
                            if plot_key in fedex_key_list[id]:
                                split_set_id = id
                                print('split_set_id:{}'.format(split_set_id))
                                split_name = fedex_split_name_set[split_set_id]
                                break

                    else:
                        split_set_id = idx
                        split_name = split_name_set[split_set_id]
                    data = method_eval_data[method_name][split_name]
                    data = data[data['client'] == client_id]
                    if MAXROUND > 0:
                        plt.plot(data[data['round'] <= MAXROUND]['round'], data[data['round'] <= MAXROUND][plot_key], label='method: {}'.format(method_name))
                    else:
                        plt.plot(data['round'], data[plot_key], label='method: {}'.format(method_name))
                plt.legend()
                plt.xlabel('Round')
                plt.ylabel(plot_key)
                if plot_key.split('_')[-1] == 'acc':
                    plt.ylim([0, 0.9])
                if plot_key.split('_')[-1] == 'loss':
                    plt.ylim([0, 2])
                plt.title('{}_{}'.format(plot_key, client_id))
                plt.grid()
                plt.savefig(os.path.join(plot_sav_dir, '{}_{}_{}'.format(plot_key, plot_key, client_id)))
                plt.close()






