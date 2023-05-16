import pandas as pd
import os
import matplotlib.pyplot as plt
import gzip
import matplotlib

COLORS=['red', 'blue', 'green', 'black']
MAXROUND = -1
Linewidth = 3
FontSize = 25
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
from results_post.results_plot import check_dir

def plot_regular_fedex_result(file_pth, client_num, plot_sav_dir):
    check_dir(plot_sav_dir)
    test_metric = {'round': [], 'client': [], 'test_acc': [], 'test_avg_loss': []}
    train_valid_metric = {'round': [], 'client': [], 'train_acc': [], 'train_avg_loss': [], 'val_avg_loss_before': [],
                          'val_avg_loss_after': []}
    val_metric = {'round':[], 'client': [], 'val_acc': [], 'val_avg_loss': []}

    train_valid_metric_name = ['train_acc', 'train_avg_loss', 'val_avg_loss_before', 'val_avg_loss_after']
    test_metric_name = ['test_acc', 'test_avg_loss']
    valid_metric_name = ['val_acc', 'val_avg_loss']

    # with open(file_pth, 'rb') as f:

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
            # print(results)
            if results['Role'].split(' ')[0] == 'Server':
                continue
            client_id = int(results['Role'].split('#')[1])
            # client_id = int(client_id[1:len(client_id)])

            round = results['Round']

            if 'train_acc' in results['Results_raw'].keys():

                train_valid_metric['round'].append(round)
                train_valid_metric['client'].append(client_id)
                for key in train_valid_metric_name:
                    train_valid_metric[key].append(results['Results_raw'][key])
            if 'test_acc' in results['Results_raw'].keys():
                test_metric['round'].append(round)
                test_metric['client'].append(client_id)
                for key in test_metric_name:
                    test_metric[key].append(results['Results_raw'][key])
            if 'val_acc' in results['Results_raw'].keys():
                val_metric['round'].append(round)
                val_metric['client'].append(client_id)
                for key in valid_metric_name:
                    val_metric[key].append(results['Results_raw'][key])



    train_valid_metric = pd.DataFrame(train_valid_metric)
    test_valid_metric = pd.DataFrame(test_metric)
    val_metric = pd.DataFrame(val_metric)
    train_valid_metric.to_csv(os.path.join(plot_sav_dir, 'train_valid_metric'),index=False)
    test_valid_metric.to_csv(os.path.join(plot_sav_dir, 'test_valid_metric'),index=False)
    val_metric.to_csv(os.path.join(plot_sav_dir, 'valid_metric'), index=False)
    # print(train_valid_metric)
    # print(test_valid_metric)

    for plot_key in train_valid_metric_name:
        for client_id in range(1, client_num + 1):
            data = train_valid_metric[train_valid_metric['client'] == client_id]
            if MAXROUND>0:
                plt.plot(data[data['round']<=MAXROUND]['round'], data[data['round']<=MAXROUND][plot_key], label='client: {}'.format(client_id),
                         color=COLORS[client_id - 1], linewidth = Linewidth)
            else:
                plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1], linewidth = Linewidth)
        plt.legend()
        plt.xlabel('Round')
        plt.ylabel(plot_key)
        plt.title(plot_key)
        plt.grid()
        if plot_key.split('_')[-1] == 'acc':
            if plot_key.split('_')[0] == 'train':
                plt.ylim([0.3, 1])
            else:
                plt.ylim([0.3, 0.9])
        if plot_key.split('_')[-1] == 'loss':
            plt.ylim([0, 2])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_sav_dir, plot_key))
        plt.close()

    for plot_key in test_metric_name:
        for client_id in range(1, client_num + 1):
            data = test_valid_metric[test_valid_metric['client'] == client_id]
            if MAXROUND>0:
                plt.plot(data[data['round']<=MAXROUND]['round'], data[data['round']<=MAXROUND][plot_key], label='client: {}'.format(client_id),
                         color=COLORS[client_id - 1],linewidth = Linewidth)
            else:
                plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1],linewidth = Linewidth)
            # plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1])
        plt.legend()
        plt.xlabel('Round')
        plt.ylabel(plot_key)
        if plot_key.split('_')[-1] == 'acc':
            if plot_key.split('_')[0] == 'train':
                plt.ylim([0.3, 1])
            else:
                plt.ylim([0.3, 0.9])
        if plot_key.split('_')[-1] == 'loss':
            plt.ylim([0, 2])
        plt.title(plot_key)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_sav_dir, plot_key))
        plt.close()

    for plot_key in valid_metric_name:
        for client_id in range(1, client_num + 1):
            data = val_metric[val_metric['client'] == client_id]
            if MAXROUND>0:
                plt.plot(data[data['round']<=MAXROUND]['round'], data[data['round']<=MAXROUND][plot_key], label='client: {}'.format(client_id),
                         color=COLORS[client_id - 1],linewidth = Linewidth)
            else:
                plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1],linewidth = Linewidth)
            # plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id),  color = COLORS[client_id-1])
        plt.legend()
        plt.xlabel('Round')
        plt.ylabel(plot_key)
        if plot_key.split('_')[-1] == 'acc':
            if plot_key.split('_')[0] == 'train':
                plt.ylim([0.3, 1])
            else:
                plt.ylim([0.3, 0.9])
        if plot_key.split('_')[-1] == 'loss':
            plt.ylim([0, 2])
        plt.title(plot_key)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_sav_dir, plot_key))
        plt.close()

if __name__ == '__main__':
    # file_pth = 'exp_fedex_2_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 2
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_2_clients_fedex/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)
    #
    # file_pth = 'exp_fedex/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_3_clients_fedex/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)


    # -----------------------------------------------------------
    # plot for ls case
    # -----------------------------------------------------------



    # file_pth = 'exp_ls_fedex_2_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 2
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_2_clients_fedex/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)
    #
    # file_pth = 'exp_ls_fedex/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_3_clients_fedex/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)



    # -----------------------------------------------------------
    # plot for ls no dataloader.batch_size case
    # -----------------------------------------------------------

    # file_pth = 'exp_ls_fedex_lr_lub/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_3_clients_fedex_lr_lub/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)



    # -----------------------------------------------------------
    # plot for ls no dataloader.batch_size case
    # -----------------------------------------------------------

    # file_pth = 'exp_results_from_server_new_version_debug/exp_ls_epoch_fedex/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/eval_results.raw.gz'
    # client_num = 3
    # # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_new_version_debug_results/results_exp_ls_epoch_fedex_3_clients/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)
    #
    # file_pth = 'exp_results_from_server_new_version_debug/exp_ls_epoch_fedex_2_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/eval_results.raw.gz'
    # client_num = 2
    # # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_new_version_debug_results/results_exp_ls_epoch_fedex_2_clients/'
    # plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)

    # -----------------------------------------------------------
    # plot for ls no dataloader.batch_size case
    # -----------------------------------------------------------

    file_pth = 'exp_consistent/exp_ls_epoch_fedex_consistent_label/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/eval_results.raw.gz'
    client_num = 3
    # overall_model_id = client_num
    plot_sav_dir = 'Alpha_tune_new_version_debug_results_consistent_label/results_exp_ls_epoch_fedex_consistent_label/'
    plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)

    file_pth = 'exp_consistent/exp_ls_epoch_fedex_2_clients_consistent_label/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/eval_results.raw.gz'
    client_num = 2
    # overall_model_id = client_num
    plot_sav_dir = 'Alpha_tune_new_version_debug_results_consistent_label/results_exp_ls_epoch_fedex_2_clients_consistent_label/'
    plot_regular_fedex_result(file_pth, client_num, plot_sav_dir)





