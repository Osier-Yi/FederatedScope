import pandas as pd
import os
import matplotlib.pyplot as plt
import gzip
import pickle
import matplotlib



def check_dir(pth):
    if os.path.exists(pth):
        pass
    else:
        os.mkdir(pth)

# def read_eva_logs(lines):
#     fairness_list = []
#     for line in lines:
#         tmp_line = str(line)
#         if 'Server' in tmp_line:
#             try:
#                 results = eval(line)
#             except:
#                 tmp_line = tmp_line.replace("nan", '0')
#                 # tmp_line.replace('\\', '')
#                 tmp_line = tmp_line[2:len (tmp_line)]
#                 tmp = tmp_line[0:-3].encode()
#                 results = eval(tmp)
#                 # results = eval(str.encode(tmp_line))
#                 # print(tmp_line)
#             new_results = {}
#             for key in results['Results_raw']:
#                 new_results[f'{key}_fair'] = results['Results_raw'][key]
#             fairness_list.append(new_results)
#     return fairness_list

COLORS=['red', 'blue', 'green', 'black']
MAXROUND = -1
Linewidth = 3
FontSize = 25
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)



def plot_alpha_tune_result(file_pth, client_num, plot_sav_dir, file_sav_prefix=None):

    check_dir(plot_sav_dir)
    test_valid_metric = {'round': [], 'client': [], 'test_acc': [], 'test_avg_loss': [], 'val_acc': [],
                         'val_avg_loss': []}
    train_valid_metric = {'round': [], 'client': [], 'train_acc': [], 'train_avg_loss': [], 'val_avg_loss_before': [],
                          'val_avg_loss_after': []}

    train_valid_metric_name = ['train_acc', 'train_avg_loss', 'val_avg_loss_before', 'val_avg_loss_after']
    test_valid_metric_name = ['test_acc', 'test_avg_loss', 'val_acc', 'val_avg_loss']

    if file_pth.split('.')[-1] == 'gz':
        open_func = gzip.open
    else:
        open_func = open


    with open_func(file_pth, 'rb') as f:
        global_model_idx = client_num + 1
        F = f.readlines()
        for idx, line in enumerate(F):
            # Time
            # Statistics
            try:
                results = eval(line)
            except:
                continue
            # print(results)
            if results['Role'][-1] == str(global_model_idx):
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
    # train_valid_metric.to_csv(os.path.join(plot_sav_dir, file_sav_prefix + '_'+ 'train_valid_metric'),index=False)
    # test_valid_metric.to_csv(os.path.join(plot_sav_dir, file_sav_prefix + '_'+ 'test_valid_metric'),index=False)
    # print(train_valid_metric)
    # print(test_valid_metric)

    for plot_key in train_valid_metric_name:
        for client_id in range(1, client_num + 1):
            data = train_valid_metric[train_valid_metric['client'] == client_id]
            if MAXROUND > 0:
                plt.plot(data[data['round']<=MAXROUND]['round'],
                         data[data['round']<=MAXROUND][plot_key],
                         label='client: {}'.format(client_id),
                         color = COLORS[client_id-1],
                         linewidth = Linewidth)
            else:
                plt.plot(data['round'],
                         data[plot_key],
                         label='client: {}'.format(client_id),
                         color = COLORS[client_id-1],
                         linewidth = Linewidth)
        plt.legend()
        plt.xlabel('Round')
        plt.ylabel(plot_key)
        if plot_key.split('_')[-1] == 'acc':
            if plot_key.split('_')[0] == 'train':
                plt.ylim([0.6, 1])
            elif client_num == 2:
                plt.ylim([0.3, 1])
            else:
                plt.ylim([0.6, 0.9])
        if plot_key.split('_')[-1] == 'loss':
            plt.ylim([0, 2])
        # plt.title(plot_key)
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_sav_dir, file_sav_prefix + '_' + plot_key + '.pdf'))
        plt.close()

    for plot_key in test_valid_metric_name:
        for client_id in range(1, client_num + 1):
            data = test_valid_metric[test_valid_metric['client'] == client_id]
            if MAXROUND > 0:
                plt.plot(data[data['round']<=MAXROUND]['round'],
                         data[data['round']<=MAXROUND][plot_key],
                         label='client: {}'.format(client_id),
                         color = COLORS[client_id-1],
                         linewidth = Linewidth)
            else:
                plt.plot(data['round'],
                         data[plot_key],
                         label='client: {}'.format(client_id),
                         color = COLORS[client_id-1],
                         linewidth = Linewidth)
            # plt.plot(data['round'], data[plot_key], label='client: {}'.format(client_id), color = COLORS[client_id-1])
        plt.legend()
        plt.xlabel('Round')
        plt.ylabel(plot_key)
        # plt.title(plot_key)
        plt.grid()
        if plot_key.split('_')[-1] == 'acc':
            if plot_key.split('_')[0] == 'train':
                plt.ylim([0.5, 1])
            elif client_num == 2:
                plt.ylim([0.3, 1])
            else:
                plt.ylim([0.6, 0.9])
        if plot_key.split('_')[-1] == 'loss':
            plt.ylim([0, 2])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_sav_dir, file_sav_prefix + '_' + plot_key + '.pdf'))
        plt.close()

def plot_alpha(alpha_pth, num_client, plot_sav_dir, file_sav_prefix=None):
    check_dir(plot_sav_dir)
    try:
        with open(alpha_pth, 'rb') as handle:
            alpha = pickle.load(handle)
    except:
        print('no such file')
        return


    # print(alpha)

    tmp_alpha = []
    round = []
    client_id_list = range(1, num_client+1)
    alpha_df = {'round': []}
    for i in client_id_list:
        alpha_df['client_{}'.format(i)] = []

    for key in alpha.keys():
        alpha_df['round'].append(key)
        for i in client_id_list:
            alpha_df['client_{}'.format(i)].append(alpha[key].flatten()[i-1])



    alpha_df = pd.DataFrame(alpha_df)
    # alpha_df.to_csv(os.path.join(plot_sav_dir, file_sav_prefix + '_'+ 'alpha'))
    # print(alpha_df)
    print("round 395: {}".format(alpha_df[alpha_df['round']==395]))

    for client_id in client_id_list:
        if MAXROUND > 0:
            plt.plot(alpha_df[alpha_df['round']<=MAXROUND]['round'],
                     alpha_df[alpha_df['round']<=MAXROUND]['client_{}'.format(client_id)],
                     label = 'client_{}'.format(client_id),
                     color = COLORS[client_id-1],
                     linewidth=Linewidth)
        else:
            plt.plot(alpha_df['round'],
                     alpha_df['client_{}'.format(client_id)],
                     label = 'client_{}'.format(client_id),
                     color = COLORS[client_id-1],
                     linewidth = Linewidth
                     )
    plt.xlabel('Round')
    plt.ylabel('Alpha Value')
    plt.grid()
    plt.legend()
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(os.path.join(plot_sav_dir, file_sav_prefix + '_'+ 'Alpha_value' + '.pdf'))
    plt.close()

def clientwise_plot():
    pass

def plot_dir_wise(root_dir, prefix_name=None, sav_dir=None, filter_method_list=(), summary=True):
    check_dir(sav_dir)
    for dir_name in os.listdir(root_dir):
        if len(dir_name) >= len(prefix_name) and dir_name[0:len(prefix_name)] == prefix_name:
            method_name = dir_name.split(prefix_name)[1].split('_')[1]
            if method_name not in filter_method_list:
                file_pth = os.path.join(root_dir, dir_name)
                key_name = 'results_' + dir_name
                file_sav_prefix = '_'.join('{}'.format(item) for item in dir_mapping[key_name])
                print('-' * 10 + ' handling: {} '.format(os.path.join(root_dir, dir_name) + '-' * 10))
                idx = -1
                for i in range(len(dir_name.split('_'))):
                    if dir_name.split('_')[i] == 'clients' :
                        idx = i - 1
                        break
                if idx < 0:
                    ValueError('No client number available!')
                else:
                    client_num = eval(dir_name.split('_')[idx])
                    print('client_num: {}'.format(client_num))
                for dirs in os.listdir(file_pth):
                    if dirs[0]=='.':
                        continue
                    file_pth = os.path.join(file_pth, dirs)
                    for file in os.listdir(file_pth):
                        if file.split('.')[0] == 'eval_results' and (file.split('.')[-1] == 'raw' or  file.split('.')[-1] == 'gz'):
                            final_pth = os.path.join(file_pth, file)
                            if summary:
                                plot_sav_dir = sav_dir
                            else:
                                plot_sav_dir = os.path.join(sav_dir, 'results_' + dir_name)
                            check_dir(plot_sav_dir)
                            plot_alpha_tune_result(final_pth, client_num, plot_sav_dir, file_sav_prefix)
                            alpha_pth = os.path.join(file_pth, 'alpha.pickle')
                            plot_alpha(alpha_pth, client_num, plot_sav_dir, file_sav_prefix)



                        # client_num = 3
                # overall_model_id = client_num
                # plot_sav_dir = 'Alpha_tune_results/results_ls_3_clients_alpha/'
                # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
                #
                # alpha_pth = 'exp_ls_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/alpha.pickle'
                # plot_alpha(alpha_pth, client_num, plot_sav_dir)




if __name__ == '__main__':
    # root_dir = 'exp_results_from_server_new_version_debug'
    # prefix_name = 'exp_ls_epoch'
    # filter_method = ['fedex']
    # sav_dir = 'Alpha_tune_new_version_debug_results'
    # plot_dir_wise(root_dir,prefix_name, sav_dir, filter_method)

    dir_mapping = {'results_exp_ls_epoch_alpha_tune_3_clients_extrem_entropy_tau_alpha_consistent_label': ['E10', 'E20', 'E1000'],
                   'results_exp_ls_epoch_alpha_tune_2_clients_1000_0_entropy_tau_alpha_consistent_label': ['E1000', 'E0'],
                   'results_exp_ls_epoch_alpha_tune_2_clients_1000_0_consistent_label': [1000, 0],
                   'results_exp_ls_epoch_alpha_tune_2_clients_consist_label': [0, 1000],
                   'results_exp_ls_epoch_alpha_tune_3_clients_1000_20_10_consistent_label': [1000, 20, 10],
                   'results_exp_ls_epoch_alpha_tune_3_clients_consistent_label': [10, 20, 50],
                   'results_exp_ls_epoch_alpha_tune_3_clients_extrem_consistent_label': [10, 20, 1000],
                   'results_exp_ls_epoch_alpha_tune_3_clients_zeros_consistent_label': [0, 0, 0],
                   'results_exp_ls_epoch_alpha_tune_3_clients_50_20_10_consistent_label': [50, 20, 10],
                   'results_exp_ls_epoch_alpha_tune_3_clients_50_20_10_consistent_label_entropy_tau_alpha': ['E50', 'E20', 'E10'],
                   'results_exp_ls_epoch_alpha_tune_2_clients_zeros_consistent_label': [0, 0],
                   'results_exp_ls_epoch_alpha_tune_3_clients_1000_20_10_consistent_label_entropy_tau_alpha': ['E1000', 'E20', 'E10'],
                   'results_exp_ls_epoch_alpha_tune_2_clients_0_1000_consistent_label': [0, 1000],
                   'results_exp_ls_epoch_alpha_tune_2_clients_0_100_consistent_label': [0, 100],
                   'results_exp_ls_epoch_alpha_tune_2_clients_0_10_consistent_label': [0, 10],
                   'results_exp_ls_epoch_alpha_tune_2_clients_0_50_consistent_label': [0, 50],
                   'results_exp_ls_epoch_alpha_tune_2_clients_100_0_consistent_label': [100,0],
                   'results_exp_ls_epoch_alpha_tune_2_clients_10_0_consistent_label': [10,0],
                   'results_exp_ls_epoch_alpha_tune_2_clients_50_0_consistent_label': [50,0]

                   }

    root_dir = 'exp_consistent'
    prefix_name = 'exp_ls_epoch'
    filter_method = ['fedex']
    # sav_dir = 'Alpha_tune_new_version_debug_results_consistent_label'
    sav_dir = 'results_plot_summary'
    plot_dir_wise(root_dir, prefix_name, sav_dir, filter_method)




    # # alpha tune 3 clients
    # file_pth = 'exp_alpha_tune/FedAvg_convnet2_on_CIFAR10@torchvision_lr0' \
    #            '.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_3_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    #
    #
    # alpha_pth = 'exp_alpha_tune/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    #
    # # alpha_tune 2 clients
    # file_pth = 'exp_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10' \
    #            '@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 2
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_2_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10' \
    #             '@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 3 clients extreme
    # file_pth = 'exp_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10' \
    #            '@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_3_clients_alpha_extreme/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10' \
    #             '@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)


    # -----------------------------------------------------
    # plots for large ss
    # -----------------------------------------------------
    # alpha tune 3 clients
    # file_pth = 'exp_ls_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0' \
    #            '.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_3_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    #
    # alpha_pth = 'exp_ls_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 2 clients
    # file_pth = 'exp_ls_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10' \
    #            '@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 2
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_2_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_ls_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10' \
    #             '@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 3 clients extreme
    # file_pth = 'exp_ls_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10' \
    #            '@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_3_clients_alpha_extreme/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_ls_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10' \
    #             '@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)




    # -----------------------------------------------------
    # plots for large ss epoch
    # -----------------------------------------------------

    # file_pth = 'exp_ls_epoch_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0' \
    #            '.5_lstep1/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_epoch_3_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    #
    # alpha_pth = 'exp_ls_epoch_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep1/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 2 clients
    # file_pth = 'exp_ls_epoch_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/eval_results.raw.gz'
    # client_num = 2
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_epoch_2_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_ls_epoch_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10' \
    #             '@torchvision_lr0.5_lstep2/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 3 clients extreme
    # file_pth = 'exp_ls_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_epoch_3_clients_alpha_extreme/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_ls_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)

    # -----------------------------------------------------
    # plots for large ss epoch new version
    # -----------------------------------------------------




    # file_pth = 'exp_ls_epoch_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0' \
    #            '.5_lstep1/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_epoch_3_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    #
    # alpha_pth = 'exp_ls_epoch_alpha_tune_3_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep1/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 2 clients
    # file_pth = 'exp_ls_epoch_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/eval_results.raw.gz'
    # client_num = 2
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_epoch_2_clients_alpha/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_ls_epoch_alpha_tune_2_clients/FedAvg_convnet2_on_CIFAR10' \
    #             '@torchvision_lr0.5_lstep2/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)
    #
    # # alpha_tune 3 clients extreme
    # file_pth = 'exp_ls_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/eval_results.raw.gz'
    # client_num = 3
    # overall_model_id = client_num
    # plot_sav_dir = 'Alpha_tune_results/results_ls_epoch_3_clients_alpha_extreme/'
    # plot_alpha_tune_result(file_pth, client_num, plot_sav_dir)
    # alpha_pth = 'exp_ls_alpha_tune_3_clients_extrem/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep20/alpha.pickle'
    # plot_alpha(alpha_pth, client_num, plot_sav_dir)























