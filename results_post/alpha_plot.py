import pandas as pd
from results_post.results_collection import check_dir
from results_post.results_collection import split_dir_name
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np






COLORS = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'brown', 'tan', 'navy', 'peru', 'orange']
MAXROUND = -1
Linewidth = 3
FontSize = 25
font = {'family': 'sans-serif', 'weight': 'normal', 'size': 22}

matplotlib.rc('font', **font)

MAX2RUN = 50
# client_weight_plot = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

client_weight_plot = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
COLORS = ['cyan', 'red', 'peru', 'blue', 'orange', 'green', 'navy', 'brown', 'purple', 'black', 'tan']


train_weight_2_candidate =[[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6,0.4], [0.4, 0.6],
                      [0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.0, 1.0], [1.0, 0.0]]
train_weight_3_candidate=[[0.7, 0.1, 0.2], [0.3,0.3,0.4], [0.1, 0.2, 0.7],
                              [0.1,0.7, 0.2], [0.5, 0.3, 0.2], [0.2,0.5,0.3],
                              [0.2, 0.3,0.5]]









class ResultsPlot():
    def __init__(self, plot_sav_dir, root_dir, plot_key_list, min_rep=5):
        self.plot_sav_dir = plot_sav_dir
        self.root_dir = root_dir
        self.plot_key_list = plot_key_list
        self.min_rep = min_rep
        check_dir(plot_sav_dir)

    def plot_clientwise_diff_train_weight(self, acc_all_dict, plot_key_list=None, plot_sav_dir=None, min_rep=None):
        import itertools
        client_bid_set = []
        tau_set = []
        for item in acc_all_dict.keys():
            client_bid = item.split('_')[-2]
            tau = item.split('_')[-1]
            if client_bid not in client_bid_set:
                client_bid_set.append(client_bid)
            if tau not in tau_set:
                tau_set.append(tau)

        re_run_file = []

        for client_bid, tau in itertools.product(client_bid_set, tau_set):
            client_num = len(client_bid.split(':'))
            for plot_key in plot_key_list:
                for client_id in range(1, client_num + 1):
                    plt.figure(figsize=(10, 8))

                    for rep in acc_all_dict.keys():
                        if rep.split('_')[-2] == client_bid and rep.split('_')[-1] == tau:
                            train_weight = rep.split('_')[-3].split(':')[client_id - 1]
                            # if float(train_weight)*10%2 == 0:
                            #     continue
                            if float(train_weight) not in client_weight_plot:
                                continue
                            print('client bid: {}, plot key; {}, rep: {}'.format(client_bid, plot_key, rep))
                            rep_times = len(acc_all_dict[rep])
                            color_id = int(float(train_weight) * 10)

                            acc_mean_list = []
                            acc_std_list = []
                            round_list = []
                            data_collect = []
                            len_col = []
                            for rep_id in range(rep_times):
                                tmp_data = acc_all_dict[rep][rep_id]
                                data = tmp_data[tmp_data['client'] == client_id][[
                                    'round', plot_key
                                ]].to_numpy()
                                len_col.append(data.shape[0])
                                data_collect.append(data)
                            max_plot_round = min(len_col)
                            if max_plot_round < 5:
                                print('max_plot_round: {}'.format(max_plot_round))
                                continue
                            print('file name: {}, min round: {}'.format(
                                rep, data_collect[0][max_plot_round - 1, 0]))
                            if data_collect[0][max_plot_round - 1, 0] < MAX2RUN:
                                if rep not in re_run_file:
                                    re_run_file.append(rep)
                                continue

                            print('rep:{}, client:{}, color_id: {}'.format(rep, client_id, color_id))

                            for i in range(max_plot_round):
                                round_list.append(data_collect[0][i, 0])
                                acc_mean_list.append(
                                    np.mean([rep[i, 1] for rep in data_collect]))
                                acc_std_list.append(
                                    np.std([rep[i, 1] for rep in data_collect]))
                            plt.plot(round_list,
                                     acc_mean_list,
                                     label='train_weight: {}'.format(rep.split('_')[-3].split(':')[client_id - 1]),
                                     color=COLORS[color_id])
                            plt.fill_between(
                                round_list,
                                np.array(acc_mean_list) - np.array(acc_std_list),
                                np.array(acc_mean_list) + np.array(acc_std_list),
                                color=COLORS[color_id],
                                alpha=0.2)
                            color_id += 1

                    # plt.legend()
                    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                               mode="expand", borderaxespad=0, ncol=2)
                    plt.xlabel('Round')
                    plt.ylabel(plot_key)
                    # plt.title(plot_key)
                    plt.grid()
                    if plot_key.split('_')[-1] == 'f1':
                        plt.ylim([0.1, 0.9])
                    elif plot_key.split('_')[-1] == 'auc':
                        plt.ylim([0.1, 1.0])
                    elif plot_key.split('_')[-1] == 'acc':
                        plt.ylim([0.1, 0.9])
                    else:
                        plt.ylim([0.0, 0.9])

                    # plt.ylim([0.3, 0.9])

                    # if client_num == 3:
                    #     plt.ylim([0.5, 0.9])
                    # else:
                    #     plt.ylim([0.2, 0.95])
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plot_sav_dir,
                                     'client:{}_'.format(client_id)
                                     + 'bid_{}_tau_{}'.format(client_bid, tau)
                                     + '_' + plot_key + '.pdf'))
                    plt.close()

    def plot_alpha(self, alpha_all_dict, plot_sav_dir, min_rep=5):
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
                    alpha_mean_list.append(
                        np.mean([rep[i] for rep in data_collect]))
                    alpha_std_list.append(np.std([rep[i] for rep in data_collect]))
                plt.plot(round_list,
                         alpha_mean_list,
                         label='client: {}'.format(client_id),
                         color=COLORS[client_id - 1])
                plt.fill_between(
                    round_list,
                    np.array(alpha_mean_list) - np.array(alpha_std_list),
                    np.array(alpha_mean_list) + np.array(alpha_std_list),
                    color=COLORS[client_id - 1],
                    alpha=0.2)

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
            print('Saving to ---------- {}'.format(os.path.join(plot_sav_dir, rep + '_' + 'alpha' + '.pdf')))
            plt.savefig(os.path.join(plot_sav_dir, rep + '_' + 'alpha' + '.pdf'))
            plt.close()

    def collect_csv(self, root_dir):
        acc_all_dict = {}
        alpha_all_dict = {}
        for file_name in os.listdir(root_dir):
            if file_name[0] == '.':
                continue
            file_info = split_dir_name(file_name, prefix='')
            tmp_df = pd.read_csv(os.path.join(root_dir, file_name))

            tmp_key = ''
            if file_info['train_weight'] is not None:
                tmp_key = 'train_weight_' + ':'.join(
                    '{}'.format(item)
                    for item in file_info['train_weight']) + '_'
            tmp_key += ':'.join(
                '{}'.format(item)
                for item in file_info['client_bid']) + '_' + '{}'.format(
                file_info['alpha_tau'])
            if file_name.split('.csv')[0].split('_')[-1] == 'alpha':
                print('**** handling alpha csv')
                tmp_dict = alpha_all_dict
            elif file_name.split('_')[-3] == 'test':
                if 'model' in tmp_df.keys():
                    pass
                tmp_dict = acc_all_dict
            else:
                print('**** skipping: {}'.format(file_name))
                continue
            if tmp_key in tmp_dict.keys():
                tmp_dict[tmp_key].append(tmp_df)
            else:
                tmp_dict[tmp_key] = [tmp_df]
        return acc_all_dict, alpha_all_dict

    def plot_no_agg_sha(self, acc_all_dict, plot_key_list, plot_sav_dir, min_rep=5):
        re_run_file = []
        check_dir(plot_sav_dir)
        for rep in acc_all_dict.keys():
            rep_times = len(acc_all_dict[rep])
            if rep_times < min_rep:
                print("{} repeats less then {} times".format(rep, min_rep))
                continue

            client_num = acc_all_dict[rep][0]['client'].unique().shape[0]
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
                        data = tmp_data[tmp_data['client'] == client_id][[
                            'round', plot_key
                        ]].to_numpy()
                        len_col.append(data.shape[0])
                        data_collect.append(data)
                    max_plot_round = min(len_col)
                    print('file name: {}, min round: {}'.format(
                        rep, data_collect[0][max_plot_round - 1, 0]))
                    if data_collect[0][max_plot_round - 1, 0] < MAX2RUN:
                        if rep not in re_run_file:
                            re_run_file.append(rep)
                        continue

                    for i in range(max_plot_round):
                        round_list.append(data_collect[0][i, 0])
                        acc_mean_list.append(
                            np.mean([rep[i, 1] for rep in data_collect]))
                        acc_std_list.append(
                            np.std([rep[i, 1] for rep in data_collect]))
                    plt.plot(round_list,
                             acc_mean_list,
                             label='client: {}'.format(client_id),
                             color=COLORS[client_id - 1])
                    plt.fill_between(
                        round_list,
                        np.array(acc_mean_list) - np.array(acc_std_list),
                        np.array(acc_mean_list) + np.array(acc_std_list),
                        color=COLORS[client_id - 1],
                        alpha=0.2)
                    plot_cnt += 1
                if plot_cnt == client_num:
                    plt.legend()
                    plt.xlabel('Round')
                    plt.ylabel(plot_key)
                    if len(rep.split('_')) > 2:
                        plt.title('train weight: {}'.format(rep.split('_')[2]))
                    else:
                        plt.title('client bid: {}'.format(rep.split('_')[0]))
                    plt.grid()

                    if client_num == 3:
                        plt.ylim([0.5, 0.9])
                    else:
                        if plot_key.split('_')[-1] == 'f1':
                            plt.ylim([0.1, 0.9])
                        elif plot_key.split('_')[-1] == 'auc':
                            plt.ylim([0.1, 1.0])
                        elif plot_key.split('_')[-1] == 'acc':
                            plt.ylim([0.1, 0.9])
                        else:
                            plt.ylim([0.0, 0.9])
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plot_sav_dir, rep + '_' + plot_key + '.pdf'))
                    print('saving to: {}'.format(os.path.join(plot_sav_dir, rep + '_' + plot_key + '.pdf')))
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

    def find_final_model_eval(self, tmp_df):
        pass


    def plot_agg_sha_final_model(self, acc_all_dict, plot_key_list, plot_sav_dir, min_rep=5):
        re_run_file = []

        check_dir(plot_sav_dir)
        model_sel_info = dict()
        for rep in acc_all_dict.keys():
            rep_times = len(acc_all_dict[rep])
            if rep_times < min_rep:
                print("{} repeats less then {} times".format(rep, min_rep))
                continue

            client_num = acc_all_dict[rep][0]['client'].unique().shape[0]
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

                        # find the final model

                        tmp_val_cnt = tmp_data['model'].value_counts()
                        index_id = np.argmax(tmp_val_cnt.values)
                        final_model_id = tmp_val_cnt.index[index_id]
                        tmp_data = tmp_data[tmp_data['model'] == final_model_id]
                        print('final model: {}, cnt: {}'.format(final_model_id,np.max(tmp_val_cnt)))
                        if client_num ==2:
                            agg_weight = train_weight_2_candidate[final_model_id - client_num-1]
                        elif client_num ==3:
                            agg_weight = train_weight_3_candidate[final_model_id - client_num - 1]


                        if rep.split('_')[0] not in model_sel_info.keys():
                            model_sel_info[rep.split('_')[0]] = [agg_weight]
                        else:
                            if len(model_sel_info[rep.split('_')[0]]) ==5:
                                pass
                            else:
                                model_sel_info[rep.split('_')[0]].append(agg_weight)

                        data = tmp_data[tmp_data['client'] == client_id][[
                            'round', plot_key
                        ]].to_numpy()
                        len_col.append(data.shape[0])
                        data_collect.append(data)
                    max_plot_round = min(len_col)
                    print('file name: {}, min round: {}'.format(
                        rep, data_collect[0][max_plot_round - 1, 0]))
                    if data_collect[0][max_plot_round - 1, 0] < MAX2RUN:
                        if rep not in re_run_file:
                            re_run_file.append(rep)
                        continue

                    for i in range(max_plot_round):
                        round_list.append(data_collect[0][i, 0])
                        acc_mean_list.append(
                            np.mean([rep[i, 1] for rep in data_collect]))
                        acc_std_list.append(
                            np.std([rep[i, 1] for rep in data_collect]))
                    plt.plot(round_list,
                             acc_mean_list,
                             label='client: {}'.format(client_id),
                             color=COLORS[client_id - 1])
                    plt.fill_between(
                        round_list,
                        np.array(acc_mean_list) - np.array(acc_std_list),
                        np.array(acc_mean_list) + np.array(acc_std_list),
                        color=COLORS[client_id - 1],
                        alpha=0.2)
                    plot_cnt += 1
                if plot_cnt == client_num:
                    plt.legend()
                    plt.xlabel('Round')
                    plt.ylabel(plot_key)
                    if len(rep.split('_')) > 2:
                        plt.title('train weight: {}'.format(rep.split('_')[2]))
                    else:
                        plt.title('client bid: {}'.format(rep.split('_')[0]))
                    plt.grid()

                    if client_num == 3:
                        plt.ylim([0.1, 0.9])
                    else:
                        if plot_key.split('_')[-1] == 'f1':
                            plt.ylim([0.0, 1.0])
                        elif plot_key.split('_')[-1] == 'auc':
                            plt.ylim([0.0, 1.0])
                        elif plot_key.split('_')[-1] == 'acc':
                            plt.ylim([0.0, 1.0])
                        else:
                            plt.ylim([0.0, 1.0])
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(plot_sav_dir, rep + '_' + plot_key + '.pdf'))
                    print('saving to: {}'.format(os.path.join(plot_sav_dir, rep + '_' + plot_key + '.pdf')))
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
        print(model_sel_info)
        print(pd.DataFrame(model_sel_info))
        return re_run_file

if __name__ == '__main__':
    # root_dir = 'results_rep_batch_sche_multistep_lr_label_splitter'
    # plot_sav_dir = 'results_rep_batch_sche_multistep_lr_label_splitter_plot'

    # root_dir = 'results_rep_epoch_sche_multistep_lr_label_splitter'
    # plot_sav_dir = 'results_rep_epoch_sche_multistep_lr_label_splitter_plot'
    #
    # root_dir = 'results_rep_epoch_sche_multistep_lr_label_splitter_more_metric'
    # plot_sav_dir = 'results_rep_epoch_sche_multistep_lr_label_splitter_plot_more_metric'

    # root_dir = 'results_no_search_epoch_sche_multistep_lr_label_splitter_more_metric_v1_order_bug'
    # plot_sav_dir = 'results_no_search_epoch_sche_multistep_lr_label_splitter_more_metric_v1_order_bug_plot'
    #
    # plot_key_list = ['test_acc', 'val_acc', 'test_f1', 'val_f1', 'test_roc_auc', 'val_roc_auc']
    #
    # root_dir = 'results_v1_order_bug'
    # plot_sav_dir = 'results_v1_order_bug_plot'
    # plot_key_list = ['test_acc', 'val_acc']

    root_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter_no_search'
    plot_sav_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter_no_search_plot'

    # root_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter/'
    # plot_sav_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter_plot/'

    # root_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_lda_splitter/'
    # plot_sav_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_lda_splitter_plot/'

    plot_key_list = ['test_acc', 'val_acc', 'test_f1', 'val_f1']

    min_rep = 5

    # check_dir(plot_sav_dir)

    # plotter.plo(acc_all_dict, plot_key_list, plot_sav_dir, min_rep=4)
    # plot_alpha(alpha_all_dict, plot_sav_dir, min_rep=5)
    # plotter.plot_clientwise_diff_train_weight(acc_all_dict, plot_key_list, plot_sav_dir, min_rep=5)

    plotter = ResultsPlot(plot_sav_dir= plot_sav_dir, root_dir=root_dir, plot_key_list=plot_key_list, min_rep=min_rep)
    acc_all_dict, alpha_all_dict = plotter.collect_csv(root_dir)
    # plotter.plot_clientwise_diff_train_weight(acc_all_dict, plot_key_list, plot_sav_dir, min_rep=5)
    plotter.plot_agg_sha_final_model(acc_all_dict, plot_key_list, plot_sav_dir,min_rep)

