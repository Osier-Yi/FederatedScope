import pandas as pd
from results_post.results_collection import check_dir
from results_post.results_collection import split_dir_name
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas






COLORS = ['red', 'blue', 'green', 'black', 'cyan', 'purple', 'brown', 'tan', 'navy', 'peru', 'orange']
MAXROUND = -1
Linewidth = 3
FontSize = 25
font = {'family': 'sans-serif', 'weight': 'normal', 'size': 22}

matplotlib.rc('font', **font)

MAX2RUN = 50
# client_weight_plot = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

# client_weight_plot = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]
COLORS = ['cyan', 'red', 'peru', 'blue', 'orange', 'green', 'navy', 'brown', 'purple', 'black', 'tan']
train_weight_2_candidate =[[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.6,0.4], [0.4, 0.6],
                      [0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.0, 1.0], [1.0, 0.0]]

# train_weight_3_candidate=[[0.7, 0.1, 0.2], [0.3,0.3,0.4], [0.1, 0.2, 0.7],
#                               [0.1,0.7, 0.2], [0.5, 0.3, 0.2], [0.2,0.5,0.3],
#                               [0.2, 0.3,0.5]]
train_weight_3_candidate = [[0.7, 0.1, 0.2], [0.3, 0.3, 0.4], [0.1, 0.2, 0.7],
                                [0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.5, 0.3],
                                [0.2, 0.3, 0.5], [0.8, 0.1,0.1], [0.1,0.8,0.1], [0.1,0.1,0.8], [0.5,0.4,0.1]]

def get_alpha(pth = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter/2_client_0:0_seed_111_tmp_alpha_1_alpha.csv'):
    alpha_pd = pd.read_csv(pth)
    # print(alpha_pd)
    return alpha_pd

def get_val(pth = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter/2_client_0:0_seed_111_tmp_alpha_1_test_valid_metric.csv'):
    val_pd = pd.read_csv(pth)
    model_id_set = pd.unique(val_pd['model'])
    val_dict = dict()
    for model_id in model_id_set:
        val_dict[model_id] = val_pd[val_pd['model']==model_id]
    return val_dict


def plot_metric(val_dict,sav_dir='results_sha/cat_spliter_sha_plot', bid='0:0', seed = 111, plot_metric = 'val_acc'):
    client_id_set = pd.unique(val_dict[list(val_dict.keys())[0]]['client'])
    client_num = len(client_id_set)
    if client_num ==2:
        candidate_weight = train_weight_2_candidate
    elif client_num == 3:
        candidate_weight = train_weight_3_candidate
    for client_id in client_id_set:
        for model_id in val_dict.keys():
            tmp = val_dict[model_id]
            tmp_client = tmp[tmp['client']==client_id]
            if tmp_client.shape[0] ==1:
                plt.scatter(tmp_client['round'], tmp_client[plot_metric], color = COLORS[model_id-client_num-1],label = candidate_weight[model_id-client_num-1] )
            else:
                plt.plot(tmp_client['round'].iloc[0:12], tmp_client[plot_metric].iloc[0:12], color = COLORS[model_id-client_num-1], label = candidate_weight[model_id-client_num-1])
        plt.legend(fontsize=15)
        plt.xlabel('Round')
        plt.ylabel(plot_metric)
        plt.grid()
        plt.tight_layout()
        # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=5)
        # plt.tight_layout()
        plt.title('bid: {}, seed:{}, clientID: {}'.format(bid, seed, client_id))
        print(bid)
        plt.savefig(os.path.join(sav_dir, '{}_client_bid_{}_seed_{}_clientID_{}_metric_{}.png'.format(len(bid.split(':')), bid, seed, client_id, plot_metric)))
        plt.close()

def plot_sha(alpha_pd,val_dict,sav_dir='results_sha/cat_spliter_sha_plot', bid='0:0', seed = 111):
    client_id_set = pd.unique(val_dict[list(val_dict.keys())[0]]['client'])
    client_num = len(client_id_set)
    if client_num ==2:
        candidate_weight = train_weight_2_candidate
    elif client_num == 3:
        candidate_weight = train_weight_3_candidate
    for client_id in client_id_set:
        for model_id in val_dict.keys():
            tmp = val_dict[model_id]
            tmp_client = tmp[tmp['client']==client_id]
            if tmp_client.shape[0] ==1:
                plt.scatter(tmp_client['round'], tmp_client['val_f1'], color = COLORS[model_id-client_num-1],label = candidate_weight[model_id-client_num-1] )
            else:
                plt.plot(tmp_client['round'][0:50], tmp_client['val_f1'][0:50], color = COLORS[model_id-client_num-1], label = candidate_weight[model_id-client_num-1])
        plt.legend(fontsize=15)
        plt.xlabel('Round')
        plt.ylabel('Val_F1')
        plt.grid()
        plt.tight_layout()
        # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=5)
        # plt.tight_layout()
        plt.title('bid: {}, seed:{}, clientID: {}'.format(bid, seed, client_id))
        print(bid)
        plt.savefig(os.path.join(sav_dir, '{}_client_bid_{}_seed_{}_clientID_{}.png'.format(len(bid.split(':')), bid, seed, client_id)))
        plt.close()

    for model_id in val_dict.keys():
        round = []
        tmp = val_dict[model_id]
        sha_val_list = []

        round_list = np.sort(pd.unique(tmp['round']))
        # print(round_list)

        for round_id in round_list[0:50]:
            if round_id <2:
                continue
            round.append(round_id)
            tmp_val_sha = 0
            for client_id in client_id_set:
                # print(tmp)
                # print(tmp[(tmp['round'] == round_id) & (tmp['client'] == client_id)]['val_f1'])
                client_id_val = tmp[(tmp['round'] == round_id) & (tmp['client'] == client_id)]['val_f1'].to_numpy()[0]
                # print(client_id_val)
                # print(alpha_pd.keys())
                # print(alpha_pd[alpha_pd['round']==round_id]['client_{}'.format(client_id)])
                client_id_alpha = alpha_pd[alpha_pd['round']==round_id]['client_{}'.format(client_id)].to_numpy()[0]
                # print(client_id_alpha)
                tmp_val_sha += client_id_val * client_id_alpha
            sha_val_list.append(tmp_val_sha)
            print('round: {}, model: {}, sha_val{}, weight: {}'.format(round_id, model_id, tmp_val_sha, candidate_weight[model_id-client_num-1]))
        # print(sha_val_list)

        if len(round)==1:
            plt.scatter(round, sha_val_list, color = COLORS[model_id-client_num-1], label = candidate_weight[model_id-client_num-1])
        else:
            print(round)
            print(sha_val_list)
            plt.plot(round, sha_val_list, color = COLORS[model_id-client_num-1], label = candidate_weight[model_id-client_num-1])
    plt.legend(fontsize=15)
    plt.xlabel('Round')
    plt.ylabel('Weighted Val_F1')
    plt.grid()
    plt.tight_layout()
    plt.title('bid: {}, seed:{}'.format(bid, seed))
    plt.savefig(os.path.join(sav_dir, '{}_client_bid_{}_seed_{}_sha_val.png'.format(len(bid.split(':')), bid, seed, client_id)))
    plt.close()






if __name__ == '__main__':
    root_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter_sha_10/'
    sav_dir = 'results_sha/lda_splitter_sha_10_plot'

    # root_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_lda_splitter_sha_10/'
    # sav_dir = 'results_sha/cat_splitter_lda_10_plot'
    root_dir = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter_sha_10_3clients_alpha_no_norm'
    sav_dir = 'results_sha/cat_splitter_cat_3clients'



    check_dir(sav_dir)
    # client_bid_2 = [[0, 0], [2, 0], [0, 2], [0, 5], [0, 10], [5, 0],
    #                 [10, 0], [100, 0], [0, 100]]
    # client_bid_2 = [[0, 5]]
    client_bid_3 = [[0, 0, 0], [10, 20, 50], [10, 20, 1000], [10, 20, 100],
                    [50, 20, 10], [1000, 20, 10], [100, 20, 10], [20, 50, 10], [20, 100, 10], [20, 1000, 10]]
    client_bid_3 = [[0,0,0], [0, 0, 10], [0,0,50], [0,0,100], [0,10,0], [0, 50, 0], [0, 100, 0], [10, 0, 0], [50, 0, 0], [100, 0, 0]]

    seed_list = [111, 222, 333, 444, 555]
    # seed_list = [222]
    client_num = 3

    for bid in client_bid_3:
        for seed in seed_list:
            bid_str = ':'.join(str(item) for item in bid)
            file_prefix = '{}_client_{}_seed_{}' \
                             '_tmp_alpha_1_'.format(client_num,
                                                    bid_str,
                                                    seed)
            test_file_name = file_prefix + 'train_valid_metric.csv'
            # test_file_name = file_prefix + 'test_valid_metric.csv'
            alpha_file_name = file_prefix + 'alpha.csv'

            alpha_pd = get_alpha(pth = os.path.join(root_dir, alpha_file_name)
                                 )
            val_dict = get_val(pth = os.path.join(root_dir, test_file_name))
            plot_sha(alpha_pd, val_dict, sav_dir,bid=bid_str, seed=seed)
            # print(val_dict[9].keys())
            # plot_metric(val_dict, sav_dir,bid=bid_str, seed=seed, plot_metric='val_acc')


