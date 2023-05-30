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


def get_alpha(pth = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter/2_client_0:0_seed_111_tmp_alpha_1_alpha.csv'):
    alpha_pd = pd.read_csv(pth)
    return alpha_pd

def get_val(pth = 'results_sha/results_exp_v1_order_bug_agg_weight_sha_cat_splitter/2_client_0:0_seed_111_tmp_alpha_1_test_valid_metric.csv'):
    val_pd = pd.read_csv(pth)
    model_id_set = pd.unique(val_pd['model'])
    val_dict = dict()
    for model_id in model_id_set:
        val_dict[model_id] = val_pd[val_pd['model']==model_id]
    return val_dict


def plot_sha(alpha_pd,val_dict,sav_dir='results_sha/cat_spliter_sha_plot', bid='0:0', seed = 111):
    client_id_set = pd.unique(val_dict[list(val_dict.keys())[0]]['client'])
    for client_id in client_id_set:
        for model_id in val_dict.keys():
            tmp = val_dict[model_id]
            tmp_client = tmp[tmp['client']==client_id]
            if tmp_client.shape[0] ==1:
                plt.scatter(tmp_client['round'], tmp_client['val_f1'],label = model_id )
            else:
                plt.plot(tmp_client['round'], tmp_client['val_f1'], label = model_id)
        plt.legend(ncol=2)
        # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #            mode="expand", borderaxespad=0, ncol=5)
        # plt.tight_layout()
        plt.title('bid: {}, seed:{}, clientID: {}'.format(bid, seed, client_id))
        plt.savefig(os.path.join(sav_dir, '{}_client_bid_{}_seed_{}_clientID_{}.png'.format(len(bid.split(':')), bid, seed, client_id)))
        plt.close()

    for model_id in val_dict.keys():
        round = []
        for client_id in client_id_set:
            tmp = val_dict[model_id]
            tmp_client = tmp[tmp['client'] == client_id]





if __name__ == '__main__':
    alpha_pd = get_val()
    val_dict = get_val()
    plot_sha(alpha_pd, val_dict)