import os
import gzip
import pandas as pd
import pickle
import numpy as np


def check_dir(pth):
    if os.path.exists(pth):
        pass
    else:
        os.mkdir(pth)


def split_dir_name(dir_name, prefix):

    tmp = dir_name[len(prefix):]
    print(tmp)
    tmp_list = tmp.split('_')
    print(tmp_list)
    client_bid = [int(item) for item in tmp_list[2].split(':')]
    seed = tmp_list[4]
    if 'tmp' in tmp_list:
        for i in range(len(tmp_list)):
            if tmp_list[i] == 'alpha':

                alpha_tau = float(tmp_list[i+1])
                break
    elif 'entropy' in tmp_list:
        alpha_tau = np.nan
    else:
        print('cannot handle {}'.format(dir_name))
    train_weight = None
    if 'train' in tmp_list and 'weight' in tmp_list:
        for i in range(len(tmp_list)):
            if tmp_list[i] == 'weight':

                train_weight = [float(item) for item in tmp_list[i+1].split(':')]
                break

    return {'client_bid': client_bid, 'seed': seed, 'alpha_tau': alpha_tau, 'train_weight': train_weight}


def collect_alpha_tune_result(file_pth,
                              client_num,
                              plot_sav_dir,
                              file_sav_prefix=None,
                              exit_inf_matrix = False):

    check_dir(plot_sav_dir)
    test_valid_metric = {
        'round': [],
        'client': [],
        'test_acc': [],
        'test_avg_loss': [],
        'val_acc': [],
        'val_avg_loss': []
    }
    train_valid_metric = {
        'round': [],
        'client': [],
        'train_acc': [],
        'train_avg_loss': [],
        'val_avg_loss_before': [],
        'val_avg_loss_after': []
    }

    train_valid_metric_name = [
        'train_acc', 'train_avg_loss', 'val_avg_loss_before',
        'val_avg_loss_after'
    ]
    test_valid_metric_name = [
        'test_acc', 'test_avg_loss', 'val_acc', 'val_avg_loss'
    ]

    if file_pth.split('.')[-1] == 'gz':
        open_func = gzip.open
    else:
        open_func = open

    with open_func(file_pth, 'rb') as f:
        if exit_inf_matrix:
            global_model_idx = 0
        else:
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
                        train_valid_metric[key].append(
                            results['Results_raw'][key])
                if 'test_acc' in results['Results_raw'].keys():
                    test_valid_metric['round'].append(round)
                    test_valid_metric['client'].append(client_id)
                    for key in test_valid_metric_name:
                        test_valid_metric[key].append(
                            results['Results_raw'][key])

    train_valid_metric = pd.DataFrame(train_valid_metric)
    test_valid_metric = pd.DataFrame(test_valid_metric)
    train_valid_metric.to_csv(os.path.join(
        plot_sav_dir, file_sav_prefix + '_' + 'train_valid_metric.csv'),
                              index=False)
    test_valid_metric.to_csv(os.path.join(
        plot_sav_dir, file_sav_prefix + '_' + 'test_valid_metric.csv'),
                             index=False)


def collect_alpha(alpha_pth, num_client, plot_sav_dir, file_sav_prefix=None):
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
    client_id_list = range(1, num_client + 1)
    alpha_df = {'round': []}
    for i in client_id_list:
        alpha_df['client_{}'.format(i)] = []

    for key in alpha.keys():
        alpha_df['round'].append(key)
        for i in client_id_list:
            alpha_df['client_{}'.format(i)].append(alpha[key].flatten()[i - 1])

    alpha_df = pd.DataFrame(alpha_df)
    alpha_df.to_csv(
        os.path.join(plot_sav_dir, file_sav_prefix + '_' + 'alpha.csv'))


def collect_dir_wise(root_dir,
                     prefix_name=None,
                     sav_dir=None,
                     filter_method_list=(),
                     summary=True,
                     has_inf_matrix=False):
    check_dir(sav_dir)
    for dir_name in os.listdir(root_dir):
        if len(dir_name) >= len(
                prefix_name) and dir_name[0:len(prefix_name)] == prefix_name:
            method_name = dir_name.split(prefix_name)[1].split('_')[1]
            if method_name not in filter_method_list:
                info = split_dir_name(dir_name, prefix_name)
                # {'client_bid': client_bid, 'seed': seed, 'alpha_tau': alpha_tau}
                file_pth = os.path.join(root_dir, dir_name)
                key_name = 'results_' + dir_name
                file_sav_prefix = dir_name[len(prefix_name):]

                client_num = len(info['client_bid'])
                print('-' * 10 + ' handling: {} '.format(
                    os.path.join(root_dir, dir_name) + '-' * 10))
                for dirs in os.listdir(file_pth):
                    if dirs[0] == '.':
                        continue

                    file_pth = os.path.join(file_pth, dirs)

                    has_sub_exp = False

                    for file in os.listdir(file_pth):

                        if 'sub' in file.split('_') and 'exp' in file.split('_'):
                            print('Exist: {}'.format(file))
                            has_sub_exp=True
                            pth_add = file
                            break
                    if has_sub_exp:
                        file_pth = os.path.join(file_pth, pth_add)
                    print("working: {}".format(file_pth))

                    for file in os.listdir(file_pth):
                        if file.split('.')[0] == 'eval_results' and (
                                file.split('.')[-1] == 'raw'
                                or file.split('.')[-1] == 'gz'):
                            final_pth = os.path.join(file_pth, file)
                            if summary:
                                plot_sav_dir = sav_dir
                            else:
                                plot_sav_dir = os.path.join(
                                    sav_dir, 'results_' + dir_name)
                            check_dir(plot_sav_dir)
                            collect_alpha_tune_result(final_pth, client_num,
                                                      plot_sav_dir,
                                                      file_sav_prefix,
                                                      exit_inf_matrix=has_inf_matrix)
                            alpha_pth = os.path.join(file_pth, 'alpha.pickle')
                            collect_alpha(alpha_pth, client_num, plot_sav_dir,
                                          file_sav_prefix)





if __name__ == '__main__':
    sav_dir = 'results_rep_train_weight_fix_ss_small_lr_summary'
    root_dir = 'exp_consistent_repeat_small_lr/'
    prefix_name = 'exp_ls_epoch_alpha_tune_'
    has_inf_matrix = True
    collect_dir_wise(root_dir,
                     prefix_name,
                     sav_dir,
                     filter_method_list=(),
                     has_inf_matrix = has_inf_matrix,
                     summary=True)
