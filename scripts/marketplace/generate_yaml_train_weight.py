import yaml
import os

from results_post.results_collection import check_dir
def replace_yaml(original_yaml_file_3, yaml_root):
    check_dir(yaml_root)
    with open(original_yaml_file_3, 'r') as f:
        original_yaml = yaml.safe_load(f)

        for client_bid in client_bid_3:
            for seed_val in seed:
                for train_weight_val in train_weight_3:
                    original_yaml['train']['optimizer']['lr'] = train_lr
                    original_yaml['hpo']['fedex'][
                        'ss'] = ss
                    original_yaml['marketplace']['alpha_tune'][
                        'train_weight_control'] = True
                    original_yaml['marketplace']['alpha_tune'][
                        'train_weight'] = train_weight_val
                    original_yaml['marketplace']['alpha_tune'][
                        'info_matrix_pth'] = inf_matrix_pth_prefix.format(3, seed_val)
                    original_yaml['seed'] = seed_val
                    original_yaml['marketplace']['alpha_tune'][
                        'client_bid'] = client_bid
                    original_yaml['federate']['total_round_num'] = total_round
                    for entropy_determined_val in entropy_determined:
                        original_yaml['marketplace']['alpha_tune'][
                            'entropy_determined'] = entropy_determined_val
                        client_bid_text = ':'.join('{}'.format(item)
                                                   for item in client_bid)
                        train_weight_txt = ':'.join('{}'.format(item)
                                                    for item in train_weight_val)
                        if entropy_determined_val:
                            file_post = '{}_client_{}_seed_{}_train_weight_{}_entropy'.format(
                                len(client_bid), client_bid_text, seed_val, train_weight_txt)
                            original_yaml['outdir'] = os.path.join(
                                outdir_root, our_dir_name_prefix + file_post)
                            yaml_file_name = yaml_file_name_prefix + file_post + '.yaml'
                            yaml_sav_pth = os.path.join(
                                yaml_root, yaml_file_name)
                            yaml_file_pth_list_three_clients.append(
                                yaml_sav_pth)
                            optional_run_set.append(yaml_sav_pth)
                            with open(yaml_sav_pth, 'w') as yaml_f:
                                yaml.dump(original_yaml, yaml_f)
                        else:
                            for tau_alpha_val in tau_alpha:
                                file_post = '{}_client_{}_seed_{}_train_weight_{}_tmp_alpha_{}'.format(
                                    len(client_bid), client_bid_text, seed_val, train_weight_txt,
                                    tau_alpha_val)
                                original_yaml['marketplace']['alpha_tune'][
                                    'tau_alpha'] = tau_alpha_val
                                original_yaml['outdir'] = os.path.join(
                                    outdir_root,
                                    our_dir_name_prefix + file_post)
                                yaml_file_name = yaml_file_name_prefix + file_post + '.yaml'
                                yaml_sav_pth = os.path.join(
                                    yaml_root, yaml_file_name)
                                yaml_file_pth_list_three_clients.append(
                                    yaml_sav_pth)
                                if tau_alpha_val == 1:
                                    top_run_set.append(yaml_sav_pth)
                                else:
                                    optional_run_set.append(yaml_sav_pth)
                                with open(yaml_sav_pth, 'w') as yaml_f:
                                    yaml.dump(original_yaml, yaml_f)


if __name__ == '__main__':
    original_yaml_file_2 = 'scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cifar10_2_clients_0_10.yaml'

    # original_yaml = yaml.safe_load(original_yaml_file)
    # 'device': 0  maketplace.alpha_tune.client_bid= [0,10] seed= [22,33,44,55,66]
    # outdir: 'exp_consistent_repeat/exp_ls_epoch_alpha_tune_2_clients_1000_0_entropy_tau_alpha_consistent_label'
    #
    # ourdir_root = exp_consistent_repeat_tmp
    #
    # cfg.marketplace.alpha_tune.tau_alpha = 1
    #
    # marketplace:
    #     alpha_tune_use: True
    #     alpha_tune:
    #         client_bid: [1000, 0]
    #         entropy_determined: True

    total_round = 1000

    seed = [111, 222, 333, 444, 555]

    # client_bid_2 = [[0, 0], [0, 5], [0, 10], [0, 100], [0, 1000], [5, 0],
    #                 [10, 0], [100, 0], [1000, 0]]
    # client_bid_3 = [[0, 0, 0], [10, 20, 50], [10, 20, 1000], [10, 20, 100],
    #                 [50, 20, 10], [1000, 20, 10], [100, 20, 10]]
    # client_bid_2 = [[0, 0], [0, 5], [5, 0]]
    client_bid_2 = [[0, 0]]
    # client_bid_3 = [[0, 0, 0], [10, 20, 50], [10, 20, 1000], [10, 20, 100],
    #                 [50, 20, 10], [1000, 20, 10], [100, 20, 10]]
    client_bid_3 = [[10, 20, 50]]
    # entropy_determined = [True, False]
    entropy_determined = [False]
    train_weight_2 = [[0.9,0.1], [0.8,0.2], [0.3,0.7], [0.6,0.4], [0.4, 0.6], [0.5,0.5], [0.2, 0.8],[0.7,0.3], [0.1,0.9]]
    # train_weight_2 = [[0.9,0.1]]
    # train_weight_3 = [[0.7, 0.2, 0.1]]

    train_weight_3 = [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.2,0.1,0.7], [0.3,0.3,0.4]]

    # train_lr = 0.005
    # batch_or_epoch = 'batch'
    # local_update_steps = 30


    tau_alpha = [1]
    outdir_root = 'exp_consistent_repeat_no_search_epoch_sche_multistep_lr_label_splitter_more_metric'
    our_dir_name_prefix = 'exp_ls_epoch_alpha_tune_'
    yaml_file_name_prefix = 'alpha_tune_fedex_for_cifar10_'
    ss = 'scripts/marketplace/example_scripts/cifar10/avg/fedex_grid_search_space_no_search_epoch_sche_ver.yaml'
    yaml_root_dir = 'scripts/marketplace/example_scripts/ls_run_scripts_repeated_no_search_epoch_sche_multistep_lr_label_splitter_more_metric'
    sh_pth = 'scripts/marketplace/rep_sh_epoch_sche_multistep_lr_label_splitter_more_metric'
    change_splitter = True
    splitter_args_alpha = 0.01
    splitter_name = 'alpha_tune_cifar10_splitter'
    add_eval_metric = True
    metric_list = ['acc', 'correct',  'f1', 'roc_auc']


    check_dir(yaml_root_dir)
    yaml_root = os.path.join(yaml_root_dir, 'two_clients')
    check_dir(yaml_root)

    inf_matrix_pth_prefix = 'info_matrix/{}_clients_{}_inf_matrix.pickle'
    # inf_matrix_pth_prefix = ''
    use_batch = False
    batch_size = 64
    use_sche = True
    sche_type = 'multisteplr'
    # use_update_hp = True
    update_hp = False

    local_update_steps = 1
    lr = 0.1





    top_run_set = []
    optional_run_set = []

    yaml_file_pth_list_two_clients = []
    yaml_file_pth_list_three_clients = []


    original_yaml_file_2 = 'scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cifar10_2_clients_0_10.yaml'
    with open(original_yaml_file_2, 'r') as f:
        original_yaml = yaml.safe_load(f)

        for client_bid in client_bid_2:
            for seed_val in seed:
                for train_weight_val in train_weight_2:
                    # original_yaml['train']['optimizer']['batch_or_epoch'] = batch_or_epoch
                    # original_yaml['train']['optimizer']['local_update_steps'] = local_update_steps
                    # original_yaml['train']['optimizer']['lr'] = train_lr
                    original_yaml['dataloader']['batch_size'] = batch_size
                    if change_splitter:
                        if 'splitter' not in original_yaml['data']:
                            original_yaml['data']['splitter'] = dict()
                        original_yaml['data']['splitter'] = splitter_name
                        original_yaml['data']['splitter_args']=[{'alpha':splitter_args_alpha}]
                    if add_eval_metric:
                        if 'metrics' not in original_yaml['eval']:
                            original_yaml['eval']['metrics'] = dict()
                        original_yaml['eval']['metrics'] = metric_list
                    if use_batch:
                        original_yaml['train']['batch_or_epoch'] = 'batch'
                    if use_sche:
                        # print(original_yaml['train'].keys())
                        if 'scheduler' not in original_yaml['train'].keys():
                            original_yaml['train']['scheduler'] = dict()
                        original_yaml['train']['scheduler']['type'] = sche_type
                    if update_hp == False:
                        if 'update_hp' not in original_yaml['marketplace']['alpha_tune'].keys():
                            original_yaml['marketplace']['alpha_tune']['update_hp'] = update_hp
                        original_yaml['train']['local_update_steps']=local_update_steps
                        original_yaml['train']['optimizer']['lr'] = lr
                    original_yaml['hpo']['fedex'][
                        'ss'] = ss
                    original_yaml['marketplace']['alpha_tune'][
                        'train_weight_control'] = True
                    original_yaml['marketplace']['alpha_tune'][
                        'train_weight'] = train_weight_val
                    original_yaml['marketplace']['alpha_tune'][
                        'info_matrix_pth'] = inf_matrix_pth_prefix.format(2, seed_val)
                    original_yaml['marketplace']['alpha_tune'][
                        'client_bid'] = client_bid
                    original_yaml['seed'] = seed_val
                    original_yaml['federate']['total_round_num'] = total_round
                    for entropy_determined_val in entropy_determined:
                        original_yaml['marketplace']['alpha_tune'][
                            'entropy_determined'] = entropy_determined_val
                        client_bid_text = ':'.join('{}'.format(item)
                                                   for item in client_bid)
                        train_weight_txt = ':'.join('{}'.format(item)
                                                   for item in train_weight_val)
                        if entropy_determined_val:
                            file_post = '{}_client_{}_seed_{}_train_weight_{}_entropy'.format(
                                len(client_bid), client_bid_text, seed_val, train_weight_txt)
                            original_yaml['outdir'] = os.path.join(
                                outdir_root, our_dir_name_prefix + file_post)
                            yaml_file_name = yaml_file_name_prefix + file_post + '.yaml'
                            yaml_sav_pth = os.path.join(yaml_root, yaml_file_name)
                            yaml_file_pth_list_two_clients.append(yaml_sav_pth)
                            optional_run_set.append(yaml_sav_pth)
                            with open(yaml_sav_pth, 'w') as yaml_f:
                                yaml.dump(original_yaml, yaml_f)
                        else:
                            for tau_alpha_val in tau_alpha:
                                file_post = '{}_client_{}_seed_{}_train_weight_{}_tmp_alpha_{}'.format(
                                    len(client_bid), client_bid_text, seed_val,train_weight_txt,
                                    tau_alpha_val)

                                original_yaml['marketplace']['alpha_tune'][
                                    'tau_alpha'] = tau_alpha_val
                                original_yaml['outdir'] = os.path.join(
                                    outdir_root, our_dir_name_prefix + file_post)
                                yaml_file_name = yaml_file_name_prefix + file_post + '.yaml'
                                yaml_sav_pth = os.path.join(
                                    yaml_root, yaml_file_name)
                                yaml_file_pth_list_two_clients.append(yaml_sav_pth)
                                if tau_alpha_val == 1:
                                    top_run_set.append(yaml_sav_pth)
                                else:
                                    optional_run_set.append(yaml_sav_pth)
                                with open(yaml_sav_pth, 'w') as yaml_f:
                                    yaml.dump(original_yaml, yaml_f)


        original_yaml_file_3 = 'scripts/marketplace/example_scripts/ls_run_scripts/alpha_tune_fedex_for_cnn_cifar10_3_clients_50_20_10.yaml'
        # yaml_root = 'scripts/marketplace/example_scripts/ls_run_scripts_repeated_exp_small_lr/three_clients'
        yaml_root = os.path.join(yaml_root_dir, 'three_clients')
        check_dir(yaml_root)
        with open(original_yaml_file_3, 'r') as f:
            original_yaml = yaml.safe_load(f)

            for client_bid in client_bid_3:
                for seed_val in seed:
                    for train_weight_val in train_weight_3:
                        original_yaml['dataloader']['batch_size'] = batch_size

                        if change_splitter:
                            if 'splitter' not in original_yaml['data']:
                                original_yaml['data']['splitter'] = dict()
                            original_yaml['data']['splitter'] = splitter_name
                            original_yaml['data']['splitter_args']=[{'alpha':splitter_args_alpha}]
                        if add_eval_metric:
                            if 'metrics' not in original_yaml['eval']:
                                original_yaml['eval']['metrics'] = dict()
                            original_yaml['eval']['metrics'] = metric_list

                        if use_batch:
                            original_yaml['train']['batch_or_epoch'] = 'batch'
                        if use_sche:
                            # print(original_yaml['train'].keys())
                            if 'scheduler' not in original_yaml['train'].keys():
                                original_yaml['train']['scheduler'] = dict()
                            original_yaml['train']['scheduler']['type'] = sche_type
                        if update_hp == False:
                            if 'update_hp' not in original_yaml['marketplace']['alpha_tune'].keys():
                                original_yaml['marketplace']['alpha_tune']['update_hp'] = update_hp
                        # original_yaml['train']['optimizer']['lr'] = train_lr
                        original_yaml['hpo']['fedex'][
                            'ss'] = ss
                        original_yaml['marketplace']['alpha_tune'][
                            'train_weight_control'] = True
                        original_yaml['marketplace']['alpha_tune'][
                            'train_weight'] = train_weight_val
                        original_yaml['marketplace']['alpha_tune'][
                            'info_matrix_pth'] = inf_matrix_pth_prefix.format(3, seed_val)
                        original_yaml['seed'] = seed_val
                        original_yaml['marketplace']['alpha_tune'][
                            'client_bid'] = client_bid
                        original_yaml['federate']['total_round_num'] = total_round
                        for entropy_determined_val in entropy_determined:
                            original_yaml['marketplace']['alpha_tune'][
                                'entropy_determined'] = entropy_determined_val
                            client_bid_text = ':'.join('{}'.format(item)
                                                       for item in client_bid)
                            train_weight_txt = ':'.join('{}'.format(item)
                                                        for item in train_weight_val)
                            if entropy_determined_val:
                                file_post = '{}_client_{}_seed_{}_train_weight_{}_entropy'.format(
                                    len(client_bid), client_bid_text, seed_val, train_weight_txt)
                                original_yaml['outdir'] = os.path.join(
                                    outdir_root, our_dir_name_prefix + file_post)
                                yaml_file_name = yaml_file_name_prefix + file_post + '.yaml'
                                yaml_sav_pth = os.path.join(
                                    yaml_root, yaml_file_name)
                                yaml_file_pth_list_three_clients.append(
                                    yaml_sav_pth)
                                optional_run_set.append(yaml_sav_pth)
                                with open(yaml_sav_pth, 'w') as yaml_f:
                                    yaml.dump(original_yaml, yaml_f)
                            else:
                                for tau_alpha_val in tau_alpha:
                                    file_post = '{}_client_{}_seed_{}_train_weight_{}_tmp_alpha_{}'.format(
                                        len(client_bid), client_bid_text, seed_val, train_weight_txt,
                                        tau_alpha_val)
                                    original_yaml['marketplace']['alpha_tune'][
                                        'tau_alpha'] = tau_alpha_val
                                    original_yaml['outdir'] = os.path.join(
                                        outdir_root,
                                        our_dir_name_prefix + file_post)
                                    yaml_file_name = yaml_file_name_prefix + file_post + '.yaml'
                                    yaml_sav_pth = os.path.join(
                                        yaml_root, yaml_file_name)
                                    yaml_file_pth_list_three_clients.append(
                                        yaml_sav_pth)
                                    if tau_alpha_val == 1:
                                        top_run_set.append(yaml_sav_pth)
                                    else:
                                        optional_run_set.append(yaml_sav_pth)
                                    with open(yaml_sav_pth, 'w') as yaml_f:
                                        yaml.dump(original_yaml, yaml_f)


    available_gpu = [0, 1, 2, 3, 4, 5, 6, 7]
    per_gpu_running = 6
    print(len(yaml_file_pth_list_three_clients))
    print(len(yaml_file_pth_list_two_clients))
    print(len(top_run_set))
    print(len(optional_run_set))

    top_run_set = yaml_file_pth_list_two_clients
    optional_run_set = []

    total_run = (len(yaml_file_pth_list_three_clients) +
                 len(yaml_file_pth_list_two_clients)) / (len(available_gpu) *
                                                         per_gpu_running)
    running_temp = 'CUDA_VISIBLE_DEVICES={} python federatedscope/main.py --cfg {} &\n'
    bashfile_id = 0
    count = 0
    sh_text = ''
    new_sh = True
    limit = per_gpu_running * len(available_gpu)

    check_dir(sh_pth)

    while len(optional_run_set) > 0 or len(top_run_set) > 0:

        for gpu_id in available_gpu:
            for i in range(per_gpu_running):
                if len(top_run_set) > 0:
                    current = top_run_set.pop(0)
                else:
                    if len(optional_run_set) == 0:
                        break
                    current = optional_run_set.pop(0)

                sh_text += running_temp.format(gpu_id, current)
                count += 1
                if count == limit or (len(top_run_set) == 0 and len(optional_run_set) == 0):
                    bash_pth = os.path.join(sh_pth,
                                            'run_{}.sh'.format(bashfile_id))
                    with open(bash_pth, 'w') as f:
                        f.write(sh_text)
                    bashfile_id += 1
                    sh_text = ''
                    count = 0

        # for client_bid in client_bid_2:
        #     for seed_val in seed:
        #         for entropy_determined_val in entropy_determined:
        #             if entropy_determined_val:
        #                 original_yaml['marketplace']['alpha_tune']['entropy_determined'] = entropy_determined_val
        #             e

        # print(original_yaml)
