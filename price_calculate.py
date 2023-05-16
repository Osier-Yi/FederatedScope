import numpy as np
import pickle

def sum_normalize_func(vec):
    assert len(vec.shape) == 1 or (len(vec.shape) == 2 and
                                   (vec.shape[0] == 1 or vec.shape[1] == 1))
    return vec / sum(vec)

def exp_normalize_func(vec, tau=1):
    print(vec.shape)
    assert len(vec.shape) == 1 or (len(vec.shape) == 2 and
                                   (vec.shape[0] == 1 or vec.shape[1] == 1))
    if len(vec.shape) > 1:
        origin_shape = vec.shape
        vec = vec.flatten()
        tmp = np.exp(vec / tau)
        tmp = sum_normalize_func(tmp)
        return tmp.reshape(origin_shape)
    else:
        tmp = np.exp(vec / tau)
        return sum_normalize_func(tmp)

def price_calculation(influence_matrix_info, client_bid, price_tau, state_idx=398):
    if state_idx not in influence_matrix_info.keys():
        tmp = np.copy(state_idx)
        state_idx = np.max(list(influence_matrix_info.keys()))
        print('state_idx {} is not in influence_matrix_info.keys(), replace as {}'.format(tmp, state_idx))
    tmp_I = np.sum(influence_matrix_info[state_idx], axis=1)
    print(tmp_I)
    perf_contrib = exp_normalize_func(tmp_I, tau=price_tau)
    print(perf_contrib)
    total = np.sum(client_bid)
    print(total)
    payment = np.array(client_bid) - perf_contrib * total
    return payment

def split_dir_name(dir_name, prefix):

    tmp = dir_name[len(prefix):]
    print(tmp)
    tmp_list = tmp.split('_')
    print(tmp_list)
    client_bid = [int(item) for item in tmp_list[2].split(':')]
    seed = tmp_list[4]
    if tmp_list[5] == 'tmp':
        alpha_tau = int(tmp_list[7])
    elif tmp_list[5] == 'entropy':
        alpha_tau = np.nan
    else:
        print('cannot handle {}'.format(dir_name))
    return {'client_bid': client_bid, 'seed': seed, 'alpha_tau': alpha_tau}




client_bid = [1000,20,10]
info_matrix_pth = 'exp_consistent/exp_ls_epoch_alpha_tune_3_clients_1000_20_10_consistent_label_entropy_tau_alpha/FedAvg_convnet2_on_CIFAR10@torchvision_lr0.5_lstep2/inf_matrix.pickle'
f = open(info_matrix_pth, 'rb')
info_matrx = pickle.load(f)
print(info_matrx[398])
price_tau = 1
state_idx = 498
payment = price_calculation(info_matrx, client_bid,price_tau, state_idx)
# print(payment)
# dir_name = 'exp_ls_epoch_alpha_tune_3_client_10:20:50_seed_111_tmp_alpha_1'
# print(split_dir_name(dir_name, prefix='exp_ls_epoch_alpha_tune_'))


