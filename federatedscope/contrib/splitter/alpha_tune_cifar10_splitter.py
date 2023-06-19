# cifar10_label_type_based_slice

import numpy as np
from federatedscope.core.splitters import BaseSplitter
from federatedscope.core.splitters.utils import _split_according_to_prior
from federatedscope.register import register_splitter


def cifar10_label_type_based_slice(label,
                                        client_num,
                                        alpha,
                                        min_size=1,
                                        prior=None,
                                   label_class_category = {0:[0,1,8,9], 1:[2,3,4,5,6,7]}):
    # if client_num != 2:
    #     raise ValueError('Only support two clients!')
    assert client_num == len(label_class_category.keys())

    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # print('work on class: {}'.format(k))
            # for label k
            class_category = None

            for item in label_class_category.keys():
                if k in label_class_category[item]:
                    class_category = item
            if client_num is None:
                raise ValueError("label lass not in label_class_category")


            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = []
            idx_prev = 0
            for client_id in range(client_num):
                if client_id == client_num-1:
                    break
                if client_id != class_category:
                    # print('idx_prev: {}'.format(idx_prev))
                    prop.append(idx_prev + int(len(idx_k)*alpha/(client_num-1)))
                    # print('client_id: {}, len(idx_k):{}, '
                    #       'alpha: {}, '
                    #       'client_num: {}, '
                    #       'int(len(idx_k)*alpha/(client_num-1)): {}'.format(client_id, len(idx_k), alpha, client_num, int(len(idx_k)*alpha/(client_num-1))))
                    idx_prev = idx_prev + int(len(idx_k)*alpha/(client_num-1))
                else:
                    prop.append(idx_prev + int(len(idx_k) * (1-alpha)))
                    idx_prev += int(len(idx_k) * (1-alpha))

            print(prop)




            #
            # if class_category == 0:
            #     prop = [int(len(idx_k)*(1-alpha))]
            # else:
            #     prop = [int(len(idx_k) * (alpha))]

            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
            # print('size:{}'.format(size))
    print(len(idx_slice))
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice



class AlphaTuneCifar10Splitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, label_class_category, alpha=0.5):
        self.alpha = alpha
        self.label_class_category = label_class_category
        super(AlphaTuneCifar10Splitter, self).__init__(client_num)

    def __call__(self, dataset, prior=None, **kwargs):
        from torch.utils.data import Dataset, Subset

        tmp_dataset = [ds for ds in dataset]
        label = np.array([y for x, y in tmp_dataset])
        # np.random.seed(seed=split_seed )
        idx_slice = cifar10_label_type_based_slice(label,
                                                   self.client_num,
                                                   self.alpha,
                                                   prior=prior,
                                                   label_class_category=self.label_class_category)
        if isinstance(dataset, Dataset):
            data_list = [Subset(dataset, idxs) for idxs in idx_slice]
        else:
            data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
        # np.random.seed(seed=train_seed )
        return data_list


def call_alpha_tune_cifar10_splitter(splitter_type, client_num, **kwargs):
    if splitter_type == 'alpha_tune_cifar10_splitter':
        splitter = AlphaTuneCifar10Splitter(client_num, **kwargs)
        return splitter


register_splitter('alpha_tune_cifar10_splitter', call_alpha_tune_cifar10_splitter)
