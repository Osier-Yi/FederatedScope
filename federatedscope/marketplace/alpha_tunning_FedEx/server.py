from federatedscope.autotune.fedex.server import FedExServer, discounted_mean
import pickle
import numpy as np
import os
import logging
from federatedscope.core.message import Message
from federatedscope.core.auxiliaries.utils import merge_dict_of_results
from federatedscope.autotune.fedex.utils import HyperNet
from numpy.linalg import norm
from scipy.special import logsumexp
import torch

import yaml

logger = logging.getLogger(__name__)
from itertools import product

from scipy.stats import entropy


def is_finish_current_state(check_model_state):
    all_model_finish = False
    for item in check_model_state:
        if item == False:
            return all_model_finish
    else:
        return True


def sum_normalize_func(vec):
    assert len(vec.shape) == 1 or (len(vec.shape) == 2 and
                                   (vec.shape[0] == 1 or vec.shape[1] == 1))
    return vec / sum(vec)


def sum_normalize_func_ep(vec, epsilon=1):
    assert len(vec.shape) == 1 or (len(vec.shape) == 2 and
                                   (vec.shape[0] == 1 or vec.shape[1] == 1))
    if len(vec.shape) > 1:
        origin_shape = vec.shape
        vec = vec.flatten()
        tmp = (vec + epsilon) / sum(vec + epsilon)
        return tmp.reshape(origin_shape)
    else:
        return (vec + epsilon) / sum(vec + epsilon)


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


class AlphaFedExServer(FedExServer):
    '''
    Compared with FedExServer, we add multiple model, alpha calculation
    '''
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(AlphaFedExServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)

        self.marketplace_client_bid = config.marketplace.alpha_tune.client_bid

        if len(config.marketplace.alpha_tune.info_matrix_pth) > 0:
            self.has_info_matrix = True
            import pickle
            f = open(config.marketplace.alpha_tune.info_matrix_pth, 'rb')
            self.info_matrix_all = pickle.load(f)
        else:
            self.has_info_matrix = False

        logger.info("client bid: {}".format(self.marketplace_client_bid))
        self.tau_v = config.marketplace.alpha_tune.tau_v
        self.task_value = sum_normalize_func_ep(
            np.array(self.marketplace_client_bid).reshape(1, client_num),
            self.tau_v)
        logger.info('The task value of each clients is: {}'.format(
            self.task_value))
        if config.marketplace.alpha_tune.entropy_determined:
            self.tau_alpha = 1.0 / entropy(np.ones(self.client_num)) \
                             * entropy(self.task_value.flatten())
        else:
            self.tau_alpha = config.marketplace.alpha_tune.tau_alpha
        logger.info("the tau_alpha: {}".format(self.tau_alpha))

        self.tau_p = config.marketplace.alpha_tune.tau_p

        self.current_model_idx = 0

        if self.has_info_matrix:
            self.model_num = 1
            self.models = [self.models[0]]
        else:

            self.model_num = self.client_num + 2

        if config.marketplace.alpha_tune.train_weight_control:
            from federatedscope.marketplace.alpha_tunning_FedEx import AlphaTuneAggretor

            logger.info('Replace the last aggregator to AlphaTuneAggretor')

            self.aggregators[self.model_num - 1] = AlphaTuneAggretor(
                model=model, device=device, config=config)

        # self.check_model_updates = [False for i in range(client_num)]

        # global model with alpha tune; global model without alpha tune;
        # plus model without client i

        self.val_info = dict()

        self.influence_matrix_info = dict()
        self.alpha_info = dict()

        # self.modelset = dict()
        assert len(
            self.models
        ) == self.model_num, 'the model number must be equal to the client num'
        # fedex_servers = [FedExServer()]

        with open(config.hpo.fedex.ss, 'r') as ips:
            ss = yaml.load(ips, Loader=yaml.FullLoader)

        if next(iter(ss.keys())).startswith('arm'):
            # This is a flattened action space
            # ensure the order is unchanged
            ss = sorted([(int(k[3:]), v) for k, v in ss.items()],
                        key=lambda x: x[0])
            self._grid = []
            self._cfsp = [[tp[1] for tp in ss]]
        else:
            # This is not a flat search space
            # be careful for the order
            self._grid = sorted(ss.keys())
            self._cfsp = [ss[pn] for pn in self._grid]

        sizes = [len(cand_set) for cand_set in self._cfsp]
        eta0 = 'auto' if config.hpo.fedex.eta0 <= .0 else float(
            config.hpo.fedex.eta0)
        self._eta0 = [
            np.sqrt(2.0 * np.log(size)) if eta0 == 'auto' else eta0
            for size in sizes
        ]
        self._sched = config.hpo.fedex.sched
        self._cutoff = config.hpo.fedex.cutoff
        self._baseline = config.hpo.fedex.gamma
        self._diff = config.hpo.fedex.diff
        if self._cfg.hpo.fedex.psn:
            # personalized policy
            # in this version no psn
            # TODO: client-wise RFF
            self._client_encodings = torch.randn(
                (client_num, 8), device=device) / np.sqrt(8)
            self._policy_net = HyperNet(
                self._client_encodings.shape[-1],
                sizes,
                client_num,
                device,
            ).to(device)
            self._policy_net.eval()
            theta4stat = [
                theta.detach().cpu().numpy()
                for theta in self._policy_net(self._client_encodings)
            ]
            self._pn_optimizer = torch.optim.Adam(
                self._policy_net.parameters(),
                lr=self._cfg.hpo.fedex.pi_lr,
                weight_decay=1e-5)
        else:
            self._z = [[np.full(size, -np.log(size)) for size in sizes]
                       for i in range(self.model_num)]
            self._theta = [[np.exp(z).flatten() for z in self._z[i]]
                           for i in range(self.model_num)]
            theta4stat = [self._theta[i] for i in range(self.model_num)]
            self._store = [[0.0 for _ in sizes] for i in range(self.model_num)]
        self._stop_exploration = [False for i in range(self.model_num)]
        self._trace = [{
            'global': [],
            'refine': [],
            'entropy': [self.entropy(theta4stat[i])],
            'mle': [self.mle(theta4stat[i])]
        } for i in range(self.model_num)]

    def update_val_info(self, info):
        if self.state not in self.val_info.keys():
            self.val_info[self.state] = dict()
        if self.current_model_idx not in self.val_info[self.state].keys():
            self.val_info[self.state][self.current_model_idx] = dict()
        self.val_info[self.state][self.current_model_idx][
            info['client_id']] = info

    def update_influence_matrix(self, metric_name='val_avg_loss_before'):

        if self.has_info_matrix:
            try:
                inf_matrix = self.info_matrix_all[self.state]
                self.last_round_inf_matrix = inf_matrix
            except:
                logger.info(
                    "existing inf matrix not has inf matrix of round: {}, replaced with last round!"
                    .format(self.state))
                inf_matrix = self.last_round_inf_matrix
            logger.info(
                "Round: {}, Model:{}, influence matrix (Exist): {}".format(
                    self.state, self.current_model_idx, inf_matrix))
            self.influence_matrix_info[self.state] = inf_matrix
            return inf_matrix
        else:
            inf_matrix = np.zeros([self.client_num, self.client_num])

            # here client start from 0
            global_id = self.model_num - 2
            for row_id in range(self.client_num):
                for col_id in range(self.client_num):
                    inf_matrix[row_id, col_id] = \
                        self.val_info[self.state][row_id][col_id][metric_name] - \
                        self.val_info[self.state][global_id][col_id][metric_name]
            self.influence_matrix_info[self.state] = inf_matrix
            logger.info("Round: {}, Model:{}, influence matrix: {}".format(
                self.state, self.current_model_idx, inf_matrix))
            return inf_matrix

    def update_alpha(self):
        self.update_influence_matrix()
        tmp_I = np.sum(self.influence_matrix_info[self.state], axis=1)
        alpha = exp_normalize_func(np.multiply(tmp_I, self.task_value),
                                   tau=self.tau_alpha)
        self.alpha_info[self.state] = alpha

    def save_alpha_related_info(self):
        with open(os.path.join(self._cfg.outdir, "alpha.pickle"),
                  "wb") as outfile:
            pickle.dump(self.alpha_info,
                        outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self._cfg.outdir, "val_info.pickle"),
                  "wb") as outfile:
            pickle.dump(self.val_info,
                        outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)
        with open(
                os.path.join(self._cfg.outdir,
                             "3_clients_111_inf_matrix.pickle"),
                "wb") as outfile:
            pickle.dump(self.influence_matrix_info,
                        outfile,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def trace(self, key, model_id):
        '''returns trace of one of three tracked quantities
        Args:
            key (str): 'entropy', 'global', or 'refine'
        Returns:
            numpy vector with length equal to number of rounds up to now.
        '''

        return np.array(self._trace[model_id][key])

    def sample(self, thetas, model_id):
        """samples from configs using current probability vector
        Arguments:
          thetas (list): probabilities for the hyperparameters.
        """

        # determine index
        if self._stop_exploration[model_id]:
            cfg_idx = [theta.argmax() for theta in thetas]
        else:
            cfg_idx = [
                np.random.choice(len(theta), p=theta) for theta in thetas
            ]

        # get the sampled value(s)
        if self._grid:
            sampled_cfg = {
                pn: cands[i]
                for pn, cands, i in zip(self._grid, self._cfsp, cfg_idx)
            }
        else:
            sampled_cfg = self._cfsp[0][cfg_idx[0]]

        return cfg_idx, sampled_cfg

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True):
        """
        To broadcast the message to all clients or sampled clients
        """

        # self.unseen_clients_id = 1

        # if self.current_model_idx + 1 >= self.model_num:
        #     model_id =0
        #     self.check_model_updates = [False for i in range(self.model_num)]
        #     self.current_model_idx = 0
        #     self.check_model_updates[model_id] = True
        # else:
        #     model_id = self.current_model_idx + 1
        #     self.check_model_updates[model_id] = True
        #     self.current_model_idx += 1

        model_id = self.current_model_idx

        logger.info(" broadcast model {}".format(model_id))
        if filter_unseen_clients:
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:
            receiver = self.sampler.sample(size=sample_client_num)
        else:
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._noise_injector is not None and msg_type == 'model_para':
            # Inject noise only when broadcast parameters
            for model_idx_i in range(len(self.models)):
                num_sample_clients = [
                    v["num_sample"] for v in self.join_in_info.values()
                ]
                self._noise_injector(self._cfg, num_sample_clients,
                                     self.models[model_idx_i])

        # if self.model_num > 1:
        #     model_para = [model.state_dict() for model in self.models]
        # else:
        #     model_para = self.model.state_dict()
        model_para = self.models[model_id].state_dict()

        # sample the hyper-parameter config specific to the clients
        if self._cfg.hpo.fedex.psn:
            self._policy_net.train()
            self._pn_optimizer.zero_grad()
            self._theta = self._policy_net(self._client_encodings)
        for rcv_idx in receiver:
            if self._cfg.hpo.fedex.psn:
                cfg_idx, sampled_cfg = self.sample([
                    theta[model_id][rcv_idx - 1].detach().cpu().numpy()
                    for theta in self._theta[self.unseen_clients_id]
                ],
                                                   model_id=model_id)
            else:
                cfg_idx, sampled_cfg = self.sample(self._theta[model_id],
                                                   model_id=model_id)
            content = {
                'model_param': model_para,
                "arms": cfg_idx,
                'hyperparam': sampled_cfg,
                'model_id': model_id
            }
            self.comm_manager.send(
                Message(msg_type=msg_type,
                        sender=self.ID,
                        receiver=[rcv_idx],
                        state=self.state,
                        content=content))
        if self._cfg.federate.online_aggr:
            for idx in range(self.model_num):
                self.aggregators[idx].reset()

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def callback_funcs_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        # For a new round
        if round not in self.msg_buffer['train'].keys():
            self.msg_buffer['train'][round] = dict()

        self.msg_buffer['train'][round][sender] = content

        # check whetehr the received model parameter is the current model id
        logger.info('handling model id: {}'.format(content[3]))
        assert content[3] == self.current_model_idx

        if self._cfg.federate.online_aggr:
            self.aggregator.inc(tuple(content[0:2]))

        return self.check_and_move_on()

    def update_policy(self, feedbacks, model_idx):
        """Update the policy. This implementation is borrowed from the
        open-sourced FedEx (
        https://github.com/mkhodak/FedEx/blob/ \
        150fac03857a3239429734d59d319da71191872e/hyper.py#L151)
        Arguments:
            feedbacks (list): each element is a dict containing "arms" and
            necessary feedback.
        """

        index = [elem['arms'] for elem in feedbacks]
        cids = [elem['client_id'] for elem in feedbacks]
        before = np.asarray(
            [elem['val_avg_loss_before'] for elem in feedbacks])
        after = np.asarray([elem['val_avg_loss_after'] for elem in feedbacks])

        if self.state != 0 and model_idx == self.model_num - 1:
            self.update_alpha()
            weight = self.alpha_info[self.state].flatten()
            logger.info('Round: #{}, alpha: {}'.format(self.state, weight))
        else:
            weight = np.asarray([elem['val_total'] for elem in feedbacks],
                                dtype=np.float64)
            weight /= np.sum(weight)
            # logger.info('weight size: {}'.format(weight.shape))
        #
        #
        # if model_idx != self.model_num-1:
        #     weight = np.asarray([elem['val_total'] for elem in feedbacks],
        #                         dtype=np.float64)
        #     weight /= np.sum(weight)
        # else:
        #     self.update_alpha()
        #     weight = self.alpha_info[self.state]
        # weight = np.asarray([elem['val_total'] for elem in feedbacks],
        #                     dtype=np.float64)
        # weight /= np.sum(weight)
        logger.info("Round: {}, Model:{}, weight: {}".format(
            self.state, self.current_model_idx, weight))

        logger.info('self._trace is: {}'.format(self._trace))

        if self._trace[model_idx]['refine']:
            trace = self.trace('refine', model_id=model_idx)
            if self._diff:
                trace -= self.trace('global', model_id=model_idx)
            baseline = discounted_mean(trace, self._baseline)
        else:
            baseline = 0.0
        self._trace[model_idx]['global'].append(np.inner(before, weight))
        self._trace[model_idx]['refine'].append(np.inner(after, weight))
        if self._stop_exploration[model_idx]:
            self._trace[model_idx]['entropy'].append(0.0)
            self._trace[model_idx]['mle'].append(1.0)
            return

        if self._cfg.hpo.fedex.psn:
            pass
            # policy gradients
            # pg_obj = .0
            # for i, theta in enumerate(self._theta[model_idx]):
            #     for idx, cidx, s, w in zip(
            #             index, cids, after - before if self._diff else after,
            #             weight):
            #         pg_obj += w * -1.0 * (s - baseline) * torch.log(
            #             torch.clip(theta[cidx][idx[i]], min=1e-8, max=1.0))
            # pg_loss = -1.0 * pg_obj
            # pg_loss.backward()
            # self._pn_optimizer.step()
            # self._policy_net.eval()
            # thetas4stat = [
            #     theta.detach().cpu().numpy()
            #     for theta in self._policy_net(self._client_encodings)
            # ]
        else:
            for i, (z, theta) in enumerate(
                    zip(self._z[model_idx], self._theta[model_idx])):
                grad = np.zeros(len(z))
                for idx, s, w in zip(index,
                                     after - before if self._diff else after,
                                     weight):
                    grad[idx[i]] += w * (s - baseline) / theta[idx[i]]
                if self._sched == 'adaptive':
                    self._store[model_idx][i] += norm(grad, float('inf'))**2
                    denom = np.sqrt(self._store[model_idx][i])
                elif self._sched == 'aggressive':
                    denom = 1.0 if np.all(
                        grad == 0.0) else norm(grad, float('inf'))
                elif self._sched == 'auto':
                    self._store[model_idx][i] += 1.0
                    denom = np.sqrt(self._store[model_idx][i])
                elif self._sched == 'constant':
                    denom = 1.0
                elif self._sched == 'scale':
                    denom = 1.0 / np.sqrt(2.0 * np.log(len(grad))) if len(
                        grad) > 1 else float('inf')
                else:
                    raise NotImplementedError
                eta = self._eta0[i] / denom
                z -= eta * grad
                z -= logsumexp(z)
                self._theta[model_idx][i] = np.exp(z)
            thetas4stat = self._theta[model_idx]

        self._trace[model_idx]['entropy'].append(self.entropy(thetas4stat))
        self._trace[model_idx]['mle'].append(self.mle(thetas4stat))
        if self._trace[model_idx]['entropy'][-1] < self._cutoff:
            self._stop_exploration[model_idx] = True

        logger.info(
            'Server: working on model: {};'
            'Updated policy as {} with entropy {:f} and mle {:f}'.format(
                model_idx, thetas4stat, self._trace[model_idx]['entropy'][-1],
                self._trace[model_idx]['mle'][-1]))

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):
        """
        To check the message_buffer, when enough messages are receiving,
        trigger some events (such as perform aggregation, evaluation,
        and move to the next training round)
        """
        if min_received_num is None:
            min_received_num = self._cfg.federate.sample_client_num
        assert min_received_num <= self.sample_client_num

        if check_eval_result:
            min_received_num = len(list(self.comm_manager.neighbors.keys()))

        move_on_flag = True  # To record whether moving to a new training
        # round or finishing the evaluation

        if self.check_buffer(self.state, min_received_num, check_eval_result):

            if not check_eval_result:  # in the training process

                tmp_mab_feedbacks = list()
                # Get all the message
                train_msg_buffer = self.msg_buffer['train'][self.state]
                model = self.models[self.current_model_idx]
                logger.info('self.current_model_idx: {}'.format(
                    self.current_model_idx))
                aggregator = self.aggregators[self.current_model_idx]
                tmp_msg_list = list()
                # in train_msg_buffer, client_id start from 1 ....
                client_id_order = []
                if self.has_info_matrix:

                    for client_id in train_msg_buffer:
                        client_id_order.append(client_id - 1)
                        tmp_msg_list.append(
                            tuple(train_msg_buffer[client_id][0:2]))
                        tmp_mab_feedbacks.append(
                            train_msg_buffer[client_id][2])

                else:
                    for client_id in train_msg_buffer:
                        self.update_val_info(train_msg_buffer[client_id][2])
                        if client_id - 1 != self.current_model_idx:
                            logger.info(
                                'add client: {} update to model: {}'.format(
                                    client_id, self.current_model_idx))
                            # model id = i means that this model
                            # represents the model without client i+1
                            # ** e.x.: total 3 clients, model id [0,1,2,3];
                            # *** model_id = 0: model without client 1
                            tmp_msg_list.append(
                                tuple(train_msg_buffer[client_id][0:2]))
                            tmp_mab_feedbacks.append(
                                train_msg_buffer[client_id][2])

                # change the order of tmp_mab_feedbacks & tmp_mab_feedbacks according to client id
                # ensure msg_list[i] and mab_feedbacks[i] is the client i+1's information
                msg_list = ['*'] * len(client_id_order)
                mab_feedbacks = ['*'] * len(client_id_order)
                for order_id in range(len(client_id_order)):
                    msg_list[
                        client_id_order[order_id]] = tmp_msg_list[order_id]
                    mab_feedbacks[client_id_order[
                        order_id]] = tmp_mab_feedbacks[order_id]

                # Trigger the monitor here (for training)
                self._monitor.calc_model_metric(
                    self.models[self.current_model_idx].state_dict(),
                    msg_list,
                    rnd=self.state)
                self._monitor.calc_model_metric(self.model.state_dict(),
                                                msg_list,
                                                rnd=self.state)

                # Aggregate
                agg_info = {
                    'client_feedback': msg_list,
                    'recover_fun': self.recover_fun
                }
                result = aggregator.aggregate(agg_info)
                model.load_state_dict(result, strict=False)

                # for model_idx in range(self.model_num):
                # model = self.models[model_idx]
                # aggregator = self.aggregators[model_idx]
                # msg_list = list()
                # for client_id in train_msg_buffer:
                #     if self.model_num == 1:
                #         msg_list.append(
                #             tuple(train_msg_buffer[client_id][0:2]))
                #     else:
                #         train_data_size, model_para_multiple = \
                #             train_msg_buffer[client_id][0:2]
                #         msg_list.append((train_data_size,
                #                          model_para_multiple[model_idx]))
                #
                #     # collect feedbacks for updating the policy
                #     mab_feedbacks.append(
                #         train_msg_buffer[client_id][2])
                #     if model_idx == 0:
                #         pass
                #
                # # Trigger the monitor here (for training)
                # self._monitor.calc_model_metric(self.model.state_dict(),
                #                                 msg_list,
                #                                 rnd=self.state)
                #
                # # Aggregate
                # agg_info = {
                #     'client_feedback': msg_list,
                #     'recover_fun': self.recover_fun
                # }
                # result = aggregator.aggregate(agg_info)
                # model.load_state_dict(result, strict=False)
                # aggregator.update(result)

                # update the policy
                self.update_policy(mab_feedbacks,
                                   model_idx=self.current_model_idx)

                if self.state != 0 and self.state % self._cfg.eval.freq == 0\
                        and self.state != \
                        self.total_round_num:
                    #  Evaluate
                    if self.current_model_idx == self.model_num - 1:
                        logger.info(
                            'Server: Starting evaluation at round {:d}.'.
                            format(self.state))
                        self.eval()
                if self.state % 50 == 0 and self.state > 1 \
                        and self.current_model_idx == self.model_num - 1:
                    logger.info('saving save_alpha_related_info')
                    self.save_alpha_related_info()
                    # self.save_client_eval_results()

                if self.state + 1 < self.total_round_num:
                    if (self.current_model_idx + 1) % self.model_num == 0:
                        self.state += 1
                        self.current_model_idx = 0
                        logger.info(f'----------- '
                                    f'Starting a new training round (Round '
                                    f'#{self.state}) -------------')
                        # Clean the msg_buffer
                        self.msg_buffer['train'][self.state - 1].clear()
                    else:
                        self.current_model_idx = (self.current_model_idx +
                                                  1) % self.model_num

                        logger.info(
                            '----------- Round: {};  work on model: {} '.
                            format(self.state, self.current_model_idx))
                        self.msg_buffer['train'][self.state].clear()

                    # Move to next round of training
                    # logger.info(
                    #     f'----------- Starting a new training round (Round '
                    #     f'#{self.state}) -------------')
                    # Clean the msg_buffer
                    # self.msg_buffer['train'][self.state - 1].clear()

                    self.broadcast_model_para(
                        msg_type='model_para',
                        sample_client_num=self.sample_client_num)
                else:
                    self.save_alpha_related_info()

                    logger.info('Server: Training is finished! Starting '
                                'evaluation.')
                    self.eval()

            else:  # in the evaluation process
                # Get all the message & aggregate
                formatted_eval_res = self.merge_eval_results_from_all_clients()
                self.history_results = merge_dict_of_results(
                    self.history_results, formatted_eval_res)
                self.check_and_save()
        else:
            move_on_flag = False

        return move_on_flag

    def check_and_save(self):
        """
        To save the results and save model after each evaluation
        """
        # early stopping
        should_stop = False

        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_weighted_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        elif "Results_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_avg']:
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])
        else:
            should_stop = False

        if should_stop:
            self.state = self.total_round_num + 1

        if should_stop or self.state == self.total_round_num:
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round
            self.save_best_results()

            if self._cfg.federate.save_to != '':
                # save the policy
                ckpt = dict()
                if self._cfg.hpo.fedex.psn:
                    psn_pi_ckpt_path = self._cfg.federate.save_to[:self._cfg.
                                                                  federate.
                                                                  save_to.
                                                                  rfind(
                                                                      '.'
                                                                  )] + \
                                       "_pfedex.pt"
                    torch.save(
                        {
                            'client_encodings': self._client_encodings,
                            'policy_net': self._policy_net.state_dict()
                        }, psn_pi_ckpt_path)
                else:
                    z_list = [
                        z.tolist() for z in self._z[self.current_model_idx]
                    ]
                    ckpt['z'] = z_list
                    ckpt['store'] = self._store
                ckpt['stop'] = self._stop_exploration
                ckpt['global'] = self.trace(
                    'global', model_id=self.current_model_idx).tolist()
                ckpt['refine'] = self.trace(
                    'refine', model_id=self.current_model_idx).tolist()
                ckpt['entropy'] = self.trace(
                    'entropy', model_id=self.current_model_idx).tolist()
                ckpt['mle'] = self.trace(
                    'mle', model_id=self.current_model_idx).tolist()
                pi_ckpt_path = self._cfg.federate.save_to[:self._cfg.federate.
                                                          save_to.rfind(
                                                              '.'
                                                          )] + "_fedex.yaml"
                with open(pi_ckpt_path, 'w') as ops:
                    yaml.dump(ckpt, ops)

            # if self.model_num > 1:
            #     model_para = [model.state_dict() for model in self.models]
            # else:
            #     model_para = self.model.state_dict()

            model_para = self.models[self.current_model_idx].state_dict()
            self.comm_manager.send(
                Message(msg_type='finish',
                        sender=self.ID,
                        receiver=list(self.comm_manager.neighbors.keys()),
                        state=self.state,
                        content=model_para))

        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1
