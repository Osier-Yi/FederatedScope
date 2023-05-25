from federatedscope.core.aggregators import ClientsAvgAggregator
from federatedscope.core.auxiliaries.utils import param2tensor
import torch
import logging
logger = logging.getLogger(__name__)

class AlphaTuneAggretor(ClientsAvgAggregator):
    def __init__(self, model=None, device='cpu', config=None, train_weight=None):
        super(ClientsAvgAggregator, self).__init__()
        self.model = model
        self.device = device
        self.cfg = config
        if train_weight is not None:
            self.weight = train_weight
        else:
            self.weight = self.cfg.marketplace.alpha_tune.train_weight

    def update_train_weight(self):
        # TODO update the weight
        logger.info('Current weight: {}'.format(self.weight))

    def aggregate(self, agg_info):
        """
        To preform aggregation

        Arguments:
            agg_info (dict): the feedbacks from clients

        Returns:
            dict: the aggregated results
        """
        self.update_train_weight()
        # logger.info()

        models = agg_info["client_feedback"]
        recover_fun = agg_info['recover_fun'] if (
            'recover_fun' in agg_info and self.cfg.federate.use_ss) else None
        avg_model = self._para_weighted_avg(models, recover_fun=recover_fun)

        return avg_model

    def _para_weighted_avg(self, models, recover_fun=None):
        """
        Calculates the weighted average of models.
        """
        logger.info("aggregation weight: {}".format(self.weight))

        training_set_size = 0
        for i in range(len(models)):
            sample_size, _ = models[i]
            training_set_size += sample_size

        sample_size, avg_model = models[0]
        for key in avg_model:
            for i in range(len(models)):
                local_sample_size, local_model = models[i]
                weight = self.weight[i]
                #
                # if self.cfg.federate.ignore_weight:
                #     weight = 1.0 / len(models)
                # elif self.cfg.federate.use_ss:
                #     # When using secret sharing, what the server receives
                #     # are sample_size * model_para
                #     weight = 1.0
                # else:
                #     weight = local_sample_size / training_set_size

                if not self.cfg.federate.use_ss:
                    local_model[key] = param2tensor(local_model[key])
                if i == 0:
                    avg_model[key] = local_model[key] * weight
                else:
                    avg_model[key] += local_model[key] * weight

            if self.cfg.federate.use_ss and recover_fun:
                avg_model[key] = recover_fun(avg_model[key])
                # When using secret sharing, what the server receives are
                # sample_size * model_para
                avg_model[key] /= training_set_size
                avg_model[key] = torch.FloatTensor(avg_model[key])

        return avg_model
