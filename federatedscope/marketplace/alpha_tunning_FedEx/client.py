from federatedscope.autotune.fedex.client import FedExClient
import logging

logger = logging.getLogger(__name__)

import logging
import json
import copy

from federatedscope.core.message import Message
from federatedscope.core.workers import Client


class AlphaFedExClient(FedExClient):
    def callback_funcs_for_model_para(self, message: Message):
        round, sender, content = message.state, message.sender, message.content
        # model_id = content.model_id
        model_params, arms, hyperparams, model_id = content[
            "model_param"], content["arms"], content["hyperparam"], content[
                "model_id"]
        attempt = {
            'Role': 'Client #{:d}'.format(self.ID),
            'Round': round,
            'Arms': arms,
            'Hyperparams': hyperparams,
            'Model_id': model_id
        }
        logger.info(json.dumps(attempt))

        if self._cfg.marketplace.alpha_tune.update_hp:

            logger.info('==== updating hyperparams =====')

            self._apply_hyperparams(hyperparams)

        self.trainer.update(model_params)
        # logger.info("self.trainer.ctx.scheduler.get_last_lr(): ".format(self.trainer.ctx.scheduler.get_last_lr()))
        # logger.info("self.trainer.ctx.scheduler._step_count: ".format(self.trainer.ctx.scheduler._step_count))

        # self._step_coun

        # self.model.load_state_dict(content)
        self.state = round
        sample_size, model_para_all, results = self.trainer.train()
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para_all = copy.deepcopy(model_para_all)
        logger.info(
            self._monitor.format_eval_res(results,
                                          rnd=self.state,
                                          role='Client #{} Model #{}'.format(
                                              self.ID, model_id),
                                          return_raw=True))

        results['arms'] = arms
        results['client_id'] = self.ID - 1
        content = (sample_size, model_para_all, results, model_id)
        self.comm_manager.send(
            Message(msg_type='model_para',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=content))

    def callback_funcs_for_evaluate(self, message: Message):
        sender = message.sender
        self.state = message.state
        model_id = -1
        if message.content is not None:
            model_params, model_id = message.content[
                "model_param"], message.content["model_id"]
            self.trainer.update(model_params)
        if self._cfg.finetune.before_eval:
            self.trainer.finetune()
        metrics = {}
        for split in self._cfg.eval.split:
            eval_metrics = self.trainer.evaluate(target_data_split_name=split)

            for key in eval_metrics:

                if self._cfg.federate.mode == 'distributed':
                    logger.info(
                        'Client #{:d}: (Evaluation ({:s} set) at '
                        'Round #{:d}) {:s} on Model #{:d} is {:.6f}'.format(
                            self.ID, split, self.state, key, model_id,
                            eval_metrics[key]))
                metrics.update(**eval_metrics)
        logger.info(
            self._monitor.format_eval_res(metrics,
                                          rnd=self.state,
                                          role='Client #{} Model #{}'.format(
                                              self.ID, model_id),
                                          return_raw=True))
        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    content=metrics))
