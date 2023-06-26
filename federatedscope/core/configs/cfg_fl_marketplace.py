import logging

from federatedscope.core.configs.config import CN
from federatedscope.register import register_config
from federatedscope.core.monitors.metric_calculator import SUPPORT_METRICS

logger = logging.getLogger(__name__)


def extend_marketplace_cfg(cfg):
    cfg.marketplace = CN()

    # ---------------------------------------------------------------------- #
    # hfl alpha-tunning related options
    # ---------------------------------------------------------------------- #
    cfg.marketplace.alpha_tune = CN()
    cfg.marketplace.alpha_tune_use = False
    cfg.marketplace.alpha_tune.hpo = 'fedex'
    cfg.marketplace.alpha_tune.client_bid = []
    cfg.marketplace.alpha_tune.working_folder = 'alpha_tune_marketplace'
    cfg.marketplace.alpha_tune.valuation_method = 'naive_influence_score'
    cfg.marketplace.alpha_tune.tau_v = 1.0
    cfg.marketplace.alpha_tune.tau_alpha = 1.0
    cfg.marketplace.alpha_tune.tau_p = 1.0
    cfg.marketplace.alpha_tune.entropy_determined = False
    cfg.marketplace.alpha_tune.train_weight_control = False
    cfg.marketplace.alpha_tune.train_weight = []
    cfg.marketplace.alpha_tune.val_info_pth = ''
    cfg.marketplace.alpha_tune.update_hp = True

    cfg.marketplace.alpha_tune.aggregation_weight_sha_use = False
    cfg.marketplace.alpha_tune.aggregation_weight_candidates = []
    cfg.marketplace.alpha_tune.aggregation_weight_sha_round = 5
    cfg.marketplace.alpha_tune.aggregation_weight_sha_metric = 'acc' # 'acc', 'f1', 'avg_loss'
    cfg.marketplace.alpha_tune.is_normalize_alpha = True
    cfg.marketplace.alpha_tune.fedex_metric = 'avg_loss' #'acc', 'f1', 'avg_loss'
    cfg.marketplace.alpha_tune.inf_matrix_metric = 'avg_loss' # 'acc', 'f1', 'avg_loss'


    # --------------- register corresponding check function ----------
    cfg.register_cfg_check_fun(assert_marketplace_cfg)


def assert_marketplace_cfg(cfg):
    if cfg.marketplace.alpha_tune_use:
        if cfg.federate.client_num > 0:
            assert len(
                cfg.marketplace.alpha_tune.client_bid)\
                   == cfg.federate.client_num, 'The number of client bids ' \
                                               'must be qual with with the ' \
                                               'total client number'
        else:
            assert len(cfg.marketplace.alpha_tune.client_bid
                       ) > 0, 'the client bids are not provided.'

    if cfg.marketplace.alpha_tune_use and \
            cfg.marketplace.alpha_tune.valuation_method\
            == 'naive_influence_score':
        assert cfg.model.model_num_per_trainer \
               == \
               cfg.federate.client_num + 2, 'if use naive ' \
                                            'influence score, the ' \
                                            'cfg.model.model_num_per_trainer' \
                                            'must equal to ' \
                                            'cfg.federate.client_num plus 2'
    if cfg.marketplace.alpha_tune.train_weight_control:
        assert len(cfg.marketplace.alpha_tune.train_weight) \
               == len(cfg.marketplace.alpha_tune.client_bid)


register_config("marketplace", extend_marketplace_cfg)
