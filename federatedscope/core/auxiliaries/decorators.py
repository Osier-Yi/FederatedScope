def use_diff(func):
    def wrapper(self, *args, **kwargs):
        if self.cfg.federate.use_diff or self.cfg.marketplace.alpha_tune.aggregation_weight_sha_use:
            # TODO: any issue for subclasses?
            before_metric = self.evaluate(target_data_split_name='val')



        num_samples_train, model_para, result_metric = func(
            self, *args, **kwargs)

        if self.cfg.marketplace.alpha_tune.aggregation_weight_sha_use:
            if 'val_f1' in before_metric.keys():
                result_metric['val_f1'] = before_metric['val_f1']
            if 'val_acc' in before_metric.keys():
                result_metric['val_acc'] = before_metric['val_acc']


        if self.cfg.federate.use_diff:
            # TODO: any issue for subclasses?
            after_metric = self.evaluate(target_data_split_name='val')
            result_metric['val_total'] = before_metric['val_total']
            result_metric['val_avg_loss_before'] = before_metric[
                'val_avg_loss']
            result_metric['val_avg_loss_after'] = after_metric['val_avg_loss']



        return num_samples_train, model_para, result_metric

    return wrapper
