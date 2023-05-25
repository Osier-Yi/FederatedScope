from federatedscope.register import register_metric



def eval_confusion_matrix(y_true, y_pred, **kwargs):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred)

def eval_classification_report(y_true, y_pred, **kwargs):
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred)


# def MyMetric(ctx, **kwargs):
#     return ctx.num_train_data


def call_confusion_matrix(types):
    if 'confusion_matrix' in types:
        the_larger_the_better = True
        metric_builder = eval_confusion_matrix
        return 'confusion_matrix', metric_builder, the_larger_the_better

def call_classification_report(types):
    if 'classification_report' in types:
        the_larger_the_better = True
        metric_builder = eval_classification_report
        return 'classification_report', metric_builder, the_larger_the_better

# def call_my_metric(types):
#     if METRIC_NAME in types:
#         the_larger_the_better = True
#         metric_builder = MyMetric
#         return METRIC_NAME, metric_builder, the_larger_the_better


register_metric('confusion_matrix', call_confusion_matrix)
register_metric('classification_report', call_classification_report)
