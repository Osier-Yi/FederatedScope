from federatedscope.register import register_scheduler


def call_my_scheduler(optimizer, reg_type):
    try:
        import torch.optim as optim
    except ImportError:
        optim = None
        scheduler = None

    if reg_type == 'myscheduler':
        if optim is not None:
            lr_lambda = [lambda epoch: epoch // 30]
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, verbose=False)
        return scheduler

    if reg_type == 'steplr':
        if optim is not None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5, verbose=False)
            return scheduler
    if reg_type == 'multisteplr':
        if optim is not None:
            milestones = [10, 30, 50, 100]

            # milestones = [3,5,7,9,11]
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, verbose=False)
            return scheduler





register_scheduler('myscheduler', call_my_scheduler)
register_scheduler('steplr', call_my_scheduler)
register_scheduler('multisteplr', call_my_scheduler)
