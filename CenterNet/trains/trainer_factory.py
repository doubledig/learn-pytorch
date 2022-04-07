from trains.ctdet_train import CtdetTrainer

train_factory = {
    'ctdet': CtdetTrainer
}


def get_trainer(opt, model, optimizer):
    return train_factory[opt.task](opt, model, optimizer)
