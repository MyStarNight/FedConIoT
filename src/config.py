from src.nn_model import loss_fn


class Config:
    def __init__(self,
                 batch_size=32,
                 epochs=3,
                 optimizer="SGD",
                 optimizer_args=None,
                 loss=loss_fn,
                 shuffle=True,
                 training_rounds=5):
        if optimizer_args is None:
            optimizer_args = {"lr": 0.001}
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.loss = loss
        self.shuffle = shuffle
        self.training_rounds = training_rounds


config_dict = {
    "batch_size": 32,
    "epochs": 3,
    "optimizer": "SGD",
    "optimizer_args": {"lr": 0.001},
    "loss_fn": loss_fn,
    "shuffle": True
}
