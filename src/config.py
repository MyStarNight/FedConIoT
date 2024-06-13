from src.nn_model import loss_fn

config = {
    "batch_size": 32,
    "epochs": 3,
    "optimizer": "SGD",
    "optimizer_args": {"lr": 0.001},
    "loss_fn": loss_fn,
    "shuffle": True
}