from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, MultitaskRegressor, Regressor


def save_checkpoint(model):
    raise NotImplementedError("You need to implement this")


def training_loop(model, train_dataloader, optimizer, device="cuda"):
    for batch in train_dataloader:
        raise NotImplementedError("You need to implement this")

    return ...  # Return train_loss and anything else you need


def validation_loop(model, val_dataloader, device="cuda"):
    for batch in val_dataloader:
        raise NotImplementedError("You need to implement this")

    return ...  # Return validation_loss and anything else you need


def overfit_with_a_couple_of_batches(model, train_dataloader):
    epochs = 10000  # An absurd number of epochs
    # TODO: select a couple of batches from the dataloader
    # TODO: Run the training loop for an absurd amount of epochs
    raise NotImplementedError("You need to implement this")


def train(model, train_dataloader, val_dataloader, optimizer, epochs, device="cuda", overfit_batch=False):
    if overfit_batch:
        overfit_with_a_couple_of_batches(model, train_dataloader)
    else:
        for epoch in range(epochs):
            # TODO: training loop
            # TODO: validation loop
            # TODO: early stopping
    raise NotImplementedError("You need to implement this")


if __name__ == "__main__":
    backbone = ...
    model = ...
    optimizer = ...
    epochs = ...
    train_dataloader = ...
    val_dataloader = ...
    device = ...
    overfit_batch = ...
    train(model, train_dataloader, val_dataloader, optimizer, epochs, device=device, overfit_batch=overfit_batch)

