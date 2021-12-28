from convolution import CNNBackbone
from lstm import LSTMBackbone
from modules import Classifier, MultitaskRegressor, Regressor


def evaluate(model, test_dataloader, device="cuda"):
    raise NotImplementedError("You need to implement this")
    return ...  # Return the model predictions


def kaggle_submission(model, test_dataloader, device="cuda"):
    outputs = evaluate(model, test_dataloader, device=device)
    # TODO: Write a csv file for your kaggle submmission
    raise NotImplementedError("You need to implement this")


if __name__ == "__main__":
    backbone = ...
    model = ...
    test_dataloader = ...
    device = ...
    kaggle_submission(model, test_dataloader, device=device)
