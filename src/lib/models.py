import copy
from typing import TypedDict

import torch
from torch import nn


def normalize(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    return ((x - x_min) / (x_max - x_min)) * 2 - 1


def denormalize(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    return ((x + 1) / 2) * (x_max - x_min) + x_min


class DataInfo(TypedDict):
    in_min: torch.Tensor
    in_max: torch.Tensor
    out_min: torch.Tensor
    out_max: torch.Tensor


class BaseModel(nn.Module):
    def loss_fn(self, Y_pred, Y_true, params=None):
        return torch.mean((Y_pred - Y_true) ** 2)

    def _fit(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        optimizer,
    ):
        self.train()
        Y_pred = self(X_train)
        train_loss = self.loss_fn(Y_pred, Y_train)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # validation step
        self.eval()
        with torch.no_grad():
            Y_pred_val = self(X_val)
            val_loss = self.loss_fn(Y_pred_val, Y_val)
        return train_loss, val_loss

    def fit(
        self,
        X_train,
        Y_train,
        X_val,
        Y_val,
        optimizer,
        epochs: int,
    ):
        history = {"train_loss": [], "val_loss": []}

        best_val_loss = torch.inf
        best_state_dict = None
        best_epoch = -1

        for epoch in range(epochs):
            train_loss, val_loss = self._fit(
                X_train,
                Y_train,
                X_val,
                Y_val,
                optimizer,
            )

            # record history
            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())

            # save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state_dict = copy.deepcopy(self.state_dict())
                best_epoch = epoch

            # print progress
            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"train_loss={train_loss.item():.6f} | "
                    f"val_loss={val_loss.item():.6f}"
                )

        print("Final Training Loss:", history["train_loss"][-1])
        print("Final Validation Loss:", history["val_loss"][-1])

        # load best model
        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)

        return history, best_epoch


class FNN(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataInfo: DataInfo | None = None,
    ):
        super(FNN, self).__init__()
        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

        if dataInfo is not None:
            self.in_min = dataInfo["in_min"]
            self.in_max = dataInfo["in_max"]
            self.out_min = dataInfo["out_min"]
            self.out_max = dataInfo["out_max"]
        else:
            self.in_min = -torch.ones(input_size)
            self.in_max = torch.ones(input_size)
            self.out_min = -torch.ones(output_size)
            self.out_max = torch.ones(output_size)

    def forward(self, x, disable_norm: bool = False):
        if disable_norm:
            return self.hidden_layer(x)

        return denormalize(
            self.hidden_layer(normalize(x, self.in_min, self.in_max)),
            self.out_min,
            self.out_max,
        )


class RNN(BaseModel):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dataInfo: DataInfo | None = None,
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.RNN = nn.RNN(
            input_size,
            hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.FNN = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
        )

        if dataInfo is not None:
            self.in_min = dataInfo["in_min"]
            self.in_max = dataInfo["in_max"]
            self.out_min = dataInfo["out_min"]
            self.out_max = dataInfo["out_max"]
        else:
            self.in_min = -torch.ones(input_size)
            self.in_max = torch.ones(input_size)
            self.out_min = -torch.ones(output_size)
            self.out_max = torch.ones(output_size)

    def forward(self, x: torch.Tensor, disable_norm: bool = False) -> torch.Tensor:
        if disable_norm is False:
            x = normalize(x, self.in_min, self.in_max)

        rnn_output, _ = self.RNN(x)
        output = self.FNN(rnn_output)

        if disable_norm is False:
            output = denormalize(output, self.out_min, self.out_max)

        # Para facilitar, selecionamos apenas o valor de saída da ultima camada
        # Similar ao return_sequences=False do Keras
        return output[:, -1, :]  # Saida no formato de (batch_size, output_size)
