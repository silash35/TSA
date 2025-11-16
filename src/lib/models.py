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
        best_loss = torch.inf
        epochs_without_improvement = 0

        for epoch in range(epochs):
            # Train
            self.train()

            Y_pred = self(X_train)
            train_loss = self.loss_fn(Y_pred, Y_train)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            # Validation
            self.eval()
            with torch.no_grad():
                Y_pred_val = self(X_val)
                val_loss = self.loss_fn(Y_pred_val, Y_val)

            # Save history
            history["train_loss"].append(train_loss.item())
            history["val_loss"].append(val_loss.item())

            # Early Stopping
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 1000:
                print("Early stopping ativado. Treinamento interrompido.")
                break

            if torch.isnan(train_loss):
                raise ValueError("Loss is NaN. Treinamento interrompido.")

            # Notify epoch
            if epoch % 500 == 0:
                print(
                    f"Epoch {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}"
                )
        print("Final Training Loss:", history["train_loss"][-1])
        print("Final Validation Loss:", history["val_loss"][-1])
        return history


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
