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
    def fit(self, optimizer, loss_fn, epochs: int, loss_fn_params=None):
        history = []
        best_loss = torch.inf
        epochs_without_improvement = 0

        for epoch in range(epochs):
            self.train()
            loss, status = loss_fn(self, loss_fn_params)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            history.append(status)

            # Early Stopping
            if loss < best_loss:
                best_loss = loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= 1000:
                print("Early stopping ativado. Treinamento interrompido.")
                break

            if torch.isnan(loss):
                raise ValueError("Loss is NaN. Treinamento interrompido.")

            # Notify epoch
            if epoch % 100 == 0:
                print(f"Epoch {epoch} loss: {loss}")
        print("Loss Final:", torch.sum(torch.tensor(history[-1])).item())
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
