import torch


def normalize(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    return ((x - x_min) / (x_max - x_min)) * 2 - 1


def denormalize(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor
) -> torch.Tensor:
    return ((x + 1) / 2) * (x_max - x_min) + x_min


def prepare_sequences(
    input_sequences, target_sequences, history_size: int, overlap=True
):
    step = 1 if overlap else history_size

    # Crie as entradas (X_true) a partir das sequências usando unfold
    input_sequences = [
        seq.unfold(0, history_size, step)[:-1] for seq in input_sequences
    ]

    # Empilhe e permute as dimensões para obter o tensor final de entradas
    X_true = torch.stack(input_sequences).permute(1, 2, 0)

    # Crie os alvos (Y_true) pegando o último valor de cada sequência
    target_seqs = [
        seq.unfold(0, history_size, step)[1:, -1] for seq in target_sequences
    ]

    # Empilhe e permute as dimensões para obter o tensor final de alvos
    Y_true = torch.stack(target_seqs).permute(1, 0)

    return X_true, Y_true
