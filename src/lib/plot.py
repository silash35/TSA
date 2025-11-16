import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.figsize": (16, 6),
        "figure.autolayout": True,
        "axes.grid": True,
        # 'figure.dpi': 300,
        "font.size": 15,
        "legend.loc": "upper right",
    }
)


def plot_comparison(cycles, real_y, model_y, model_name):
    plt.figure()

    plt.scatter(cycles, real_y[:, 0], label="Purity H2")
    plt.plot(cycles, model_y[:, 0], label=f"Purity H2 ({model_name})")

    plt.scatter(cycles, real_y[:, 1], label="H2/CO ratio")
    plt.plot(cycles, model_y[:, 1], label=f"H2/CO ratio ({model_name})")

    plt.scatter(cycles, real_y[:, 2], label="Purity CO2")
    plt.plot(cycles, model_y[:, 2], label=f"Purity CO2 ({model_name})")

    plt.scatter(cycles, real_y[:, 3], label="Recovery CO2")
    plt.plot(cycles, model_y[:, 3], label=f"Recovery CO2 ({model_name})")

    plt.scatter(cycles, real_y[:, 4], label="Productivity")
    plt.plot(cycles, model_y[:, 4], label=f"Productivity ({model_name})")

    plt.ylim(0, real_y.max() * 1.1)

    plt.xlabel("Ciclo")
    plt.ylabel("Saída")
    plt.legend()
    # plt.show()

    plt.savefig(f"../figures/comparision-{model_name}.png")
    plt.close()


def plot_loss(train_loss, validation_loss, model_name):
    plt.figure()
    plt.plot(train_loss, label="Training Loss")
    plt.plot(validation_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss over Epochs ({model_name})")
    plt.legend()
    # plt.show()

    plt.savefig(f"../figures/loss-{model_name}.png")
    plt.close()
