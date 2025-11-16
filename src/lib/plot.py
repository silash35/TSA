import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.figsize": (16, 6),
        # 'figure.dpi': 300,
        "font.size": 15,
        "legend.loc": "upper right",
    }
)


def plot_comparison(cycles, real_y, model_y, model_name):
    plt.figure()

    plt.plot(cycles, real_y[:, 0], label="Purity H2")
    plt.scatter(cycles, model_y[:, 0], label=f"Purity H2 ({model_name})")

    plt.plot(cycles, real_y[:, 1], label="H2/CO ratio")
    plt.scatter(cycles, model_y[:, 1], label=f"H2/CO ratio ({model_name})")

    plt.plot(cycles, real_y[:, 2], label="Purity CO2")
    plt.scatter(cycles, model_y[:, 2], label=f"Purity CO2 ({model_name})")

    plt.plot(cycles, real_y[:, 3], label="Recovery CO2")
    plt.scatter(cycles, model_y[:, 3], label=f"Recovery CO2 ({model_name})")

    plt.plot(cycles, real_y[:, 4], label="Productivity")
    plt.scatter(cycles, model_y[:, 4], label=f"Productivity ({model_name})")

    plt.ylim(0, real_y.max() * 1.1)

    plt.xlabel("Ciclo")
    plt.ylabel("Saída")
    plt.legend()
    plt.grid(True)
    plt.show()
