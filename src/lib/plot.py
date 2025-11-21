import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "figure.figsize": (16, 6),
        "figure.autolayout": True,
        "axes.grid": True,
        # 'figure.dpi': 300,
        "font.size": 16,
    }
)


def legend_outside(ax: plt.Axes):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)


def plot_comparison(cycles, real_y, model_y, model_name):
    _, axes = plt.subplots(2, 1, sharex=True)
    variables = [
        ("Purity H2", "tab:blue", 0),
        ("H2/CO ratio", "tab:orange", 1),
        ("Purity CO2", "tab:green", 2),
        ("Recovery CO2", "tab:red", 3),
        ("Productivity", "tab:purple", 4),
    ]

    group1 = [0, 2, 3]
    ymin = min(real_y[:, variables[i][2]].min() for i in group1)
    ymax = max(real_y[:, variables[i][2]].max() for i in group1)
    ax = axes[0]
    for i in group1:
        label, color, idx = variables[i]
        ax.scatter(cycles, real_y[:, idx], label=f"{label} (gPROMS)", color=color)
        ax.plot(cycles, model_y[:, idx], label=f"{label} ({model_name})", color=color)
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.set_ylabel("Valores")
    legend_outside(ax)

    group2 = [1, 4]
    ymin = min(real_y[:, variables[i][2]].min() for i in group2)
    ymax = max(real_y[:, variables[i][2]].max() for i in group2)
    ax = axes[1]
    for i in group2:
        ax = axes[1]
        label, color, idx = variables[i]
        ax.scatter(cycles, real_y[:, idx], label=f"{label} (gPROMS)", color=color)
        ax.plot(cycles, model_y[:, idx], label=f"{label} ({model_name})", color=color)
    ax.set_ylim(ymin * 0.9, ymax * 1.1)
    ax.set_ylabel("Valores")
    legend_outside(ax)

    axes[-1].set_xlabel("Ciclo")

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
