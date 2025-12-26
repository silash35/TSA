import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from .read_data import u_labels, y_labels

colors = list(mcolors.TABLEAU_COLORS.keys())

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
    ymin = min(real_y[:, i].min() for i in range(len(y_labels)))
    ymax = max(real_y[:, i].max() for i in range(len(y_labels)))

    for i in range(len(y_labels)):
        plt.scatter(
            cycles,
            real_y[:, i],
            label=f"{y_labels[i]} (gPROMS)",
            color=colors[i],
            alpha=0.6,
        )
        plt.plot(
            cycles,
            model_y[:, i],
            label=f"{y_labels[i]} ({model_name})",
            color=colors[i],
        )
    plt.ylim(ymin * 0.9, ymax * 1.1)

    legend_outside(plt.gca())
    plt.ylabel("Valores / %")
    plt.xlabel("Ciclo")

    # plt.show()
    plt.savefig(f"../figures/comparision-{model_name}.png")
    plt.close()


def plot_control(cycles, u, y, sp, u_sp, model_name: str):
    # --- Criar subplots ---
    _, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

    # Entradas
    for i in range(len(u_labels)):
        axes[0].plot(cycles, u[:, i], label=u_labels[i])
        # axes[0].plot(cycles, u_sp[:, i], linestyle="--", color=colors[i], label=f"{u_labels[i]} (SP)")

    axes[0].set_ylabel("Duração da etapa / s")

    # Saídas
    for i in range(len(y_labels)):
        axes[1].plot(cycles, y[:, i], label=y_labels[i], color=colors[i])
        axes[1].plot(
            cycles,
            sp[:, i],
            linestyle="--",
            color=colors[i],
            label=f"{y_labels[i]} (SP)",
        )

    axes[1].set_ylabel("KPIs / %")

    axes[-1].set_xlabel("Ciclo")
    for ax in axes:
        legend_outside(ax)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"../figures/{model_name}.png")
    plt.close()
