import matplotlib.pyplot as plt
import pandas as pd

# Substitua pelo caminho correto do arquivo
csv_file = "gPROMS/results.csv"

# Pega o cabeçalho
colunas = pd.read_csv(csv_file, skiprows=10, nrows=0).columns.to_list()

# Lê o CSV (a partir da linha onde começam as variáveis)
df = pd.read_csv(csv_file, skiprows=4368, names=colunas)

# Converte colunas numéricas
df["TIME"] = pd.to_numeric(df["TIME"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df["NOCOMP"] = pd.to_numeric(df["NOCOMP"], errors="coerce")
if "X" in df.columns:  # garante que X existe
    df["X"] = pd.to_numeric(df["X"], errors="coerce")

# ---- FILTRA A VARIÁVEL ----
var = "simul.qi"
df_var = df[df["Variable"] == var].copy()

comps = ["CO2", "CO", "H2"]

# ---- PLOT ----
for n in sorted(df_var["NOCOMP"].dropna().unique()):
    subset_n = df_var[df_var["NOCOMP"] == n]

    plt.figure(figsize=(10, 6))

    # lista de posições z únicas e ordenadas
    z_vals = sorted(subset_n["X"].dropna().unique())

    # seleciona só ~10 valores espaçados
    step = max(1, len(z_vals) // 10)  # garante pelo menos 1
    z_selecionados = [z_vals[i] for i in range(0, len(z_vals), step)]
    if z_vals[-1] not in z_selecionados:
        z_selecionados.append(z_vals[-1])  # garante que o último entra

    # plota só os selecionados
    for x in z_selecionados:
        subset_x = subset_n[subset_n["X"] == x]
        plt.plot(subset_x["TIME"], subset_x["Value"], label=f"z={x:.4f}")

    # Plot timestamp
    t_feed_duration = 679.5
    t_rinse_duration = 187
    t_blow_duration = (0.55e5 - 4.8e5) / (-3158)
    t__purge_duration = 80

    t_feed_start = 0
    t_rinse_start = t_feed_start + t_feed_duration
    t_blow_start = t_rinse_start + t_rinse_duration
    t_purge_start = t_blow_start + t_blow_duration
    plt.axvline(x=t_feed_start, color="g", linestyle="--", label="Feed Start")
    plt.axvline(x=t_rinse_start, color="r", linestyle="--", label="Rinse Start")
    plt.axvline(x=t_blow_start, color="b", linestyle="--", label="Blow Start")
    plt.axvline(x=t_purge_start, color="m", linestyle="--", label="Purge Start")

    plt.xlabel("Tempo [s]")
    plt.ylabel(var)
    plt.title(f"{var} - {comps[int(n - 1)]}")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"gPROMS/figures/{var}-{comps[int(n - 1)]}.png")
    plt.close()
