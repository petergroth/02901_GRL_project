import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
from matplotlib import rc
import numpy as np
import pandas as pd
import yaml
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


def show_histograms():
    df = pd.read_csv("data/cm.csv", index_col=0)
    colors = sns.color_palette("colorblind", 5)
    fig, ax = plt.subplots(5, 1, figsize=(5, 14))
    plt.subplots_adjust(hspace=0.3)
    indicators = (0, 1, 2, 3, 4)
    split_names = ("Wildtype", "bmDCA (T=0.33)", "bmDCA (T=0.66)", "bmDCA (T=1.0)", "Profile model")
    low, high = df["target_reg"].min(), df["target_reg"].max()
    for i, (indicator, split_name) in enumerate(zip(indicators, split_names)):
        axi = ax[i]
        sns.histplot(data=df.loc[df["indicator"] == indicator], x="target_reg", ax=axi, bins=15, binrange=(low, high),
                     color=colors[i])
        axi.set_title(split_name)
        axi.set_xlabel("Normalized enzyme activity")
    plt.tight_layout()
    plt.savefig("report/graphics/histogram.pdf")
    plt.show()


    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(data=df.loc[df["indicator"].isin((0, 1, 2))], x="target_reg", ax=ax, bins=15, binrange=(low, high),
                 color="#f58a00")
    ax.set_title("Histogram over processed dataset")
    ax.set_xlabel("Normalized enzyme activity")
    ax.axvline(x=0.42, color="black", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig("report/graphics/histogram_used.pdf")
    plt.show()



def show_learning_curve(path: str):
    metrics_path = f"{path}/metrics.csv"
    params_path = f"{path}/hparams.yaml"

    # Read parameters
    with open(params_path, "r") as stream:
        params = yaml.safe_load(stream)

    df = pd.read_csv(metrics_path)
    ckpt_idx = df["val_bce"].idxmin()
    ckpt_epoch = df.loc[ckpt_idx, "epoch"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.lineplot(
        data=df, x="epoch", y="train_bce_epoch", ax=ax, linewidth=2, label="Training"
    )
    sns.lineplot(
        data=df, x="epoch", y="val_bce", ax=ax, linewidth=2, label="Validation"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE")
    ax.set_title(
        f"Model: {params['model_name']}. ESM: {params['esm']}. Test MCC: {df['test_mcc'].iloc[-1]:.4f}. "
        f"Test BCE: {df['test_bce'].iloc[-1]:.4f}"
    )
    ax.axvline(x=ckpt_epoch, color="black", alpha=0.2, linestyle="--")

    plt.show()


def main():
    # path = "logs/lightning_logs/version_6"
    # show_learning_curve(path)
    n_setups = 7
    seeds = np.zeros((n_setups * 3))
    test_mcc = np.zeros((n_setups * 3))
    test_bce = np.zeros((n_setups * 3))
    setups = np.zeros((n_setups * 3))

    k = 0
    for setup in range(n_setups):
        for version in range(3):
            df = pd.read_csv(f"logs/{str(setup)}/version_{str(version)}/metrics.csv")
            seeds[k] = version
            setups[k] = setup
            test_mcc[k] = df.iloc[-1]["test_mcc"]
            test_bce[k] = df.iloc[-1]["test_bce"]
            k += 1

    df = pd.DataFrame({"seed": seeds, "setup": setups, "test_mcc": test_mcc, "test_bce": test_bce})

    df["setup"] = df["setup"].astype(int).replace(
        {
            0: "MLP (MSA)",
            1: "MLP (ESM)",
            2: "GAT (1-of-k)",
            3: "GAT (ESM)",
            4: "GCN (1-of-k)",
            5: "GCN (ESM)",
            6: "GAT (ESM, edges)"
        }
    )

    order_dict = {
        "MLP (MSA)": 0,
        "MLP (ESM)": 1,
        "GCN (1-of-k)": 2,
        "GCN (ESM)": 3,
        "GAT (1-of-k)": 4,
        "GAT (ESM)": 5,
        "GAT (ESM, edges)": 6,
    }
    df = df.sort_values(by=["setup"], key=lambda x: x.map(order_dict))

    palette = sns.color_palette("Paired", 4)
    extra_col = sns.color_palette("YlOrBr", 3)
    for i in range(3):
        palette.append(extra_col[i])


    best_val = df.loc[df["setup"] == "GAT (ESM)", "test_mcc"].agg("mean")

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(data=df, x="setup", y="test_mcc", ax=ax, palette=palette, capsize=0.3,
                errwidth=1)
    ax.set_ylabel("Matthews correlation coefficient")
    ax.set_xlabel("")
    ax.axhline(y=best_val, color="black", alpha=0.2, linestyle="--")
    ax.tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    plt.savefig("report/graphics/results_mcc.pdf")
    plt.show()

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.barplot(data=df, x="setup", y="test_bce", ax=ax, palette=palette, capsize=0.3,
                errwidth=1)
    ax.set_ylabel("Binary cross-entropy")
    ax.set_xlabel("")
    ax.tick_params(axis='x', labelrotation=45)

    plt.tight_layout()
    plt.savefig("report/graphics/results_bce.pdf")

    plt.show()


if __name__ == "__main__":
   main()
   # show_histograms()
