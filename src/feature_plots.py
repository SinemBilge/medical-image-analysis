import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def summary_to_dataframe(summary):
    """Convert nested summary dict into a flat pandas DataFrame for easier plotting."""
    rows = []
    for layer, patients in summary.items():
        for patient, slices in patients.items():
            for slice_idx, vals in slices.items():
                for entry in vals["top_mad"]:
                    rows.append({
                        "Layer": layer,
                        "Patient": patient,
                        "Slice": slice_idx,
                        "Metric": "MAD",
                        "Value": entry["MAD"],
                        "Channel": entry["Channel"],
                        "MSE": entry["MSE"],
                        "SSIM": entry["SSIM"]
                    })
                for entry in vals["top_mse"]:
                    rows.append({
                        "Layer": layer,
                        "Patient": patient,
                        "Slice": slice_idx,
                        "Metric": "MSE",
                        "Value": entry["MSE"],
                        "Channel": entry["Channel"],
                        "MAD": entry["MAD"],
                        "SSIM": entry["SSIM"]
                    })
    return pd.DataFrame(rows)


def plot_layer_boxplots(summary, save_dir="plots"):
    """Boxplots of MAD & MSE values per layer (bigger, cleaner)."""
    df = summary_to_dataframe(summary)
    os.makedirs(save_dir, exist_ok=True)

    for metric in ["MAD", "MSE"]:
        plt.figure(figsize=(16, 7))
        sns.boxplot(data=df[df["Metric"] == metric], x="Layer", y="Value")
        plt.title(f"{metric} Distribution per Layer", fontsize=14)
        plt.xticks(rotation=60, ha="right")
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_boxplot_per_layer.png"))
        plt.close()


def plot_slice_heatmap(summary, metric="MAD", save_dir="plots"):
    """Heatmap of average MAD/MSE per patient × slice (averaged over layers)."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric].groupby(["Patient", "Slice"])["Value"].mean().unstack()

    plt.figure(figsize=(16, 8))
    sns.heatmap(df_metric, cmap="viridis", annot=False, cbar_kws={'label': f"Avg {metric}"})
    plt.title(f"Average {metric} per Patient × Slice", fontsize=14)
    plt.xlabel("Slice")
    plt.ylabel("Patient")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_heatmap_patient_slice.png"))
    plt.close()


def plot_mad_mse_scatter(summary, save_dir="plots"):
    """Scatter plot of MAD vs MSE colored by layer."""
    df = summary_to_dataframe(summary)
    df_mad = df[df["Metric"] == "MAD"].copy()
    df_mse = df[df["Metric"] == "MSE"].copy()

    merged = pd.merge(
        df_mad, df_mse,
        on=["Layer", "Patient", "Slice", "Channel"],
        suffixes=("_mad", "_mse")
    )

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        merged["Value_mad"], merged["Value_mse"],
        c=pd.factorize(merged["Layer"])[0], alpha=0.5, cmap="viridis"
    )
    plt.xlabel("MAD")
    plt.ylabel("MSE")
    plt.title("MAD vs MSE (colored by Layer)", fontsize=14)
    plt.colorbar(scatter, label="Layer Index")
    plt.grid(True, linestyle="--", alpha=0.6)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "mad_vs_mse_scatter.png"))
    plt.close()


def plot_patient_slice_profiles(summary, metric="MAD", save_dir="plots"):
    """Line plot of MAD/MSE values across slices for each patient (averaged across layers)."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric].groupby(["Patient", "Slice"])["Value"].mean().reset_index()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_metric, x="Slice", y="Value", hue="Patient", marker="o")
    plt.title(f"{metric} across slices per patient", fontsize=14)
    plt.xlabel("Slice")
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_slice_profile_per_patient.png"))
    plt.close()


def plot_layer_slice_heatmap(summary, metric="MAD", save_dir="plots"):
    """Heatmap of average MAD/MSE per layer × slice (averaged across patients)."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric].groupby(["Layer", "Slice"])["Value"].mean().unstack()

    plt.figure(figsize=(16, 10))
    sns.heatmap(df_metric, cmap="magma", annot=False, cbar_kws={'label': f"Avg {metric}"})
    plt.title(f"Average {metric} per Layer × Slice", fontsize=14)
    plt.xlabel("Slice")
    plt.ylabel("Layer")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_heatmap_layer_slice.png"))
    plt.close()
    

def plot_channel_frequency(summary, metric="MAD", save_dir="plots"):
    """
    Histogram of how often each channel appears in the top-3 values
    for the given metric across all layers/patients/slices.
    """
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric]

    counts = df_metric["Channel"].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    sns.barplot(x=counts.index, y=counts.values, color="skyblue", edgecolor="black")
    plt.title(f"Top-3 Channel Frequency ({metric})", fontsize=14)
    plt.xlabel("Channel")
    plt.ylabel("Frequency in Top-3")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_channel_frequency.png"))
    plt.close()

def plot_top_frequencies_per_layer(summary, metric="MAD", save_dir="mır/plots"):
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric]

    patients = df_metric["Patient"].unique()
    os.makedirs(save_dir, exist_ok=True)

    for patient_id in patients:
        df_patient = df_metric[df_metric["Patient"] == patient_id]

        counts = (
            df_patient.groupby(["Layer", "Slice", "Channel"])
            .size()
            .reset_index(name="Count")
        )

        plt.figure(figsize=(18, 8))
        ax = sns.barplot(
            data=counts,
            x="Layer", y="Count", hue="Slice",
            dodge=True
        )

        for patch, (_, row) in zip(ax.patches, counts.iterrows()):
            if row["Count"] > 0:
                ax.annotate(
                    f"Ch{int(row['Channel'])}",
                    (patch.get_x() + patch.get_width() / 2, patch.get_height()),
                    ha="center", va="bottom",
                    fontsize=8, rotation=90
                )

        plt.title(f"Top Slice Frequencies per Layer – {patient_id} ({metric})", fontsize=14)
        plt.xticks(rotation=75, ha="right")
        plt.xlabel("Layer")
        plt.ylabel("Count")
        plt.legend(title="Slice ID")
        plt.tight_layout()

        fname = f"top_slices_channels_{metric}_{patient_id}.png"
        plt.savefig(os.path.join(save_dir, fname))
        plt.close()


def plot_layer_boxplots(summary, save_dir="sinem/plots"):
    """Boxplots of MAD & MSE values per layer (bigger, cleaner)."""
    df = summary_to_dataframe(summary)
    os.makedirs(save_dir, exist_ok=True)

    for metric in ["MAD", "MSE"]:
        plt.figure(figsize=(16, 7))
        sns.boxplot(data=df[df["Metric"] == metric], x="Layer", y="Value")
        plt.title(f"{metric} Distribution per Layer", fontsize=14)
        plt.xticks(rotation=60, ha="right")
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{metric}_boxplot_per_layer.png"))
        plt.close()


def plot_slice_heatmap(summary, metric="MAD", save_dir="sinem/plots"):
    """Heatmap of average MAD/MSE per patient × slice (averaged over layers)."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric].groupby(["Patient", "Slice"])["Value"].mean().unstack()

    plt.figure(figsize=(16, 8))
    sns.heatmap(df_metric, cmap="viridis", annot=False, cbar_kws={'label': f"Avg {metric}"})
    plt.title(f"Average {metric} per Patient × Slice", fontsize=14)
    plt.xlabel("Slice")
    plt.ylabel("Patient")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_heatmap_patient_slice.png"))
    plt.close()


def plot_mad_mse_scatter(summary, save_dir="sinem/plots"):
    """Scatter plot of MAD vs MSE colored by layer."""
    df = summary_to_dataframe(summary)
    df_mad = df[df["Metric"] == "MAD"].copy()
    df_mse = df[df["Metric"] == "MSE"].copy()

    merged = pd.merge(
        df_mad, df_mse,
        on=["Layer", "Patient", "Slice", "Channel"],
        suffixes=("_mad", "_mse")
    )

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        merged["Value_mad"], merged["Value_mse"],
        c=pd.factorize(merged["Layer"])[0], alpha=0.5, cmap="viridis"
    )
    plt.xlabel("MAD")
    plt.ylabel("MSE")
    plt.title("MAD vs MSE (colored by Layer)", fontsize=14)
    plt.colorbar(scatter, label="Layer Index")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "mad_vs_mse_scatter.png"))
    plt.close()


def plot_patient_slice_profiles(summary, metric="MAD", save_dir="sinem/plots"):
    """Line plot of MAD/MSE values across slices for each patient (averaged across layers)."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric].groupby(["Patient", "Slice"])["Value"].mean().reset_index()

    plt.figure(figsize=(14, 7))
    sns.lineplot(data=df_metric, x="Slice", y="Value", hue="Patient", marker="o")
    plt.title(f"{metric} across slices per patient", fontsize=14)
    plt.xlabel("Slice")
    plt.ylabel(metric)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_slice_profile_per_patient.png"))
    plt.close()


def plot_layer_slice_heatmap(summary, metric="MAD", save_dir="sinem/plots"):
    """Heatmap of average MAD/MSE per layer × slice (averaged across patients)."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric].groupby(["Layer", "Slice"])["Value"].mean().unstack()

    plt.figure(figsize=(16, 10))
    sns.heatmap(df_metric, cmap="magma", annot=False, cbar_kws={'label': f"Avg {metric}"})
    plt.title(f"Average {metric} per Layer × Slice", fontsize=14)
    plt.xlabel("Slice")
    plt.ylabel("Layer")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_heatmap_layer_slice.png"))
    plt.close()


def plot_channel_frequency(summary, metric="MAD", save_dir="sinem/plots"):
    """Histogram of how often each channel appears in top-3 for given metric."""
    df = summary_to_dataframe(summary)
    df_metric = df[df["Metric"] == metric]

    counts = df_metric["Channel"].value_counts().sort_index()

    plt.figure(figsize=(14, 6))
    sns.barplot(x=counts.index, y=counts.values, color="skyblue", edgecolor="black")
    plt.title(f"Top-3 Channel Frequency ({metric})", fontsize=14)
    plt.xlabel("Channel")
    plt.ylabel("Frequency in Top-3")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{metric}_channel_frequency.png"))
    plt.close()

