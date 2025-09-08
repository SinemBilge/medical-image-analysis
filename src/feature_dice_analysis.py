import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.utils import dice_score
from src.hooks import extract_features_all, target_layers


def compute_feature_dice_correlation(
    model, 
    images, 
    bright_images, 
    labels, 
    patient_ids, 
    target_layers, 
    class_name="LV"
):

    features_orig, features_bright = extract_features_all(
        model, patient_ids, images, bright_images, target_layers
    )

    rows = []

    for pid in patient_ids:
        gt_vol = labels[pid]
        seg_orig = np.argmax(model.predict(images[pid]), axis=1)
        seg_bright = np.argmax(model.predict(bright_images[pid]), axis=1)

        for slice_idx in range(seg_orig.shape[0]):
            if not np.any(gt_vol[slice_idx] == 3):  # skip if no LV in GT
                continue

            dice_orig = dice_score(seg_orig[slice_idx], gt_vol[slice_idx], class_name)
            dice_bright = dice_score(seg_bright[slice_idx], gt_vol[slice_idx], class_name)
            dice_delta = dice_bright - dice_orig

            for layer in target_layers:
                fmap_orig = features_orig[pid][slice_idx][layer]  
                fmap_bright = features_bright[pid][slice_idx][layer]

                # Channel-wise mean activation
                mean_orig = fmap_orig.mean(axis=(2, 3)).squeeze()
                mean_bright = fmap_bright.mean(axis=(2, 3)).squeeze()

                # Δactivation per channel
                delta = mean_bright - mean_orig

                for ch, val in enumerate(delta):
                    rows.append({
                        "patient": pid,
                        "slice": slice_idx,
                        "layer": layer,
                        "channel": ch,
                        "dice_orig": dice_orig,
                        "dice_bright": dice_bright,
                        "dice_delta": dice_delta,
                        "feature_delta": float(val)
                    })

    return pd.DataFrame(rows)


def plot_feature_dice_results(df):
    """
    Simple plots: scatter feature_delta vs dice_delta, per layer.
    """
    layers = df["layer"].unique()

    for layer in layers:
        plt.figure(figsize=(6, 4))
        subset = df[df["layer"] == layer]
        plt.scatter(subset["feature_delta"], subset["dice_delta"], alpha=0.5)
        plt.xlabel("ΔFeature activation (channel mean)")
        plt.ylabel("ΔDice (bright - orig)")
        plt.title(f"Layer: {layer}")
        plt.grid(True)
        plt.show()
        
        
        
def export_feature_dice_results(df, filepath="feature_dice_results.csv"):
    if filepath.endswith(".csv"):
        df.to_csv(filepath, index=False)
    elif filepath.endswith(".xlsx"):
        df.to_excel(filepath, index=False)
    else:
        raise ValueError("Filepath must end with .csv or .xlsx")

    print(f"Results saved to {filepath}")
    
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_feature_dice_relationship(df, topk=3, plot=True):
    """
    Analyze how feature map changes relate to Dice score changes.
    """
    results = []

    for (layer, channel), group in df.groupby(["layer", "channel"]):
        if len(group) < 3:
            continue
        corr = np.corrcoef(group["feature_delta"], group["dice_delta"])[0, 1]
        results.append({
            "layer": layer,
            "channel": channel,
            "corr": corr
        })
    
    channel_scores = pd.DataFrame(results)
    if channel_scores.empty:
        raise ValueError("No valid correlations computed. Check df input.")

    layer_scores = (
        channel_scores.groupby("layer")["corr"]
        .mean()
        .reset_index()
        .sort_values("corr", ascending=False)
    )

    top_channels = (
        channel_scores.sort_values("corr", ascending=False)
        .groupby("layer")
        .head(topk)
        .reset_index(drop=True)
    )

    if plot:
        pivot = channel_scores.pivot(index="layer", columns="channel", values="corr")
        plt.figure(figsize=(12, 6))
        sns.heatmap(pivot, cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation between ΔFeature and ΔDice per Channel", fontsize=14)
        plt.xlabel("Channel")
        plt.ylabel("Layer")
        plt.tight_layout()
        plt.show()

    return channel_scores, layer_scores, top_channels
