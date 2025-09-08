import os
import re
import pandas as pd
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

IMG_PATTERN = re.compile(
    r"(?P<patient>patient\d+_frame\d+)_slice(?P<slice>\d+)_dice(?P<dice>[\d.]+)\.(?:png|jpg|jpeg)$",
    re.IGNORECASE
)

LAYER_RE   = re.compile(r"^\s*===\s*Layer:\s*(.+?)\s*===\s*$")
PATIENT_RE = re.compile(r"^\s*Patient:\s*(patient\d+_frame\d+)\s*$")
PATH_RE    = re.compile(r"^\s*===\s*path:\s*(patient\d+_frame\d+)\s*$")
SLICE_RE_A = re.compile(r"^\s*--\s*Slice\s*(\d+)\s*--\s*$")
SLICE_RE_B = re.compile(r"\bSlice\s+(\d+)\b")
METRIC_RE  = re.compile(
    r"(?:MAD)\s*→\s*Ch\s*(\d+)\s*→\s*MAD\s*=\s*([0-9.]+)\s*,\s*MSE\s*=\s*([0-9.]+)\s*,\s*SSIM\s*=\s*([0-9.]+)",
    re.IGNORECASE
)

def gather_images(base_dir: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            m = IMG_PATTERN.match(f)
            if not m:
                continue
            d = m.groupdict()
            rows.append({
                "category": os.path.basename(root),
                "image_file": f,
                "rel_path": os.path.relpath(os.path.join(root, f), base_dir),
                "patient": d["patient"],
                "slice": int(d["slice"]),
                "dice": float(d["dice"]),
            })
    return pd.DataFrame(rows)

def parse_summary(summary_txt: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    layer = patient = None
    slc = None

    with open(summary_txt, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            ml = LAYER_RE.match(line)
            if ml:
                layer = ml.group(1).strip()
                continue

            mp = PATIENT_RE.match(line) or PATH_RE.match(line)
            if mp:
                patient = mp.group(1).strip()
                continue

            ms = SLICE_RE_A.match(line) or SLICE_RE_B.search(line)
            if ms:
                slc = int(ms.group(1))
                continue

            mm = METRIC_RE.search(line)
            if mm and layer and patient and (slc is not None):
                ch, mad, mse, ssim = mm.groups()
                rows.append({
                    "layer": layer,
                    "patient": patient,
                    "slice": slc,
                    "channel": int(ch),
                    "MAD": float(mad),
                    "MSE": float(mse),
                    "SSIM": float(ssim),
                })
    return pd.DataFrame(rows)

def match_results(results_dir: str, summary_file: str, output_csv: str) -> pd.DataFrame:
    imgs = gather_images(results_dir)
    summary = parse_summary(summary_file)

    if imgs.empty:
        raise ValueError(f"No images found in {results_dir}")
    if summary.empty:
        raise ValueError(f"No summary parsed from {summary_file}")

    merged = imgs.merge(summary, on=["patient", "slice"], how="left") \
                 .sort_values(["category", "patient", "slice", "layer", "channel"])

    merged.to_csv(output_csv, index=False)
    print(f"Matched {len(merged)} rows. Saved to {output_csv}")
    return merged


def heatmap_top_channels_by_frequency(
    df: pd.DataFrame,
    out_png: str = "heatmap_top_channels_by_frequency.png",
    *,
    filter_category: Optional[str] = None,
    filter_patient: Optional[str] = None,
    filter_slice: Optional[int] = None,
    topk_per_slice: Optional[int] = 3,   
    topN_per_layer: int = 6,           
    distinct_by: Tuple[str, ...] = ("patient", "slice")
):
    data = df.copy()

    if filter_category is not None:
        data = data[data["category"] == filter_category]
    if filter_patient is not None:
        data = data[data["patient"] == filter_patient]
    if filter_slice is not None:
        data = data[data["slice"] == filter_slice]

    required_cols = {"layer", "channel", "MAD"}
    if not required_cols.issubset(set(data.columns)):
        raise ValueError(f"DataFrame must contain {required_cols}")

    if topk_per_slice is not None:
        data = (data.sort_values(["layer","patient","slice","MAD"],
                                 ascending=[True,True,True,False])
                    .groupby(["layer","patient","slice"], as_index=False)
                    .head(topk_per_slice))

    if distinct_by:
        dedup_cols = ["layer", "channel"] + list(distinct_by)
        data = data.drop_duplicates(subset=dedup_cols)

    freq = (data.groupby(["layer", "channel"])
                .size()
                .reset_index(name="frequency"))

    if freq.empty:
        raise ValueError("No data after filtering; cannot build heatmap.")

    freq["channel"] = freq["channel"].astype(int)
    freq = freq.sort_values(["layer", "frequency", "channel"],
                            ascending=[True, False, True])

    top_rows = (freq.groupby("layer", as_index=False)
                     .head(topN_per_layer))

    layers = sorted(top_rows["layer"].unique(), key=lambda s: str(s))

    H = np.zeros((len(layers), topN_per_layer), dtype=float)  
    L = np.empty((len(layers), topN_per_layer), dtype=object)  
    L[:] = ""

    for i, layer in enumerate(layers):
        sub = top_rows[top_rows["layer"] == layer].reset_index(drop=True)
        for j in range(min(topN_per_layer, len(sub))):
            H[i, j] = float(sub.loc[j, "frequency"])
            L[i, j] = f"Ch{int(sub.loc[j, 'channel'])}"

    fig_h = max(4.0, 0.5 * len(layers))
    fig_w = max(15.0, 1.5 * topN_per_layer)  
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.cm.viridis.copy()
    cmap.set_under('white')  

    im = ax.imshow(H,
                   aspect="auto",
                   interpolation="nearest",
                   cmap=cmap,
                   vmin=1e-9)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency of Occurrence", rotation=270, labelpad=15)

    ax.set_title("Top channels per layer by frequency", fontsize=14)
    ax.set_ylabel("Layer")
    ax.set_xticks(np.arange(topN_per_layer))
    ax.set_xticklabels([f"Top {i+1}" for i in range(topN_per_layer)],
                       fontsize=9)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers, fontsize=14)

    vmax = H.max() if H.size else 1.0
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if L[i, j]:
                val = H[i, j] / (vmax if vmax > 0 else 1.0)
                ax.text(j, i, L[i, j],
                        ha="center", va="center",
                        fontsize=10,
                        fontweight="bold",
                        color=("white"))
                

    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to {out_png}")
