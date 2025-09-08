import numpy as np
import matplotlib.pyplot as plt
from src.utils import CLASS_MAP, dice_score
import re
from collections import defaultdict

def compute_dice_per_slice(all_segmentations, all_labels, patient_ids, class_name="LV"):
    """
    Compute Dice scores per slice for each patient.
    Returns: dict {patient_id: [dice_per_slice]}
    """
    class_id = CLASS_MAP[class_name]
    results = {}

    for pid in patient_ids:
        seg = all_segmentations[pid]
        gt = all_labels[pid]

        dice_scores = []
        for slide in range(seg.shape[0]):
            score = dice_score(seg[slide], gt[slide], class_name=class_name)
            dice_scores.append(score)

        results[pid] = dice_scores

    return results


def plot_dice_per_slice(dice_orig, dice_bright, patient_ids, class_name="LV"):
    """Line plot: Dice per slice for each patient under two conditions."""
    for pid in patient_ids:
        plt.figure(figsize=(5, 3))
        plt.plot(dice_orig[pid], "-o", label="Original")
        plt.plot(dice_bright[pid], "-x", label="Bright-Adjusted")
        plt.ylim(0, 1.05)
        plt.xlabel("Slice Index")
        plt.ylabel("Dice Score")
        plt.title(f"{pid} – Dice Score per Slice ({class_name})")
        plt.legend()
        plt.tight_layout()
        plt.show()


def plot_boxplot(dice_orig, dice_bright, class_name="LV"):
    """Boxplot comparing Original vs Brightness-Adjusted Dice distributions."""
    scores_orig = np.concatenate(list(dice_orig.values()))
    scores_bright = np.concatenate(list(dice_bright.values()))


    plt.figure(figsize=(5, 5))
    plt.boxplot([scores_orig, scores_bright], labels=["Original", "Bright-Adjusted"])
    plt.ylabel("Dice Score")
    plt.title(f"Dice Score Distribution – Original vs. Bright-Adjusted ({class_name})")
    plt.show()


def plot_histogram_and_cdf(dice_orig, dice_bright, class_name="LV"):
    """Histogram of ΔDice and CDF of Dice scores."""
    scores_orig = np.concatenate(list(dice_orig.values()))
    scores_bright = np.concatenate(list(dice_bright.values()))
    delta = scores_bright - scores_orig

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(delta, bins=20, color="purple", alpha=0.7)
    plt.xlabel("Δ Dice Score (Bright - Original)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Dice Δ")


    plt.subplot(1, 2, 2)
    for scores, label, color in [(scores_orig, "Original", "blue"),
                                 (scores_bright, "Bright-Adjusted", "orange")]:
        sorted_vals = np.sort(scores)
        cdf = np.arange(len(sorted_vals)) / float(len(sorted_vals))
        plt.plot(sorted_vals, cdf, label=label, color=color)
    plt.xlabel("Dice Score")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of Dice Scores ({class_name})")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_mean_delta_per_patient(dice_orig, dice_bright, patient_ids, class_name="LV"):
    """Barplot: mean ΔDice per patient."""
    deltas = []
    for pid in patient_ids:
        scores_o = np.array(dice_orig[pid])
        scores_b = np.array(dice_bright[pid])
        mean_delta = np.mean(scores_b - scores_o)
        deltas.append(mean_delta)

    plt.figure(figsize=(7, 4))
    plt.bar(patient_ids, deltas, color="green")
    plt.ylabel("Δ Dice Score (Bright - Original)")
    plt.title(f"Mean Dice Δ per Patient ({class_name})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def summarize_top_mad_mse(results_file, out_file="top3_mad_mse_per_layer.txt"):
    """
    Parse results text file and extract top MAD and MSE entries per layer.
    Writes results to out_file AND returns a summary dict for notebook use.
    """

    with open(results_file, "r") as file:
        lines = file.readlines()

    entries = []
    current_patient = None
    current_layer = None


    slice_header_re = re.compile(r"^######## Patient: (.+) \| Slice (\d+) ########$")
    layer_re = re.compile(r"^=== Patient: (.+) \| Layer: (.+) \| Channels: (\d+) ===$")
    line_re = re.compile(
        r"^Slice (\d+) \| (.+) \| Ch (\d+) → MAD=([\d\.]+), MSE=([\d\.]+), SSIM=([\d\.]+)$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = slice_header_re.match(line)
        if m:
            current_patient = m.group(1).strip()
            continue

        m = layer_re.match(line)
        if m:
            current_layer = m.group(2).strip()
            continue

        m = line_re.match(line)
        if m:
            slice_idx = int(m.group(1))
            layer_name = m.group(2).strip()
            ch = int(m.group(3))
            mad = float(m.group(4))
            mse = float(m.group(5))
            ssim = float(m.group(6))

            entries.append({
                "Patient": current_patient,
                "Slice": slice_idx,
                "Layer": layer_name,
                "Channel": ch,
                "MAD": mad,
                "MSE": mse,
                "SSIM": ssim
            })

    layer_to_entries = defaultdict(list)
    for e in entries:
        layer_to_entries[e["Layer"]].append(e)

    summary = {}

    with open(out_file, "w") as out:
        out.write("Top 3 MAD and MSE entries per layer (including ties):\n")

        for layer, items in layer_to_entries.items():
            # --- MAD ---
            sorted_by_mad = sorted(items, key=lambda x: -x["MAD"])
            top_mads = sorted(set([x["MAD"] for x in sorted_by_mad]), reverse=True)[:3]
            top_mad_entries = [e for e in sorted_by_mad if e["MAD"] in top_mads]

            # --- MSE ---
            sorted_by_mse = sorted(items, key=lambda x: -x["MSE"])
            top_mses = sorted(set([x["MSE"] for x in sorted_by_mse]), reverse=True)[:3]
            top_mse_entries = [e for e in sorted_by_mse if e["MSE"] in top_mses]

            summary[layer] = {
                "top_mads": top_mads,
                "mad_entries": top_mad_entries,
                "top_mses": top_mses,
                "mse_entries": top_mse_entries,
            }

            header = (
                f"\n=== Layer: {layer} "
                f"| Top MADs: {', '.join(f'{m:.4f}' for m in top_mads)} "
                f"| Top MSEs: {', '.join(f'{m:.4f}' for m in top_mses)} ==="
            )
            out.write(header + "\n")

            for e in top_mad_entries:
                out.write(
                    f'MAD → {e["Patient"]} | Slice {e["Slice"]} | Ch {e["Channel"]:02d} '
                    f'→ MAD={e["MAD"]:.4f}, MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                )
            for e in top_mse_entries:
                out.write(
                    f'MSE → {e["Patient"]} | Slice {e["Slice"]} | Ch {e["Channel"]:02d} '
                    f'→ MAD={e["MAD"]:.4f}, MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                )

    print(f"Parsed {len(entries)} entries from {len(set(e['Patient'] for e in entries))} patients across {len(layer_to_entries)} layers.")
    print("Patients (example):", list(set(e['Patient'] for e in entries))[:5])
    print("Layers (example):", list(layer_to_entries.keys())[:5])
    print(f"Results written to {out_file}")

    return summary

def summarize_top_mad_mse_per_slice(results_file, out_file="layer_slice_summary1.txt"):
    """
    Parse results text file and extract top MAD and MSE entries per layer
    PLUS top MAD/MSE per slice (0–9)
    """

    with open(results_file, "r") as file:
        lines = file.readlines()

    entries = []
    current_patient = None
    current_layer = None

    slice_header_re = re.compile(r"^######## Patient: (.+) \| Slice (\d+) ########$")
    layer_re = re.compile(r"^=== Patient: (.+) \| Layer: (.+) \| Channels: (\d+) ===$")
    line_re = re.compile(
        r"^Slice (\d+) \| (.+) \| Ch (\d+) → MAD=([\d\.]+), MSE=([\d\.]+), SSIM=([\d\.]+)$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = slice_header_re.match(line)
        if m:
            current_patient = m.group(1).strip()
            continue

        m = layer_re.match(line)
        if m:
            current_layer = m.group(2).strip()
            continue

        m = line_re.match(line)
        if m:
            slice_idx = int(m.group(1))
            layer_name = m.group(2).strip()
            ch = int(m.group(3))
            mad = float(m.group(4))
            mse = float(m.group(5))
            ssim = float(m.group(6))

            entries.append({
                "Patient": current_patient,
                "Slice": slice_idx,
                "Layer": layer_name,
                "Channel": ch,
                "MAD": mad,
                "MSE": mse,
                "SSIM": ssim
            })

    layer_to_entries = defaultdict(list)
    for e in entries:
        layer_to_entries[e["Layer"]].append(e)

    summary = {}

    with open(out_file, "w") as out:
        out.write("Top 3 MAD and MSE entries per layer (including ties):\n")

        for layer, items in layer_to_entries.items():
            sorted_by_mad = sorted(items, key=lambda x: -x["MAD"])
            top_mads = sorted(set([x["MAD"] for x in sorted_by_mad]), reverse=True)[:3]
            top_mad_entries = [e for e in sorted_by_mad if e["MAD"] in top_mads]

            sorted_by_mse = sorted(items, key=lambda x: -x["MSE"])
            top_mses = sorted(set([x["MSE"] for x in sorted_by_mse]), reverse=True)[:3]
            top_mse_entries = [e for e in sorted_by_mse if e["MSE"] in top_mses]

            summary[layer] = {
                "top_mads": top_mads,
                "mad_entries": top_mad_entries,
                "top_mses": top_mses,
                "mse_entries": top_mse_entries,
            }

            header = (
                f"\n=== Layer: {layer} "
                f"| Top MADs: {', '.join(f'{m:.4f}' for m in top_mads)} "
                f"| Top MSEs: {', '.join(f'{m:.4f}' for m in top_mses)} ==="
            )
            out.write(header + "\n")

            for e in top_mad_entries:
                out.write(
                    f'MAD → {e["Patient"]} | Slice {e["Slice"]} | Ch {e["Channel"]:02d} '
                    f'→ MAD={e["MAD"]:.4f}, MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                )
            for e in top_mse_entries:
                out.write(
                    f'MSE → {e["Patient"]} | Slice {e["Slice"]} | Ch {e["Channel"]:02d} '
                    f'→ MAD={e["MAD"]:.4f}, MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                )

            for slice_idx in range(10):  
                slice_items = [e for e in items if e["Slice"] == slice_idx]
                if not slice_items:
                    continue

                out.write(f"-- Slice {slice_idx} --\n")

                sorted_by_mad = sorted(slice_items, key=lambda x: -x["MAD"])
                top_mads_slice = sorted(set([x["MAD"] for x in sorted_by_mad]), reverse=True)[:3]
                for e in [e for e in sorted_by_mad if e["MAD"] in top_mads_slice]:
                    out.write(
                        f'MAD → {e["Patient"]} | Slice {slice_idx} | Ch {e["Channel"]:02d} '
                        f'→ MAD={e["MAD"]:.4f}, MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                    )

                sorted_by_mse = sorted(slice_items, key=lambda x: -x["MSE"])
                top_mses_slice = sorted(set([x["MSE"] for x in sorted_by_mse]), reverse=True)[:3]
                for e in [e for e in sorted_by_mse if e["MSE"] in top_mses_slice]:
                    out.write(
                        f'MSE → {e["Patient"]} | Slice {slice_idx} | Ch {e["Channel"]:02d} '
                        f'→ MAD={e["MAD"]:.4f}, MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                    )

    print(f"Results written to {out_file}")
    return summary


def summarize_top3_per_patient_slice(results_file, out_file="layer_patient_slice_summary_3.txt"):
    """
    Parse results text file and extract:
    - For each layer
      - For each patient
        - For each slice (0–9)
          - Top 3 MAD entries
          - Top 3 MSE entries
    """

    with open(results_file, "r") as file:
        lines = file.readlines()

    entries = []
    current_patient = None
    current_layer = None

    slice_header_re = re.compile(r"^######## Patient: (.+) \| Slice (\d+) ########$")
    layer_re = re.compile(r"^=== Patient: (.+) \| Layer: (.+) \| Channels: (\d+) ===$")
    line_re = re.compile(
        r"^Slice (\d+) \| (.+) \| Ch (\d+) → MAD=([\d\.]+), MSE=([\d\.]+), SSIM=([\d\.]+)$"
    )

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = slice_header_re.match(line)
        if m:
            current_patient = m.group(1).strip()
            continue

        m = layer_re.match(line)
        if m:
            current_layer = m.group(2).strip()
            continue

        m = line_re.match(line)
        if m:
            slice_idx = int(m.group(1))
            layer_name = m.group(2).strip()
            ch = int(m.group(3))
            mad = float(m.group(4))
            mse = float(m.group(5))
            ssim = float(m.group(6))

            entries.append({
                "Patient": current_patient,
                "Slice": slice_idx,
                "Layer": layer_name,
                "Channel": ch,
                "MAD": mad,
                "MSE": mse,
                "SSIM": ssim
            })

    layer_to_patient_entries = defaultdict(lambda: defaultdict(list))
    for e in entries:
        layer_to_patient_entries[e["Layer"]][e["Patient"]].append(e)

    summary = defaultdict(lambda: defaultdict(dict))

    with open(out_file, "w") as out:
        out.write("Top 3 MAD and MSE per patient, per slice, grouped by layer:\n")

        for layer, patient_dict in layer_to_patient_entries.items():
            out.write(f"\n=== Layer: {layer} ===\n")

            for patient, items in patient_dict.items():
                out.write(f"\nPatient: {patient}\n")

                for slice_idx in range(10):  # slices 0–9
                    slice_items = [e for e in items if e["Slice"] == slice_idx]
                    if not slice_items:
                        continue

                    out.write(f"-- Slice {slice_idx} --\n")

                    sorted_by_mad = sorted(slice_items, key=lambda x: -x["MAD"])
                    top_mads_slice = sorted(set([x["MAD"] for x in sorted_by_mad]), reverse=True)[:3]
                    mad_entries = [e for e in sorted_by_mad if e["MAD"] in top_mads_slice]

                    sorted_by_mse = sorted(slice_items, key=lambda x: -x["MSE"])
                    top_mses_slice = sorted(set([x["MSE"] for x in sorted_by_mse]), reverse=True)[:3]
                    mse_entries = [e for e in sorted_by_mse if e["MSE"] in top_mses_slice]

                    for e in mad_entries:
                        out.write(
                            f'   MAD → Ch {e["Channel"]:02d} → MAD={e["MAD"]:.4f}, '
                            f'MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                        )
                    for e in mse_entries:
                        out.write(
                            f'   MSE → Ch {e["Channel"]:02d} → MAD={e["MAD"]:.4f}, '
                            f'MSE={e["MSE"]:.4f}, SSIM={e["SSIM"]:.4f}\n'
                        )

   
                    summary[layer][patient][slice_idx] = {
                        "top_mad": mad_entries,
                        "top_mse": mse_entries,
                    }

    print(f"Results written to {out_file}")
    return summary



def visualize_top3_mad_featuremaps(summary, features_orig_dict, features_bright_dict, out_dir="top3_mad_featuremaps1"):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    def _to_np(x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def _safe(s):
        return str(s).replace(".", "_").replace("/", "_").replace(" ", "_")

    n_saved = 0
    for layer, per_patient in summary.items():
        for patient, per_slice in per_patient.items():
            for slice_idx, groups in per_slice.items():
                top_mad_entries = groups.get("top_mad", [])
                if not top_mad_entries:
                    continue

                for e in top_mad_entries:
                    ch   = int(e["Channel"])
                    mad  = float(e["MAD"])
                    mse  = float(e["MSE"])
                    ssim = float(e["SSIM"])

                    fmap_o = features_orig_dict[patient][slice_idx][layer][0, ch]
                    fmap_b = features_bright_dict[patient][slice_idx][layer][0, ch]

                    orig   = _to_np(fmap_o)
                    bright = _to_np(fmap_b)
                    diff   = np.abs(orig - bright)

                    vmin = float(min(orig.min(), bright.min()))
                    vmax = float(max(orig.max(), bright.max()))

                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    fig.suptitle(
                        f"{layer} – Ch {ch}, Slice {slice_idx}, {patient}\n"
                        f"MAD={mad:.4f} | MSE={mse:.4f} | SSIM={ssim:.4f}",
                        fontsize=10
                    )

                    axs[0].imshow(orig, cmap="viridis", vmin=vmin, vmax=vmax)
                    axs[0].set_title("Original"); axs[0].axis("off")

                    axs[1].imshow(bright, cmap="viridis", vmin=vmin, vmax=vmax)
                    axs[1].set_title("Brightness Adjusted"); axs[1].axis("off")

                    axs[2].imshow(diff, cmap="magma")
                    axs[2].set_title("|Orig−Bright|"); axs[2].axis("off")

                    plt.tight_layout()

                    out_path = os.path.join(
                        out_dir,
                        f"path_{_safe(patient)}_layer_{_safe(layer)}_ch{ch:02d}_slice{slice_idx:02d}.png"
                    )
                    plt.savefig(out_path, dpi=150)
                    plt.close(fig)
                    n_saved += 1
    print(f"Saved {n_saved} images to {out_dir}/")
