import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
from src.model import PATIENT_IDS

target_layers = [
    "0.conv.unit0.adn.N",
    "0.conv.unit1.adn.N",
    "0.conv.unit2.adn.N",
    "0.conv.unit3.adn.N",
    "1.submodule.0.conv.unit0.adn.N",
    "1.submodule.0.conv.unit1.adn.N",
    "1.submodule.0.conv.unit2.adn.N",
    "1.submodule.0.conv.unit3.adn.N",
    "1.submodule.1.submodule.0.conv.unit0.adn.N",
    "1.submodule.1.submodule.0.conv.unit1.adn.N",
    "1.submodule.1.submodule.0.conv.unit2.adn.N",
    "1.submodule.1.submodule.0.conv.unit3.adn.N",
    "1.submodule.1.submodule.1.submodule.0.conv.unit0.adn.N",
    "1.submodule.1.submodule.1.submodule.0.conv.unit1.adn.N",
    "1.submodule.1.submodule.1.submodule.0.conv.unit2.adn.N",
    "1.submodule.1.submodule.1.submodule.0.conv.unit3.adn.N",
    "1.submodule.1.submodule.2.0.adn.N",
    "1.submodule.1.submodule.2.1.conv.unit0.adn.N",
    "1.submodule.2.0.adn.N",
    "1.submodule.2.1.conv.unit0.adn.N",
    "2.0.adn.N"
]

def extract_features_all(model, patient_ids, img_dict, img_bright_dict, target_layers, num_slices=10):
    """
    Extract feature maps per slice, per layer for both original and brightness-adjusted inputs.
    Returns two dicts: {pid: [ {layer: fmap}, ... per slice ]}
    """
    features_orig_dict = {}
    features_bright_dict = {}

    layer_map = dict(model.model.model.named_modules())

    def make_hook(storage, layer_name):
        def hook(_, __, output):
            storage[layer_name] = output.detach().cpu().numpy()
        return hook

    for pid in patient_ids:
        features_orig = [dict() for _ in range(num_slices)]
        features_bright = [dict() for _ in range(num_slices)]

        img = img_dict[pid]
        img_bright = img_bright_dict[pid]

        for i in range(num_slices):
            # --- Original input ---
            handles = []
            for name in target_layers:
                module = layer_map[name]
                handles.append(module.register_forward_hook(make_hook(features_orig[i], name)))
            _ = model.predict(img[i][None])
            for h in handles: h.remove()

            # --- Brightness-adjusted input ---
            handles = []
            for name in target_layers:
                module = layer_map[name]
                handles.append(module.register_forward_hook(make_hook(features_bright[i], name)))
            _ = model.predict(img_bright[i][None])
            for h in handles: h.remove()

        features_orig_dict[pid] = features_orig
        features_bright_dict[pid] = features_bright

    return features_orig_dict, features_bright_dict


def analyze_features_all(features_orig_dict, features_bright_dict, target_layers, patient_ids, all_labels, class_name="LV", out_file="brightness_analysis_results.txt"):
    """
    Compute MAD, MSE, SSIM for every patient × slice × layer × channel,
    but only for slices where the chosen class (e.g. LV) exists.
    """
    from src.utils import CLASS_MAP

    class_id = CLASS_MAP[class_name]

    with open(out_file, "w") as f:
        for pid in patient_ids:
            features_orig = features_orig_dict[pid]
            features_bright = features_bright_dict[pid]
            gt = all_labels[pid]

            for i in range(len(features_orig)):
                # Skip slices without the class
                if not np.any(gt[i] == class_id):
                    continue

                f.write(f"\n######## Patient: {pid} | Slice {i} ########\n")

                for layer_name in target_layers:
                    fmap_orig   = features_orig[i][layer_name]
                    fmap_bright = features_bright[i][layer_name]

                    num_channels = fmap_orig.shape[1]
                    f.write(f"=== Patient: {pid} | Layer: {layer_name} | Channels: {num_channels} ===\n")

                    for ch in range(num_channels):
                        orig   = fmap_orig[0, ch]
                        bright = fmap_bright[0, ch]

                        orig_n   = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
                        bright_n = (bright - bright.min()) / (bright.max() - bright.min() + 1e-8)

                        mad  = np.mean(np.abs(orig - bright))
                        mse  = np.mean((orig - bright) ** 2)
                        ssim_val = ssim(orig_n, bright_n, data_range=1)

                        line = (
                            f"Slice {i} | {layer_name} | Ch {ch:02d} → "
                            f"MAD={mad:.8f}, MSE={mse:.8f}, SSIM={ssim_val:.8f}"
                        )
                        f.write(line + "\n")
