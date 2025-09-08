import numpy as np
import matplotlib.pyplot as plt
from MonaiUNets import UNetHeart_8_4_4
from src.utils import CLASS_MAP, dice_score

PATIENT_IDS = [
    "patient001_frame01", "patient001_frame12",
    "patient002_frame01", "patient002_frame12",
    "patient003_frame01", "patient003_frame15",
    "patient004_frame01", "patient004_frame15",
    "patient005_frame01", "patient005_frame13",
]


def load_model():
    return UNetHeart_8_4_4()

def run_inference(model, all_images, patient_ids=PATIENT_IDS):
    all_predictions = {}
    all_segmentations = {}

    for pid in patient_ids:
        print(f"\n▶ Performing segmentation for: {pid}")
        img = all_images[pid]
        pred = model.predict(img)
        result = np.argmax(pred, axis=1)

        all_predictions[pid] = pred
        all_segmentations[pid] = result

    return all_predictions, all_segmentations


def visualize_results(all_images, all_labels, all_segmentations, patient_ids, class_name="LV"):
    class_id = CLASS_MAP[class_name]

    for pid in patient_ids:
        img = all_images[pid]
        gt = all_labels[pid]
        result = all_segmentations[pid]

        for slide in range(img.shape[0]):
            score = dice_score(result[slide], gt[slide], class_name="LV")

            plt.figure(figsize=(12, 4))

            # Original
            plt.subplot(1, 3, 1)
            plt.imshow(img[slide], cmap="gray")
            plt.title("Original")
            plt.axis("off")

            # Ground truth
            plt.subplot(1, 3, 2)
            plt.imshow(img[slide], cmap="gray")
            plt.imshow(gt[slide], alpha=0.4)
            plt.title("Ground Truth")
            plt.axis("off")

            # Segmentation
            plt.subplot(1, 3, 3)
            plt.imshow(img[slide], cmap="gray")
            plt.imshow(result[slide], alpha=0.4)
            plt.title("Segmentation")
            plt.axis("off")

            plt.suptitle(f"{pid} – Slice {slide} – Dice ({class_name}): {score:.4f}", fontsize=10)
            plt.tight_layout()
            plt.show()
