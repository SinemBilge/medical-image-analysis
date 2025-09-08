import nibabel as nib
import scipy.ndimage
import numpy as np

PATIENT_IDS = [
    "patient001_frame01", "patient001_frame12",
    "patient002_frame01", "patient002_frame12",
    "patient003_frame01", "patient003_frame15",
    "patient004_frame01", "patient004_frame15",
    "patient005_frame01", "patient005_frame13",
]

def load_patients(patient_ids=PATIENT_IDS,
                  image_dir="../Task500_ACDC/imagesTr",
                  label_dir="../Task500_ACDC/labelsTr"):
    all_images, all_labels = {}, {}
    for pid in patient_ids:
        img_path = f"{image_dir}/{pid}_0000.nii.gz"
        gt_path  = f"{label_dir}/{pid}.nii.gz"

        img = nib.load(img_path).get_fdata()
        gt  = nib.load(gt_path).get_fdata()

        all_images[pid] = img
        all_labels[pid] = gt

        print(f"✓ Loaded: {pid} → image shape = {img.shape}, label shape = {gt.shape}")
    return all_images, all_labels


def transpose_volumes(all_images, all_labels):
    for pid in all_images:
        all_images[pid] = all_images[pid].transpose(2, 0, 1)
        all_labels[pid] = all_labels[pid].transpose(2, 0, 1)
        print(f"✓ Transposed: {pid} → new image shape = {all_images[pid].shape}, label shape = {all_labels[pid].shape}")
    return all_images, all_labels


def resize_volume(volume, new_shape=(256, 256), is_label=False):
    resized = []
    for slice_ in volume:
        zoom_factors = (
            new_shape[0] / slice_.shape[0],
            new_shape[1] / slice_.shape[1]
        )
        order = 0 if is_label else 1  # nearest for labels, bilinear for images
        resized.append(scipy.ndimage.zoom(slice_, zoom_factors, order=order))
    out = np.stack(resized)
    if is_label:
        out = out.astype(np.int16)
    else:
        out = out.astype(np.float32)
    return out


def preprocess_volumes(all_images, all_labels, new_shape=(256, 256)):
    all_images = {pid: resize_volume(img, new_shape, is_label=False) for pid, img in all_images.items()}
    all_labels = {pid: resize_volume(lbl, new_shape, is_label=True) for pid, lbl in all_labels.items()}
    return all_images, all_labels
