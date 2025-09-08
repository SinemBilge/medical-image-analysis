import numpy as np
import matplotlib.pyplot as plt
import os

CLASS_MAP = {
    "RV": 1,
    "MYO": 2,
    "LV": 3
}

def extract_mask(gt_volume, class_name):
    class_id = CLASS_MAP[class_name]
    mask = np.zeros_like(gt_volume, dtype=np.uint8)
    mask[gt_volume == class_id] = 1
    return mask

def visualize_class(all_images, all_labels, patient_ids, class_name="LV"):
    class_id = CLASS_MAP[class_name]

    for pid in patient_ids:
        img = all_images[pid]
        gt  = all_labels[pid]

        mask = extract_mask(gt, class_name)

        for slide in range(img.shape[0]):
            if np.any(mask[slide]):
                plt.figure()
                _ = plt.imshow(mask[slide])   # 👈 capture return value
                plt.title(f"{pid} – Slice {slide} ({class_name} Mask)")
                plt.axis("off")
                plt.show()

            
def adjust_class_intensity(all_images, all_labels, patient_ids, class_name="LV", scale=0.1):
    class_id = CLASS_MAP[class_name]
    results = {}
    for pid in patient_ids:
        img = all_images[pid]
        gt  = all_labels[pid]

        class_id = CLASS_MAP[class_name]
        style_adjusted_input = img.copy()
        indices = np.where(gt == class_id)
        style_adjusted_input[indices] *= scale

        results[pid] = style_adjusted_input
        print(f"▶ {pid} ({class_name}) scaled by {scale}")
    return results


def visualize_adjusted(all_labels, results, patient_ids, class_name="LV"):
    class_id = CLASS_MAP[class_name]

    for pid in patient_ids:
        adjusted = results[pid]
        gt = all_labels[pid]

        for slide in range(adjusted.shape[0]):
            if np.any(gt[slide] == class_id): 
                plt.imshow(adjusted[slide], cmap="gray")
                plt.title(f"{pid} – Slice {slide} ({class_name} Adjusted)")
                plt.axis("off")
                plt.show()
                
                
def dice_score(pred_mask, true_mask, class_name="LV"):
    class_id = CLASS_MAP[class_name]
    pred_bin = (pred_mask == class_id).astype(np.uint8)
    true_bin = (true_mask == class_id).astype(np.uint8)
    intersection = np.sum(pred_bin * true_bin)
    total = np.sum(pred_bin) + np.sum(true_bin)
    return 1.0 if total == 0 else 2.0 * intersection / total
            
def visualize_results_bright(all_images, all_labels, model, patient_ids, class_name="RV"):
    class_id = CLASS_MAP[class_name]
    adjusted_results = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.1)
    all_predictions_bright = {}
    all_segmentations_bright = {}

    for pid in patient_ids:
        style_adjusted_input = adjusted_results[pid]
        gt = all_labels[pid]

        pred_bright = model.predict(style_adjusted_input)  
        print(np.shape(pred_bright))

        result_bright = np.argmax(pred_bright, axis=1) 
        print(np.shape(result_bright))

        all_predictions_bright[pid] = pred_bright
        all_segmentations_bright[pid] = result_bright

        for slide in range(style_adjusted_input.shape[0]):
            if not np.any(gt[slide] == CLASS_MAP[class_name]):
                continue
                
            bright_score = dice_score(result_bright[slide], gt[slide], class_name=class_name)
            print(f"▶ {pid} – Slice {slide} – Dice: {bright_score:.4f}")

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.title("Brightness-Adjusted Input")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.imshow(gt[slide], alpha=0.4)
            plt.title("Ground Truth")
            plt.axis("off")
            print(np.unique(gt[slide]))

            plt.subplot(1, 3, 3)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.imshow(result_bright[slide], alpha=0.4)
            plt.title("Brightness-Adjusted Prediction")
            plt.axis("off")
            print(np.unique(result_bright))

            plt.suptitle(f"{pid} – Bright Slice {slide}", fontsize=10)
            plt.tight_layout()
            plt.show()

    return all_predictions_bright, all_segmentations_bright


def compare_rv_intensity(all_images, all_labels, patient_ids, class_name="RV"):
    class_id = CLASS_MAP[class_name]
    
    adjusted_01 = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.1)
    adjusted_02 = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.2)

    for pid in patient_ids:
        img01 = adjusted_01[pid]
        img02 = adjusted_02[pid]
        gt = all_labels[pid]

        rv_pixels_01 = []
        rv_pixels_02 = []

        print(f"\n▶ Patient {pid}")
        for slide in range(img01.shape[0]):
            mask = (gt[slide] == class_id)
            n_rv = mask.sum()
            if n_rv == 0:
                continue

            print(f"  Slice {slide}: {n_rv} RV pixels")

            rv_pixels_01.extend(img01[slide][mask].ravel())
            rv_pixels_02.extend(img02[slide][mask].ravel())

        if len(rv_pixels_01) == 0:
            print("  ⚠ No RV pixels found in any slice")
            continue

        plt.figure(figsize=(10,4))
        plt.hist(rv_pixels_01, bins=50, alpha=0.6, color="red", label="scale=0.1")
        plt.hist(rv_pixels_02, bins=50, alpha=0.6, color="blue", label="scale=0.2")
        plt.title(f"{pid} – All RV pixels (combined across slices)")
        plt.xlabel("Pixel intensity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.show()
        
def compare_rv_intensity_all(all_images, all_labels, patient_ids, class_name="RV"):
    class_id = CLASS_MAP[class_name]
    
    adjusted_01 = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.1)
    adjusted_02 = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.2)

    rv_pixels_01 = []
    rv_pixels_02 = []

    print("\n▶ Collecting RV pixels from all patients")
    for pid in patient_ids:
        img01 = adjusted_01[pid]
        img02 = adjusted_02[pid]
        gt = all_labels[pid]

        for slide in range(img01.shape[0]):
            mask = (gt[slide] == class_id)
            n_rv = mask.sum()
            if n_rv == 0:
                continue

            print(f"  {pid} – Slice {slide}: {n_rv} RV pixels")

            rv_pixels_01.extend(img01[slide][mask].ravel())
            rv_pixels_02.extend(img02[slide][mask].ravel())

    if len(rv_pixels_01) == 0:
        print("⚠ No RV pixels found in dataset")
        return

    plt.figure(figsize=(10,5))
    plt.hist(rv_pixels_01, bins=50, alpha=0.6, color="red", label="scale=0.1")
    plt.hist(rv_pixels_02, bins=50, alpha=0.6, color="blue", label="scale=0.2")
    plt.title("All RV pixels combined (all patients, all slices)")
    plt.xlabel("Pixel intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()

def summarize_rv_intensity(all_images, all_labels, patient_ids, class_name="RV"):
    class_id = CLASS_MAP[class_name]
    
    adj01 = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.1)
    adj02 = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.2)
    
    rv_vals01, rv_vals02 = [], []
    
    for pid in patient_ids:
        gt = all_labels[pid]
        img01 = adj01[pid]
        img02 = adj02[pid]
        
        for s in range(gt.shape[0]):
            mask = (gt[s] == class_id)
            if mask.sum() == 0:
                continue
            rv_vals01.extend(img01[s][mask].ravel())
            rv_vals02.extend(img02[s][mask].ravel())
    
    rv_vals01 = np.array(rv_vals01)
    rv_vals02 = np.array(rv_vals02)
    
    mean01, std01 = rv_vals01.mean(), rv_vals01.std()
    mean02, std02 = rv_vals02.mean(), rv_vals02.std()
    
    print(f"Scale 0.1 → mean={mean01:.2f}, std={std01:.2f}, n={len(rv_vals01)}")
    print(f"Scale 0.2 → mean={mean02:.2f}, std={std02:.2f}, n={len(rv_vals02)}")
    
    return (mean01, std01, len(rv_vals01)), (mean02, std02, len(rv_vals02))

def classify_case(gt_slice, pred_slice, overlap_thresh=0.3, lv_tiny_frac=0.1, extreme_frac=0.5, rv_frac_limit=0.5):
    cid_rv, cid_myo, cid_lv = CLASS_MAP["RV"], CLASS_MAP["MYO"], CLASS_MAP["LV"]

    pred_counts = {v: int((pred_slice == v).sum()) for v in np.unique(pred_slice)}
    gt_counts   = {v: int((gt_slice == v).sum()) for v in np.unique(gt_slice)}

    gt_lv_mask = (gt_slice == cid_lv)
    pred_lv_mask = (pred_slice == cid_lv)
    pred_myo_mask = (pred_slice == cid_myo)
    pred_rv_mask = (pred_slice == cid_rv)

    lv_as_myo = int(np.sum(gt_lv_mask & pred_myo_mask))  # GT LV mis → MYO
    gt_lv_total = np.sum(gt_lv_mask)
    pred_lv_total = np.sum(pred_lv_mask)
    pred_myo_total = np.sum(pred_myo_mask)
    pred_rv_total = np.sum(pred_rv_mask)

    lv_inside = np.sum(pred_lv_mask & gt_lv_mask)
    myo_inside = np.sum(pred_myo_mask & gt_lv_mask)
    lv_frac = lv_inside / max(1, pred_lv_total) if pred_lv_total > 0 else 0
    myo_frac = myo_inside / max(1, pred_myo_total) if pred_myo_total > 0 else 0

    lv_pred_frac = pred_lv_total / max(1, gt_lv_total)   # predicted LV vs GT LV
    frac_lv_as_myo = lv_as_myo / max(1, gt_lv_total)     # portion of GT LV predicted as MYO
    rv_frac = pred_rv_total / max(1, gt_lv_total)        # RV relative to GT LV

    foreground_sum = pred_lv_total + pred_myo_total + pred_rv_total
    MYO_INSIDE_THR = 0.70
    myo_mostly_inside = (pred_myo_total > 0) and (myo_inside / pred_myo_total >= MYO_INSIDE_THR)


    if foreground_sum < 35:
        case = "Empty Prediction"
        reason = "Pred mask has <35 foreground pixels"

    elif pred_lv_total > 0 and pred_myo_total == 0 and pred_rv_total == 0:
        case = "Pure LV"
        reason = "Only LV detected"

    elif pred_lv_total > 0 and pred_myo_total > 0:
        rv_size = pred_rv_total

        lv_fully_inside = (lv_inside == pred_lv_total and pred_lv_total > 0)

        myo_inside_ratio = myo_inside / pred_myo_total if pred_myo_total > 0 else 0
        
        if lv_fully_inside and myo_mostly_inside:
            case = "LV + MYO Mixed"
            reason = (f"LV fully inside and MYO mostly inside GT LV "
                      f"(rv={rv_size}), lv_total={pred_lv_total}, "
                      f"myo_total={pred_myo_total}, myo_inside_ratio={myo_inside_ratio:.2f}")
        else:
            case = "Error"
            reason = (f"LV and/or MYO extend outside GT LV region "
                      f"(lv_inside={lv_inside}/{pred_lv_total}, "
                      f"myo_inside={myo_inside}/{pred_myo_total}, "
                      f"myo_inside_ratio={myo_inside_ratio:.2f})")


    elif (frac_lv_as_myo >= extreme_frac and (rv_frac < rv_frac_limit)):
        lv_fully_inside = (lv_inside == pred_lv_total and pred_lv_total > 0)
        if not myo_mostly_inside and lv_fully_inside:
            case = "Error"
            reason = (f"Pred MYO mostly outside GT LV in extreme case "
                      f"(myo_inside={myo_inside}/{pred_myo_total}, ratio={myo_frac:.2f})")
        else:
            case = "Extreme LV→MYO Misclassification"
            reason = f"{lv_as_myo} LV pixels misclassified as MYO ({frac_lv_as_myo:.2f} of GT LV)"

    elif (pred_lv_total < 10) and (pred_myo_total <= 20) and (pred_rv_total > 100):
        case = "Contains RV"
        reason = "Only RV detected"
    elif (frac_lv_as_myo >= lv_tiny_frac) and (pred_rv_total < 200) :
        if not myo_mostly_inside:
            case = "Error"
            reason = (f"Pred MYO mostly outside GT LV "
                      f"(myo_inside={myo_inside}/{pred_myo_total}, ratio={myo_frac:.2f})")
        else:
            case = "LV→MYO Misclassification"
            reason = f"{lv_as_myo} LV pixels misclassified as MYO"

    elif (pred_lv_total > 0) and (pred_myo_total > 0) and (pred_rv_total > 20):
        if (lv_inside == pred_lv_total and myo_inside == pred_myo_total):
            case = "Every Class"
            reason = "Every Class is visible (all LV+MYO inside GT LV)"
        else:
            case = "Error"
            reason = "Every Class predicted but LV/MYO not fully inside GT LV"

    else:
        case = "Error"
        reason = "Doesn't fit rules"

    return case, reason, gt_counts, pred_counts, lv_as_myo, frac_lv_as_myo


def visualize_and_classify_bright(all_images, all_labels, model, patient_ids, class_name="LV"):
    class_id = CLASS_MAP[class_name]
    adjusted_results = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.1)

    all_predictions_bright = {}
    all_segmentations_bright = {}

    for pid in patient_ids:
        style_adjusted_input = adjusted_results[pid]
        gt = all_labels[pid]

        pred_bright = model.predict(style_adjusted_input)
        result_bright = np.argmax(pred_bright, axis=1)

        all_predictions_bright[pid] = pred_bright
        all_segmentations_bright[pid] = result_bright

        for slide in range(style_adjusted_input.shape[0]):
            if not np.any(gt[slide] == CLASS_MAP[class_name]):
                continue

            bright_score = dice_score(result_bright[slide], gt[slide], class_name=class_name)
            case, reason, gt_counts, pred_counts, lv_as_myo, frac_lv_as_myo = classify_case(gt[slide], result_bright[slide])

            print(f"▶ {pid} – Slice {slide} → Dice: {bright_score:.4f} | Group: {case} | {reason}")
            print(f"GT unique: {gt_counts}")
            print(f"Pred unique: {pred_counts}")

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{pid} – Bright Slice {slide}\n"
                         f"Dice={bright_score:.4f} | Group: {case} | {reason}\n"
                         f"GT unique: {gt_counts} | Pred unique: {pred_counts}\n"
                         f"LV→MYO frac={frac_lv_as_myo:.2f}", fontsize=10)

            plt.subplot(1, 3, 1)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.title("Brightness-Adjusted Input")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.imshow(gt[slide], alpha=0.4)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.imshow(result_bright[slide], alpha=0.4)
            plt.title("Brightness-Adjusted Prediction")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    return all_predictions_bright, all_segmentations_bright


def save_classified_results(all_images, all_labels, model, patient_ids, class_name="LV", output_dir="results"):
    class_id = CLASS_MAP[class_name]
    os.makedirs(output_dir, exist_ok=True)

    adjusted_results = adjust_class_intensity(all_images, all_labels, patient_ids, class_name=class_name, scale=0.1)

    for pid in patient_ids:
        style_adjusted_input = adjusted_results[pid]
        gt = all_labels[pid]
        pred_bright = model.predict(style_adjusted_input)
        result_bright = np.argmax(pred_bright, axis=1)

        for slide in range(style_adjusted_input.shape[0]):
            if not np.any(gt[slide] == CLASS_MAP[class_name]):
                continue

            bright_score = dice_score(result_bright[slide], gt[slide], class_name=class_name)
            case, reason, gt_counts, pred_counts, rv_as_myo, frac_rv_as_myo, frac_rv_as_lv = \
    classify_case_rv_v1(gt[s], result_bright[s])

            case_dir = os.path.join(output_dir, case.replace(" ", "_"))
            os.makedirs(case_dir, exist_ok=True)

            filename = f"{pid}_slice{slide}_dice{bright_score:.4f}.png"
            filepath = os.path.join(case_dir, filename)

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{pid} – Bright Slice {slide}\n"
                         f"Dice={bright_score:.4f} | Group: {case} | {reason}\n"
                         f"GT: {gt_counts} | Pred: {pred_counts}\n"
                         f"LV→MYO frac={frac_lv_as_myo:.2f}", fontsize=10)

            plt.subplot(1, 3, 1)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.title("Brightness-Adjusted Input")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.imshow(gt[slide], alpha=0.4)
            plt.title("Ground Truth")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(style_adjusted_input[slide], cmap="gray")
            plt.imshow(result_bright[slide], alpha=0.4)
            plt.title("Prediction")
            plt.axis("off")

            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()

            print(f"Saved → {filepath}")

            
            
def classify_case_rv_v1(
    gt_slice,
    pred_slice,
    *,
    rv_tiny_frac=0.10,      # normal misclass threshold (fraction of GT-RV)
    extreme_frac=0.50,      # extreme misclass threshold
    rv_inside_thr=0.4,     # how much of predicted RV must lie inside GT-RV
    lv_frac_limit=0.50      # if LV is huge vs GT-RV, don't blame MYO only
):
    cid_rv, cid_myo, cid_lv = CLASS_MAP["RV"], CLASS_MAP["MYO"], CLASS_MAP["LV"]

    # counts
    pred_counts = {v: int((pred_slice == v).sum()) for v in np.unique(pred_slice)}
    gt_counts   = {v: int((gt_slice == v).sum())  for v in np.unique(gt_slice)}

    # masks
    gt_rv_mask    = (gt_slice   == cid_rv)
    pred_rv_mask  = (pred_slice == cid_rv)
    pred_myo_mask = (pred_slice == cid_myo)
    pred_lv_mask  = (pred_slice == cid_lv)

    # totals
    gt_rv_total    = int(gt_rv_mask.sum())
    pred_rv_total  = int(pred_rv_mask.sum())
    pred_myo_total = int(pred_myo_mask.sum())
    pred_lv_total  = int(pred_lv_mask.sum())

    # overlaps & fractions
    rv_inside        = int((pred_rv_mask & gt_rv_mask).sum())
    rv_inside_frac   = (rv_inside / pred_rv_total) if pred_rv_total > 0 else 0.0

    rv_as_myo        = int((gt_rv_mask & pred_myo_mask).sum())   # GT RV → MYO
    rv_as_lv         = int((gt_rv_mask & pred_lv_mask).sum())    # GT RV → LV
    frac_rv_as_myo   = (rv_as_myo / gt_rv_total) if gt_rv_total > 0 else 0.0
    frac_rv_as_lv    = (rv_as_lv  / gt_rv_total) if gt_rv_total > 0 else 0.0

    lv_frac          = (pred_lv_total / max(1, gt_rv_total))
    foreground_sum   = pred_rv_total + pred_myo_total + pred_lv_total
    

    if foreground_sum < 35:
        pass 
    elif pred_rv_total > 0 and pred_myo_total == 0 and pred_lv_total == 0:
        case, reason = "Only RV", "Only RV detected"

    elif pred_rv_total >= 10 and rv_inside_frac >= rv_inside_thr:
        case = "RV + Others"
        reason = (f"Pred RV mostly inside GT-RV (inside={rv_inside}/{pred_rv_total}, "
                  f"{rv_inside_frac:.2f}); LV={pred_lv_total}, MYO={pred_myo_total}")
    elif frac_rv_as_myo >= extreme_frac:
        case = "Extreme RV→MYO Misclassification"
        reason = f"{rv_as_myo} RV pixels misclassified as MYO ({frac_rv_as_myo:.2f} of GT-RV)"
    elif frac_rv_as_myo >= 0.04:
        case = "RV→MYO Misclassification"
        reason = f"{rv_as_myo} RV pixels misclassified as MYO ({frac_rv_as_myo:.2f} of GT-RV)"
    elif frac_rv_as_lv >= extreme_frac:
        case = "Extreme RV→LV Misclassification"
        reason = f"{rv_as_lv} RV pixels misclassified as LV ({frac_rv_as_lv:.2f} of GT-RV)"
    elif foreground_sum < 35:
        case, reason = "Empty Prediction", "Pred mask has <35 foreground pixels"

    elif frac_rv_as_myo < 0.04 and pred_rv_total > 0 and rv_inside_frac < rv_inside_thr:
        case = "Error"
        reason = (f"Pred RV mostly outside GT-RV (inside={rv_inside}/{pred_rv_total}, "
                  f"{rv_inside_frac:.2f})")

    elif (pred_rv_total > 20) and (pred_lv_total > 50) and (pred_myo_total > 50):
        if rv_inside_frac > rv_inside_thr:
            case, reason = "Every Class", "Every class visible; RV fully inside GT-RV"
        else:
            case, reason = "Error", "Every class predicted but RV not fully/mostly inside GT-RV"
    else:
        case, reason = "Error", "Doesn't fit rules"

    return case, reason, gt_counts, pred_counts, rv_as_myo, frac_rv_as_myo, frac_rv_as_lv


def frac_rv_as_myo_range(all_gt, all_pred, patient_ids=None, decimals=2):
    if patient_ids is None:
        patient_ids = sorted(set(all_gt.keys()) & set(all_pred.keys()))

    fracs = []
    for pid in patient_ids:
        gt = all_gt[pid]
        pr = all_pred[pid]
        for s in range(gt.shape[0]):
            *_, frac_rv_as_myo, _ = classify_case_rv_v1(gt[s], pr[s])
            fracs.append(frac_rv_as_myo)

    if not fracs:
        print("[N/A - N/A]")
        return None, None

    lo, hi = min(fracs), max(fracs)
    print(f"[{lo:.2f} - {hi:.2f}]")
    return lo, hi


            
def visualize_and_classify_bright_rv(all_images, all_labels, model, patient_ids, scale=0.1):
    adjusted_results = adjust_class_intensity(all_images, all_labels, patient_ids,
                                              class_name="RV", scale=scale)

    all_predictions_bright = {}
    all_segmentations_bright = {}

    for pid in patient_ids:
        x  = adjusted_results[pid]
        gt = all_labels[pid]

        pred_bright   = model.predict(x)              # (S, K, H, W)
        result_bright = np.argmax(pred_bright, axis=1)# (S, H, W)

        all_predictions_bright[pid]   = pred_bright
        all_segmentations_bright[pid] = result_bright

        for s in range(x.shape[0]):
            if not np.any(gt[s] == CLASS_MAP["RV"]):
                continue

            bright_score = dice_score(result_bright[s], gt[s], class_name="RV")
            case, reason, gt_counts, pred_counts, rv_as_myo, frac_rv_as_myo, frac_rv_as_lv= \
                classify_case_rv_v1(gt[s], result_bright[s])

            print(f"▶ {pid} – Slice {s} → Dice: {bright_score:.4f} | Group: {case} | {reason}")
            print(f"GT unique: {gt_counts}")
            print(f"Pred unique: {pred_counts}")

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{pid} – Bright Slice {s}\n"
                         f"Dice={bright_score:.4f} | Group: {case} | {reason}\n"
                         f"GT: {gt_counts} | Pred: {pred_counts}\n"
                         f"RV→MYO frac={frac_rv_as_myo:.2f} | LV→MYO frac={frac_rv_as_lv:.2f}",
                fontsize=10)

            plt.subplot(1, 3, 1)
            plt.imshow(x[s], cmap="gray")
            plt.title("Brightness-Adjusted Input"); plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(x[s], cmap="gray")
            plt.imshow(gt[s], alpha=0.4)
            plt.title("Ground Truth"); plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(x[s], cmap="gray")
            plt.imshow(result_bright[s], alpha=0.4)
            plt.title("Brightness-Adjusted Prediction"); plt.axis("off")

            plt.tight_layout(); plt.show()

    return all_predictions_bright, all_segmentations_bright
         
            
def save_classified_results_rv(all_images, all_labels, model, patient_ids,
                               output_dir="results_rv", scale=0.1):
    os.makedirs(output_dir, exist_ok=True)
    adjusted_results = adjust_class_intensity(all_images, all_labels, patient_ids,
                                              class_name="RV", scale=scale)

    for pid in patient_ids:
        x  = adjusted_results[pid]
        gt = all_labels[pid]
        pred_bright   = model.predict(x)
        result_bright = np.argmax(pred_bright, axis=1)

        for s in range(x.shape[0]):
            if not np.any(gt[s] == CLASS_MAP["RV"]):
                continue

            bright_score = dice_score(result_bright[s], gt[s], class_name="RV")
            case, reason, gt_counts, pred_counts, rv_as_myo, frac_rv_as_myo, frac_rv_as_lv = \
                classify_case_rv_v1(gt[s], result_bright[s])

            case_dir = os.path.join(output_dir, case.replace(" ", "_"))
            os.makedirs(case_dir, exist_ok=True)
            filepath = os.path.join(case_dir, f"{pid}_slice{s}_dice{bright_score:.4f}.png")

            plt.figure(figsize=(12, 4))
            plt.suptitle(f"{pid} – Bright Slice {s}\n"
                         f"Dice={bright_score:.4f} | Group: {case} | {reason}\n"
                         f"GT: {gt_counts} | Pred: {pred_counts}\n"
                         f"RV→MYO frac={frac_rv_as_myo:.2f} | LV→MYO frac={frac_rv_as_lv:.2f}",
                fontsize=10)
                         
            plt.subplot(1, 3, 1); plt.imshow(x[s], cmap="gray")
            plt.title("Brightness-Adjusted Input"); plt.axis("off")

            plt.subplot(1, 3, 2); plt.imshow(x[s], cmap="gray"); plt.imshow(gt[s], alpha=0.4)
            plt.title("Ground Truth"); plt.axis("off")

            plt.subplot(1, 3, 3); plt.imshow(x[s], cmap="gray"); plt.imshow(result_bright[s], alpha=0.4)
            plt.title("Prediction"); plt.axis("off")

            plt.tight_layout(); plt.savefig(filepath); plt.close()
            print(f"Saved → {filepath}")
