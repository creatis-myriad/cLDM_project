import os
import pickle
import sys
import torch

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import directed_hausdorff
from monai.metrics import compute_dice
from monai.metrics import compute_hausdorff_distance


def compute_dice_for_class(seg1, seg2, class_label):
    intersection = np.sum((seg1 == class_label) & (seg2 == class_label))
    volume_sum = np.sum(seg1 == class_label) + np.sum(seg2 == class_label)
    if volume_sum == 0: return 1.0
    else: return 2.0 * intersection / volume_sum


def compute_hausdorff_distance_for_class(seg1, seg2, class_label):
    seg1_points = np.argwhere(seg1 == class_label)
    seg2_points = np.argwhere(seg2 == class_label)
    if len(seg1_points) == 0 or len(seg2_points) == 0:
        return np.nan
    forward_hausdorff = directed_hausdorff(seg1_points, seg2_points)[0]
    backward_hausdorff = directed_hausdorff(seg2_points, seg1_points)[0]
    return max(forward_hausdorff, backward_hausdorff)



def main():
    path_ = sys.argv[-1]
    with open(os.path.join(path_,"D_img_generated.pkl"), 'rb') as handle:
        D_img_generated = pickle.load(handle)

    output_path = "/home/deleat/Documents/RomainD/Working_space/Figures_Resultats/Results_LLDM_project/_figures/dice_oriSeg_genSeg/"
    folder_exp = os.path.join(output_path, os.path.basename(path_))
    os.makedirs(folder_exp, exist_ok=True)

    colors = [
        (0.0, 0.0, 1.0),  # Blue
        (0.6, 0.8, 1.0),  # Light Blue
        (1.0, 1.0, 1.0),  # White (Center)
        (1.0, 0.7, 0.6),  # Light Red
        (1.0, 0.0, 0.0)   # Red
    ]
    custom_coolwarm = LinearSegmentedColormap.from_list("custom_coolwarm", colors, N=256)


    L_dice_myo = []
    L_haus_myo = []
    L_dice_inf = []
    L_haus_inf = []
    L_origin_seg_one_hot    = []
    L_generated_seg_one_hot = []
    num_classes = 3  
    for filename in sorted(os.listdir(os.path.join(path_, "seg_generated_processed"))):
        print(filename)
        name = filename.split(".")[0]
        idx_ = D_img_generated["filename"].index(filename)
        origin_seg = D_img_generated["img_conditioning"][idx_]
        origin_seg = np.round(origin_seg).astype(np.int8).squeeze()

        with open(os.path.join(path_, "seg_generated_processed", filename), 'rb') as handle:
            roi_seg_generated = pickle.load(handle)
        
        infarct = roi_seg_generated.segmentsResampled[0]["MI"]
        myocard = roi_seg_generated.segmentsResampled[0]["non-MI"]
        generated_seg = infarct+myocard


        plt.figure(figsize=(20,15), constrained_layout=True)
        plt.subplot(331)
        plt.imshow(D_img_generated["img_generated"][idx_].squeeze(), cmap="gray")
        plt.axis('off')
        plt.title("Generated image")
        plt.subplot(334)
        plt.imshow(np.array(origin_seg-generated_seg, dtype=np.int8), cmap=custom_coolwarm, vmin=-2, vmax=2)
        plt.axis('off')
        plt.title("Difference (original - generated)")     
        plt.subplot(333)
        plt.imshow(D_img_generated["img_generated"][idx_].squeeze(), cmap="gray")
        plt.imshow(origin_seg, cmap="jet", vmin=0, vmax=2, alpha=0.4)
        plt.axis('off')
        plt.title("with original segmentation")
        plt.subplot(332)
        plt.imshow(origin_seg, cmap="gray", vmin=0, vmax=2)
        plt.axis('off')
        plt.title("Original segmentation")
        plt.subplot(336)
        plt.imshow(D_img_generated["img_generated"][idx_].squeeze(), cmap="gray")
        plt.imshow(generated_seg, cmap="jet", vmin=0, vmax=2, alpha=0.4)
        plt.axis('off')
        plt.title("with generated segmentation")
        plt.subplot(335)
        plt.imshow(generated_seg, cmap="gray", vmin=0, vmax=2)
        plt.axis('off')
        plt.title("Generated segmentation")
        plt.subplot(337)
        diff_segs = np.array(origin_seg-generated_seg, dtype=np.float32)
        diff_segs[np.where(diff_segs == 0)] = np.nan
        plt.imshow(D_img_generated["img_generated"][idx_].squeeze(), cmap="gray")
        plt.imshow(diff_segs, cmap=custom_coolwarm, vmin=-2, vmax=2)
        plt.axis('off')
        plt.title("Diff + image")
        # plt.show()
        if not os.path.exists(os.path.join(folder_exp, "comparison_segmentations")):
            os.mkdir(os.path.join(folder_exp, "comparison_segmentations"))
        plt.savefig(os.path.join(folder_exp, "comparison_segmentations", f"{name}.svg"))
        plt.close("all")


        dice_myo = compute_dice_for_class(origin_seg, generated_seg, class_label=1)
        haus_myo = compute_hausdorff_distance_for_class(origin_seg, generated_seg, class_label=1)
        dice_inf = compute_dice_for_class(origin_seg, generated_seg, class_label=2)
        haus_inf = compute_hausdorff_distance_for_class(origin_seg, generated_seg, class_label=2)

        origin_seg_one_hot    = np.eye(num_classes)[origin_seg].transpose(2, 0, 1)
        generated_seg_one_hot = np.eye(num_classes)[generated_seg].transpose(2, 0, 1)

        L_dice_myo.append(dice_myo)
        L_haus_myo.append(haus_myo)
        L_dice_inf.append(dice_inf)
        L_haus_inf.append(haus_inf)
        L_origin_seg_one_hot.append(origin_seg_one_hot)
        L_generated_seg_one_hot.append(generated_seg_one_hot)

    L_dice_myo = np.array(L_dice_myo)
    L_haus_myo = np.array(L_haus_myo)
    L_dice_inf = np.array(L_dice_inf)
    L_haus_inf = np.array(L_haus_inf)
    L_origin_seg_one_hot    = np.array(L_origin_seg_one_hot)
    L_generated_seg_one_hot = np.array(L_generated_seg_one_hot)

    L_origin_seg_one_hot    = torch.Tensor(L_origin_seg_one_hot)
    L_generated_seg_one_hot = torch.Tensor(L_generated_seg_one_hot)
    dice_monai = compute_dice(
        L_generated_seg_one_hot,
        L_origin_seg_one_hot,
        include_background=False,
        ignore_empty=True,
    ).numpy()
    dice_monai_myo = dice_monai[:, 0]
    dice_monai_inf = dice_monai[:, 1]


    haus_monai = compute_hausdorff_distance(
        L_generated_seg_one_hot,
        L_origin_seg_one_hot,
        include_background=False,
    ).numpy()
    haus_monai = haus_monai[~np.isnan(haus_monai).any(axis=1)]
    haus_monai = haus_monai[~np.isinf(haus_monai).any(axis=1)]
    haus_monai_myo = haus_monai[:, 0]
    haus_monai_inf = haus_monai[:, 1]


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    textstr = '\n'.join((
        f"Mean Dice Myocard: {np.nanmean(L_dice_myo):.4f} ± {np.nanstd(L_dice_myo):.4f}",
        f"Mean Dice Myocard Monai: {np.nanmean(dice_monai_myo):.4f} ± {np.nanstd(dice_monai_myo):.4f}",
        "",
        f"Mean Dice Infarct: {np.nanmean(L_dice_inf):.4f} ± {np.nanstd(L_dice_inf):.4f}",
        f"Mean Dice Infarct Monai: {np.nanmean(dice_monai_inf):.4f} ± {np.nanstd(dice_monai_inf):.4f}",
        "",
        f"Mean Hausdorff Myocard: {np.nanmean(L_haus_myo):.4f} ± {np.nanstd(L_haus_myo):.4f}",
        f"Mean Hausdorff Myocard Monai: {np.nanmean(haus_monai_myo):.4f} ± {np.nanstd(haus_monai_myo):.4f}",
        "",
        f"Mean Hausdorff Infarct: {np.nanmean(L_haus_inf):.4f} ± {np.nanstd(L_haus_inf):.4f}",
        f"Mean Hausdorff Infarct Monai: {np.nanmean(haus_monai_inf):.4f} ± {np.nanstd(haus_monai_inf):.4f}",
    ))
    ax.text(0.5, 0.5, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='center', horizontalalignment='center')
    plt.savefig(os.path.join(folder_exp, "metrics_summary.png"))



if __name__ == "__main__" :
    main()










