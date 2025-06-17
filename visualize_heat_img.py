import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
import os
import argparse 
import matplotlib.ticker as ticker

def create_feature_visualization(orig_img, feature_img, output_dir, scene, feature_type="feat_var", max_val=0.2, alpha=0.7, no_label=False, fontsize=16):
    """
    Create and save feature visualization overlays and heatmaps
    
    Args:
        orig_img: Original RGB image
        feature_img: Feature image (grayscale) to visualize
        output_dir: Directory to save output images
        scene: Scene name for filename
        feature_type: Type of feature ("feat" or "feat_var")
        max_val: Maximum value for feature normalization
        alpha: Transparency for overlay
        no_label: Whether to hide titles and labels
        fontsize: Font size for titles and labels
    """
    # Ensure feature image has the same resolution as the original
    if feature_img.shape[:2] != orig_img.shape[:2]:
        feature_img = cv2.resize(feature_img, (orig_img.shape[1], orig_img.shape[0]))
    
    # Normalize feature values to 0-1 range
    feature_norm = feature_img.astype(float) / 255.0
    
    # Filter values within range
    mask = (feature_norm <= max_val) & (feature_norm > 0)
    filtered_feat = np.zeros_like(feature_norm)
    filtered_feat[mask] = feature_norm[mask]
    
    # Create heatmap
    heatmap = np.zeros((filtered_feat.shape[0], filtered_feat.shape[1], 3), dtype=np.uint8)
    
    # Map feature values to colors using jet colormap
    for i in range(filtered_feat.shape[0]):
        for j in range(filtered_feat.shape[1]):
            if filtered_feat[i, j] > 0:
                val = filtered_feat[i, j] / max_val  # Normalize to 0-1
                if val < 0.5:  # Blue to green
                    b = int(255 * (1 - 2 * val))
                    g = int(255 * (2 * val))
                    r = 0
                else:  # Green to red
                    b = 0
                    g = int(255 * (2 - 2 * val))
                    r = int(255 * (2 * val - 1))
                heatmap[i, j] = [r, g, b]
    
    # Create overlay
    beta = 1.0 - alpha
    
    # Create mask where features are present
    overlay_mask = np.any(heatmap > 0, axis=2)
    
    # Initialize result with original image
    result = orig_img.copy()
    
    # Overlay heatmap on original image
    for i in range(orig_img.shape[0]):
        for j in range(orig_img.shape[1]):
            if overlay_mask[i, j]:
                result[i, j, 0] = int(heatmap[i, j, 0] * alpha + orig_img[i, j, 0] * beta)
                result[i, j, 1] = int(heatmap[i, j, 1] * alpha + orig_img[i, j, 1] * beta)
                result[i, j, 2] = int(heatmap[i, j, 2] * alpha + orig_img[i, j, 2] * beta)
    
    # Determine title based on feature type
    if feature_type == "feat_var":
        title_prefix = "Feature Variance"
    else:
        title_prefix = "Feature"

    # Save overlay image
    output_path = os.path.join(output_dir, f"{scene}_{feature_type}_overlay.pdf")
    plt.figure(figsize=(12, 10))
    plt.imshow(result)
    plt.axis('off')
    if not no_label:
        plt.title(f"{title_prefix} Visualization (0-{max_val} range)", fontsize=fontsize+4)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    # Save heatmap with colorbar
    output_path = os.path.join(output_dir, f"{scene}_{feature_type}_heatmap.pdf")
    plt.figure(figsize=(12, 10))
    plt.imshow(filtered_feat, cmap='jet', vmin=0, vmax=max_val)
    if not no_label:
        plt.title(f"{title_prefix} Map (0-{max_val})", fontsize=fontsize+4)
    plt.axis('off')
    cbar = plt.colorbar(fraction=0.046, pad=0.04, shrink=0.8)
    cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(5))  # 设置最多显示5个刻度
    cbar.ax.tick_params(labelsize=fontsize)  # 直接使用指定的字体大小，而不是减2
    if not no_label:
        cbar.ax.tick_params(labelsize=fontsize-2)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def visualize_features_on_image(scene="all", no_label=False, fontsize=16):
    base_dir = "data/result/bigs/ablation_var/ablation_var"
    scenes = [
        "in2n_bear",
        "mip360_kitchen",
    ]
    
    if scene != "all" and scene in scenes:
        scenes = [scene]
    
    frame = 20
    for scene in scenes:
        postfix = "_crt_0.5_stride_1_radius_0_bb_0.0_var_0.2"
        orig_img_path = f"{base_dir}/{scene}/frames/{frame}.jpg"
        feat_var_img_path = f"{base_dir}/{scene}{postfix}/feat_var_imgs/{frame}.jpg"
        feat_img_path = f"{base_dir}/{scene}{postfix}/feat_imgs/{frame}.jpg"

        orig_img = cv2.imread(orig_img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        feat_var_img = cv2.imread(feat_var_img_path, cv2.IMREAD_GRAYSCALE)
        feat_img = cv2.imread(feat_img_path, cv2.IMREAD_GRAYSCALE)
        print(f"orig_img_path: {orig_img_path}")
        print(f"feat_var_img_path: {feat_var_img_path}")
        
        output_dir = "data/result/bigs/bigs_vis_heat_img/"
        os.makedirs(output_dir, exist_ok=True)

        create_feature_visualization(
            orig_img, 
            feat_var_img, 
            output_dir, 
            scene, 
            feature_type="feat_var", 
            max_val=0.2,
            no_label=no_label,
            fontsize=fontsize
        )

        create_feature_visualization(
            orig_img, 
            feat_img, 
            output_dir, 
            scene, 
            feature_type="feat", 
            max_val=1.0,  # Adjust max value as needed for features
            no_label=no_label,
            fontsize=fontsize
        )

        # save original image
        orig_img_path = os.path.join(output_dir, f"{scene}_orig_img.pdf")
        plt.figure(figsize=(12, 10))
        plt.imshow(orig_img)
        plt.axis('off')
        if not no_label:
            plt.title("Original Image", fontsize=fontsize+4)
        plt.savefig(orig_img_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize feature heatmaps on images")
    parser.add_argument("--scene", type=str, default="all", help="The scene to visualize")
    parser.add_argument("--no_label", action="store_true", help="Do not show titles and labels")
    parser.add_argument("--fontsize", type=int, default=16, help="Font size for titles and labels")
    args = parser.parse_args()
    
    visualize_features_on_image(args.scene, args.no_label, args.fontsize)