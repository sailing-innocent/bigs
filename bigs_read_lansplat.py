import os 
import numpy as np
import torch 
import matplotlib.pyplot as plt
from lerf_eval.openclip_encoder import OpenCLIPNetwork
import argparse 
from utils.lerf_utils import feat_img_to_color_img

@torch.no_grad()
def main(dataset_path: str = "E:/ws/data/datasets/lerf_ovs/teatime", frame: str = "frame_00001", out_dir: str = "E:/ws/data/mid/bigs/lang_results", level: int = 1):
    device = "cuda"
    scene_name = dataset_path.split("/")[-1]
    lang_feat_path = os.path.join(dataset_path, "language_features")
    lang_feat_dim3_path = os.path.join(dataset_path, "language_features_3d")
    seg_file_name = os.path.join(lang_feat_path, frame + "_s.npy")
    feat_file_name = os.path.join(lang_feat_path, frame + "_f.npy")

    seg_map = torch.from_numpy(np.load(seg_file_name))
    h = seg_map.shape[1]
    w = seg_map.shape[2]
    feature_map = torch.from_numpy(np.load(feat_file_name))
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    seg = seg_map[:, y, x].squeeze(-1).long()
    mask = seg != -1
    if level == 0:
        point_feature1 = feature_map[seg[0:1]].squeeze(0)
        mask = mask[0:1].reshape(-1, h, w)
    if level == 1: # s
        point_feature1 = feature_map[seg[1:2]].squeeze(0)
        mask = mask[1:2].reshape(1, h, w)
    if level == 2: # m 
        point_feature1 = feature_map[seg[2:3]].squeeze(0)
        mask = mask[2:3].reshape(1, h, w)
    if level == 3: # l 
        point_feature1 = feature_map[seg[3:4]].squeeze(0)
        mask = mask[3:4].reshape(1, h, w)

    point_feature = point_feature1.reshape(h, w, -1).cuda()
    print(point_feature.shape) # h, w, 512
    print(mask.shape) # 1, h, w 

    # query = "stuffed bear"
    # clip_model = OpenCLIPNetwork(device)
    # clip_model.set_positives([query])

    # visualize the point_feature with PCA
    point_feat_np = point_feature.cpu().numpy() # h, w, 512
    feat_pca = feat_img_to_color_img(point_feat_np)
    # apply mask, make mask == 0 to black
    mask_np = mask.cpu().numpy() # 1, h, w
    mask_np = mask_np.squeeze(0) # h, w
    mask_np = mask_np.astype(np.uint8) # h, w
    feat_pca = feat_pca * mask_np[:, :, np.newaxis] # h, w, 3

    plt.imshow(feat_pca)
    plt.axis('off')
    plt.show()

    # save to 
    os.makedirs(out_dir, exist_ok=True)
    out_file_name = os.path.join(out_dir,scene_name + "_" + frame + f"_{level}_f.png")
    plt.imsave(out_file_name, feat_pca)
    print("save to: ", out_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the language features")
    parser.add_argument("--dataset_path", type=str, default="E:/ws/data/datasets/lerf_ovs/teatime", help="Path to the dataset")
    parser.add_argument("--frame", type=str, default="frame_00001", help="Frame name")
    parser.add_argument("--out_dir", type=str, default="E:/ws/data/mid/bigs/lang_results", help="Output directory")
    parser.add_argument("--level", type=int, default=1, help="Level of the feature to visualize")
    args = parser.parse_args()
    # fix random seed
    rand_seed = 42
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

    main(args.dataset_path, args.frame, args.out_dir, args.level)