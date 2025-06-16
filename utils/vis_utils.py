from scene.gaussian_model import GaussianModel
import numpy as np 
import open3d as o3d 
import seaborn as sns
colors = sns.color_palette("hls", 7)
import itertools
color_iter = itertools.cycle(colors)
from cent.utils.math.transform import qvec2R 
from cent.utils.camera import Camera as SailCamera

def o3d_vis_gaussian(gs: GaussianModel):
    t = gs.get_xyz
    s = gs.get_scaling
    r = gs.get_rotation
    t = t.detach().cpu().numpy()
    s = s.detach().cpu().numpy()
    r = r.detach().cpu().numpy()

    N = t.shape[0]
    if (N > 10000):
        N = 10000
    els = []
    for i in range(N):
        R = qvec2R(r[i])
        S = np.diag(s[i])
        T = np.eye(4)
        T[:3, :3] = R @ S
        T[:3, 3] = t[i]

        el = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        el.transform(T)
        # iter color
        color = next(color_iter)
        color_vec3 = np.array(color)
        el.paint_uniform_color(color_vec3)
        els.append(el)

    o3d.visualization.draw_geometries(els)

def vis_pcd_with_cam(pcd, cam: SailCamera):
    cam_lines = o3d.geometry.LineSet.create_camera_visualization(
        cam.info.ResW,
        cam.info.ResH,
        cam.info.K,
        cam.view_matrix
    )
    o3d.visualization.draw_geometries(pcd, cam_lines)

import matplotlib as mpl 
from scipy import linalg

def plot_gaussians_2d(ax, means, covariances, title):
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi
        ell = mpl.patches.Ellipse(mean, v[0], v[1], angle=180.0 + angle, color=color)
        ell.set_alpha(0.5)
        ell.set_clip_box(ax.bbox)
        ax.add_artist(ell)


