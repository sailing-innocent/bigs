# import torch
# from cent.lib.sailtorch.gs import gs_vis

# if __name__ == "__main__":
#     N = 6
#     points = torch.randn(N, 3).cuda()
#     color = torch.rand(N, 3).cuda()

#     scale = torch.ones(3) * 0.5
#     scale[0] = 1.0

#     scales = torch.stack([scale for _ in range(N)], dim=0).cuda()

#     theta = torch.tensor(45.0 * (3.141592653589793 / 180.0))  # 45 degrees in radians
#     rotq = torch.tensor([1.0, 0.0, 0.0, 0.0])
#     special_rotq = torch.tensor([torch.cos(theta / 2), 0.0, 0.0, torch.sin(theta / 2)])

#     rotqs = torch.stack([rotq for _ in range(N - 1)], dim=0)
#     rotqs = torch.cat([rotqs, special_rotq.unsqueeze(0)], dim=0)

#     rotqs = rotqs.cuda()

#     gs_vis(points, color, scales, rotqs)
