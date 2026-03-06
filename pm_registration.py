
import torch
import torch.nn.functional as F

import glob
import pickle
import numpy as np
from tqdm.auto import tqdm
import tyro

# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def rot6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Batched Zhou-2019 6-D → SO(3).
      d6 : (B,6)  →  R : (B,3,3)
    """
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (a2 * b1).sum(-1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)          # (B,3,3)


def build_K(fx, fy, cx, cy) -> torch.Tensor:
    """Create 3×3 intrinsics shared by the whole batch."""
    K = torch.zeros(3, 3, device=fx.device)
    K[0, 0], K[1, 1] = fx, fy
    K[0, 2], K[1, 2] = cx, cy
    K[2, 2] = 1.0
    return K


# --------------------------------------------------------------------------- #
#  Main optimisation                                                          #
# --------------------------------------------------------------------------- #
def optimise_xyz_batch(pred_xyz: torch.Tensor,
                       n_iters: int = 1500,
                       lr: float = 5e-3,
                       verbose: int = 100):
    """
    Parallel optimisation on **B** frames:

    pred_xyz : (B, H, W, 3)  3-D predictions (already in metres or normalised units)
    Returns
    -------
      depth   : (B, H, W)
      K       : (3,3)
      R, t    : (B,3,3), (B,3)
    """
    device               = pred_xyz.device
    B, H, W, _           = pred_xyz.shape
    N                    = H * W                      # pixels per frame
    pred_xyz_flat        = pred_xyz.reshape(B, N, 3) # (B,N,3)

    # -- pixel grid (once, CPU → GPU) ----------------------------------------
    ys, xs               = torch.meshgrid(torch.arange(H),
                                          torch.arange(W),
                                          indexing='ij')
    pix_h                = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1) \
                                .reshape(1, N, 3).to(device)   # (1,N,3)

    # -- parameters ----------------------------------------------------------
    # Depth directly in predicted scale; clamp in forward pass to avoid 0.
    depth = torch.nn.Parameter(pred_xyz_flat.norm(dim=-1))     # (B,N)

    # Shared intrinsics
    fx = torch.nn.Parameter(torch.tensor(W * 1.2, device=device))
    fy = torch.nn.Parameter(torch.tensor(H * 1.2, device=device))
    cx = torch.nn.Parameter(torch.tensor(W / 2,  device=device))
    cy = torch.nn.Parameter(torch.tensor(H / 2,  device=device))

    # Per-frame extrinsics
    rot6d = torch.nn.Parameter(torch.eye(3, device=device)
                               .repeat(B, 1, 1)[:, :, :2]
                               .reshape(B, 6))          # (B,6)
    trans = torch.nn.Parameter(torch.zeros(B, 3, device=device))

    optimiser = torch.optim.Adam([depth, fx, fy, cx, cy, rot6d, trans], lr=lr)

    # -- main loop -----------------------------------------------------------
    for it in range(0, n_iters):
        optimiser.zero_grad()

        # Depth positivity (very mild clamp so gradients flow)
        d = depth.clamp(min=1e-4).unsqueeze(-1)        # (B,N,1)

        K     = build_K(fx, fy, cx, cy)                # (3,3)
        Kinv  = torch.inverse(K).unsqueeze(0)          # (1,3,3)

        # Camera-space pts:   (B,3,3)@(B,3,N) via broadcasting
        cam_pts = torch.matmul(Kinv,                   # (1,3,3)
                               (d * pix_h)             # (B,N,3)
                               .permute(0, 2, 1))      # → (B,3,N)

        # Build per-frame 4×4 cam→world matrices
        R   = rot6d_to_matrix(rot6d)                   # (B,3,3)
        t   = trans.unsqueeze(-1)                      # (B,3,1)

        ones = torch.tensor([0, 0, 0, 1], device=device) \
                    .view(1, 1, 4).repeat(B, 1, 1)     # (B,1,4)

        P_cam2world = torch.cat([torch.cat([R, t], dim=2), ones], dim=1)  # (B,4,4)

        cam_pts_h   = torch.cat([cam_pts, torch.ones(B, 1, N, device=device)], dim=1)
        world_pts   = (P_cam2world @ cam_pts_h)[:, :3]      # (B,3,N)
        world_pts   = world_pts.permute(0, 2, 1)            # (B,N,3)

        loss = (world_pts - pred_xyz_flat).pow(2).mean()
        loss.backward()
        optimiser.step()

        if verbose and it % verbose == 0:
            print(f"[{it:4d}/{n_iters}]  reprojection L2 = {loss.item():.6f}")

    depth_maps = depth.clamp(min=1e-4).detach().reshape(B, H, W)
    K_final    = build_K(fx, fy, cx, cy).detach()
    return depth_maps, K_final, R.detach(), trans.detach()

def depth_to_3d_points(depth_map: torch.Tensor, K: torch.Tensor, R: torch.Tensor, t: torch.Tensor) -> np.ndarray:
    """Convert depth map to 3D points in world coordinates.
    
    Args:
        depth_map: (H,W) depth values
        K: (3,3) camera intrinsics 
        R: (3,3) rotation matrix
        t: (3,) translation vector
    Returns:
        points: (N,3) 3D points in world coordinates
    """
    H, W = depth_map.shape
    device = depth_map.device
    
    # Get pixel coordinates
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    pix_h = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).to(device)
    
    # Unproject using depth and camera intrinsics
    d = depth_map.unsqueeze(-1)  # Add channel dim
    cam_pts = torch.matmul(torch.inverse(K),
                         (d * pix_h).reshape(-1, 3).T).T  # (N,3)
    
    # Transform to world coordinates  
    world_pts = (R @ cam_pts.T).T + t
    
    return world_pts.cpu().numpy()



def main(
    pkl_dir: str = None,
    max_frames: int = 100,
    frame_gap: int = 1,
) -> None:
    """Optimize camera parameters for Pointmap pkl files."""
    if pkl_dir is None:
        raise ValueError("--pkl_dir is required")

    from core.datasets.dataclass import Pointmap

    pkl_list = sorted(glob.glob(f'{pkl_dir}/*.pkl'))
    for pkl_path in pkl_list:
        pm = pickle.load(open(pkl_path, 'rb'))
        num_frames = min(max_frames, pm.num_frames)
        H, W = pm.height, pm.width

        xyz = pm.xyz[:num_frames].copy()  # [N, H, W, 3]
        xyz[..., :2] = xyz[..., :2] - 0.5
        xyz_tensor = torch.from_numpy(xyz).cuda()

        depth_map, K, R, t = optimise_xyz_batch(xyz_tensor)

        # Reconstruct optimized pointmap
        frame_indices = list(range(0, num_frames, frame_gap))
        updated_xyz = []
        for fi in tqdm(frame_indices):
            pts = depth_to_3d_points(depth_map[fi], K, R[fi], t[fi])
            updated_xyz.append(pts)
        updated_xyz = np.stack(updated_xyz).reshape(-1, H, W, 3)

        cams2world = np.eye(4)[None].repeat(len(frame_indices), axis=0)
        for j, fi in enumerate(frame_indices):
            cams2world[j, :3, :3] = R[fi].cpu().numpy()
            cams2world[j, :3, 3] = t[fi].cpu().numpy()

        opt_pm = Pointmap(
            xyz=updated_xyz,
            rgb=pm.rgb[frame_indices],
            depth=depth_map[frame_indices].cpu().numpy(),
            cams2world=cams2world,
            K=K.cpu().numpy()[None].repeat(len(frame_indices), axis=0),
        )

        save_path = pkl_path.replace('.pkl', '_reg.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(opt_pm, f)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    tyro.cli(main)