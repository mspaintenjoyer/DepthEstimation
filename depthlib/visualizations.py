"""
Visualization helpers for disparity and depth.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def visualize_disparity(
    disparity: np.ndarray,
    title: str = "Disparity Map",
    cmap: str = "jet",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Display a disparity map with a colorbar.

    disparity : np.ndarray
        Disparity map (float32).
    """
    valid = disparity > 0
    if vmin is None:
        vmin = float(disparity[valid].min()) if valid.any() else float(disparity.min())
    if vmax is None:
        vmax = float(disparity[valid].max()) if valid.any() else float(disparity.max())

    plt.figure(figsize=(10, 6))
    im = plt.imshow(disparity, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f"{title}\n(Range: {vmin:.1f} - {vmax:.1f} pixels)")
    plt.colorbar(im, label="Disparity (pixels)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualize_depth(
    depth: np.ndarray,
    title: str = "Depth Map",
    cmap: str = "turbo_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Display a depth map with a colorbar.

    depth : np.ndarray
        Depth map in meters. Invalid / infinite values are masked out.
    """
    valid = np.isfinite(depth)
    if vmin is None:
        vmin = float(depth[valid].min()) if valid.any() else float(depth.min())
    if vmax is None:
        vmax = float(depth[valid].max()) if valid.any() else float(depth.max())

    depth_vis = depth.copy()
    depth_vis[~valid] = vmax

    plt.figure(figsize=(10, 6))
    im = plt.imshow(depth_vis, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f"{title}\n(Range: {vmin:.2f} - {vmax:.2f} m)")
    plt.colorbar(im, label="Depth (m)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
