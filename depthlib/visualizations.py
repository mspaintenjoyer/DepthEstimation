"""Visualization utilities for stereo vision pipeline."""

import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_stereo_pair(left_img_rgb, right_img_rgb, title_left='Left Image', title_right='Right Image'):
    """Display two images (stereo pair or rectified pair) side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(left_img_rgb, cmap='gray' if left_img_rgb.ndim == 2 else None)
    axes[0].set_title(title_left)
    axes[0].axis('off')

    axes[1].imshow(right_img_rgb, cmap='gray' if right_img_rgb.ndim == 2 else None)
    axes[1].set_title(title_right)
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_disparity(disparity_px, title='Disparity Map', cmap='jet', vmin=None, vmax=None):
    """
    Visualize disparity map with proper colormap.
    
    Parameters:
    -----------
    disparity_px : np.ndarray
        Raw disparity map in pixels
    title : str
        Title for the plot
    cmap : str
        Colormap ('jet', 'turbo', 'plasma', 'viridis', 'magma')
    vmin, vmax : float
        Min/max values for color scaling (auto if None)
    """
    # Mask invalid disparities (typically <= 0)
    valid_mask = disparity_px > 0
    
    if vmin is None:
        vmin = np.percentile(disparity_px[valid_mask], 1) if valid_mask.any() else 0
    if vmax is None:
        vmax = np.percentile(disparity_px[valid_mask], 99) if valid_mask.any() else disparity_px.max()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(disparity_px, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'{title}\n(Range: {vmin:.1f} - {vmax:.1f} pixels)')
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Disparity (pixels)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()


def visualize_depth(depth_m, title='Depth Map', cmap='turbo_r', max_depth=None, 
                    show_invalid=True, show_meter=True):
    """
    Visualize depth map with proper colormap.
    
    Parameters:
    -----------
    depth_m : np.ndarray
        Depth map in meters (invalid regions may be inf or very large values)
    title : str
        Title for the plot
    cmap : str
        Colormap ('turbo_r' shows close=red, far=blue)
    max_depth : float
        Maximum depth to display (auto if None)
    show_invalid : bool
        If True, show invalid regions (inf/very far) in black
    show_meter : bool
        If True, display depth in meters on colorbar
    """
    if depth_m is None:
        print("Warning: Depth map is None. Cannot visualize.")
        return
    
    # Mask invalid depths (inf, nan, or very large values)
    valid_mask = np.isfinite(depth_m) & (depth_m > 0)
    
    if not valid_mask.any():
        print("Warning: No valid depth values to display.")
        return
    
    # Auto-scale to reasonable depth range
    if max_depth is None:
        max_depth = np.percentile(depth_m[valid_mask], 99)
    
    # Create display array
    depth_display = np.copy(depth_m)
    depth_display[~valid_mask] = max_depth if show_invalid else 0
    depth_display = np.clip(depth_display, 0, max_depth)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(depth_display, cmap=cmap, vmin=0, vmax=max_depth)
    
    # Calculate statistics
    valid_depths = depth_m[valid_mask]
    invalid_pct = 100 * (~valid_mask).sum() / valid_mask.size
    
    ax.set_title(f'{title}\n(Range: {valid_depths.min():.2f} - {max_depth:.2f}m, '
                 f'{invalid_pct:.1f}% invalid/far)')
    ax.axis('off')
    
    if show_meter:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Depth (meters)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()


def visualize_disparity_and_depth(disparity_px, depth_m, left_img=None):
    """
    Visualize disparity and depth side by side with optional reference image.
    
    Parameters:
    -----------
    disparity_px : np.ndarray
        Disparity map in pixels
    depth_m : np.ndarray
        Depth map in meters
    left_img : np.ndarray, optional
        Reference left image to display above
    """
    if left_img is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Reference image
        axes[0].imshow(left_img, cmap='gray' if left_img.ndim == 2 else None)
        axes[0].set_title('Reference Image (Left)')
        axes[0].axis('off')
        
        axes[1].axis('off')  # Empty
        
        disp_ax = axes[2]
        depth_ax = axes[3]
    else:
        fig, (disp_ax, depth_ax) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Disparity visualization
    valid_disp = disparity_px > 0
    vmin_disp = np.percentile(disparity_px[valid_disp], 1) if valid_disp.any() else 0
    vmax_disp = np.percentile(disparity_px[valid_disp], 99) if valid_disp.any() else disparity_px.max()
    
    im1 = disp_ax.imshow(disparity_px, cmap='jet', vmin=vmin_disp, vmax=vmax_disp)
    disp_ax.set_title(f'Disparity Map\n({vmin_disp:.1f} - {vmax_disp:.1f} px)')
    disp_ax.axis('off')
    plt.colorbar(im1, ax=disp_ax, fraction=0.046, pad=0.04, label='Pixels')
    
    # Depth visualization
    if depth_m is not None:
        valid_depth = np.isfinite(depth_m) & (depth_m > 0)
        if valid_depth.any():
            max_depth = np.percentile(depth_m[valid_depth], 95)
            depth_display = np.copy(depth_m)
            depth_display[~valid_depth] = max_depth
            depth_display = np.clip(depth_display, 0, max_depth)
            
            im2 = depth_ax.imshow(depth_display, cmap='turbo_r', vmin=0, vmax=max_depth)
            
            invalid_pct = 100 * (~valid_depth).sum() / valid_depth.size
            depth_ax.set_title(f'Depth Map\n({depth_m[valid_depth].min():.2f} - {max_depth:.2f}m, '
                              f'{invalid_pct:.1f}% invalid)')
            depth_ax.axis('off')
            plt.colorbar(im2, ax=depth_ax, fraction=0.046, pad=0.04, label='Meters')
        else:
            depth_ax.text(0.5, 0.5, 'No valid depth values', 
                         ha='center', va='center', transform=depth_ax.transAxes)
            depth_ax.axis('off')
    else:
        depth_ax.text(0.5, 0.5, 'Depth map not available', 
                     ha='center', va='center', transform=depth_ax.transAxes)
        depth_ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_depth_live(depth_m, fps):
    """Visualize depth map in a live setting with FPS overlay."""
    if depth_m is None:
        print("Warning: Depth map is None. Cannot visualize.")
        return

    valid_depth = np.isfinite(depth_m) & (depth_m > 0)

    if valid_depth.any():
        display_max_depth_m = 50.0
        depth_clipped = np.clip(depth_m, 0, display_max_depth_m)
        depth_clipped[~valid_depth] = display_max_depth_m  # send invalid to far color

        # Emphasize near-range variation with gamma curve
        depth_ratio = depth_clipped / display_max_depth_m
        depth_gamma = np.power(depth_ratio, 0.5)
        depth_norm = (depth_gamma * 255).astype("uint8")

        depth_norm_inv = 255 - depth_norm  # flip so nearer = hotter (red/yellow)
        depth_vis = cv2.applyColorMap(depth_norm_inv, cv2.COLORMAP_TURBO)
        cv2.putText(depth_vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_vis, f"Display cap: {display_max_depth_m:.0f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        depth_vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        cv2.putText(depth_vis, "No valid depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Depth (live)", depth_vis)

def visualize_depth_live_gray(depth_m, fps):
    """Visualize depth map in grayscale in a live setting with FPS overlay."""
    if depth_m is None:
        print("Warning: Depth map is None. Cannot visualize.")
        return

    valid_depth = np.isfinite(depth_m) & (depth_m > 0)

    if valid_depth.any():
        display_max_depth_m = 50.0
        depth_clipped = np.clip(depth_m, 0, display_max_depth_m)
        depth_clipped[~valid_depth] = display_max_depth_m  # send invalid to far (dark)
        depth_ratio = depth_clipped / display_max_depth_m
        depth_norm = ((1.0 - depth_ratio) * 255).astype("uint8")

        depth_vis = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
        cv2.putText(depth_vis, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(depth_vis, f"Display cap: {display_max_depth_m:.0f} m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        depth_vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        cv2.putText(depth_vis, "No valid depth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Depth (live)", depth_vis)
