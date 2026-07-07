"""Depth-based layer splitting via 1D k-means clustering.

When an image has distinct depth modes (common for composed cartoon scenes with
clear background / midground / character / foreground separation), clustering
the depth map directly yields clean layer masks without any manual clicking.

The approach:
1. Run 1D k-means on the depth values (subsample for speed, assign full image).
2. Sort clusters by depth center (layer 0 = farthest).
3. Clean each cluster's mask morphologically (opening + small-component removal)
   so compositing edges are crisp.
4. Emit one boolean mask per cluster.

This complements the SAM2 picker: use clustering for fast automatic splits,
the picker to refine specific edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ClusterResult:
    """Output of depth clustering.

    Attributes:
        labels: ``(H, W)`` int array; cluster id in ``[0, k)`` ordered
            far-to-near (id 0 = farthest depth center).
        centers: ``(k,)`` float array of cluster depth centers, sorted ascending.
        masks: list of ``k`` boolean ``(H, W)`` masks, indexed far-to-near.
    """

    labels: np.ndarray
    centers: np.ndarray
    masks: list[np.ndarray]


def kmeans_1d(
    x: np.ndarray,
    k: int,
    n_iter: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """1D k-means on a flat depth array.

    Args:
        x: ``(N,)`` float values in ``[0, 1]``.
        k: number of clusters.
        n_iter: max iterations.
        seed: RNG seed for determinism.

    Returns:
        ``(centers, labels)``: centers sorted ascending; labels in ``[0, k)``
        aligned with the original ``x`` ordering.
    """
    _ = seed  # quantile init is deterministic; kept for API stability.
    # Quantile-spaced init avoids empty clusters and is deterministic.
    centers = np.quantile(x, np.linspace(1.0 / (k + 1), k / (k + 1), k))

    labels = np.zeros(len(x), dtype=np.int32)
    for _ in range(n_iter):
        # Assign: nearest center.
        dists = np.abs(x[:, None] - centers[None, :])
        new_labels = dists.argmin(axis=1)
        # Update: mean of each cluster (keep center if empty).
        new_centers = np.array(
            [x[new_labels == i].mean() if (new_labels == i).any() else centers[i] for i in range(k)]
        )
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers, atol=1e-6):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers

    order = np.argsort(centers)
    centers = centers[order]
    labels = order[labels].astype(np.int32)
    return centers, labels


def _remove_small_components(mask: np.ndarray, min_fraction: float) -> np.ndarray:
    """Remove connected components smaller than ``min_fraction`` of the mask area.

    Uses scipy's label() if available (fast), else a flood-fill fallback.
    """
    if not mask.any() or min_fraction <= 0:
        return mask
    try:
        from scipy import ndimage

        labeled, n = ndimage.label(mask)
        if n <= 1:
            return mask
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
        min_size = min_fraction * mask.sum()
        keep = np.where(sizes >= min_size)[0] + 1
        out = np.isin(labeled, keep)
        return out
    except ImportError:
        # Fallback: no scipy; return mask unchanged (cleanup is best-effort).
        return mask


def _morphological_clean(mask: np.ndarray, open_radius: int) -> np.ndarray:
    """Erode then dilate (opening) to remove specks and thin protrusions.

    Uses scipy if available; otherwise a no-op (caller should ensure scipy is
    present, or accept raw clusters).
    """
    if open_radius <= 0:
        return mask
    try:
        from scipy import ndimage

        structure = ndimage.generate_binary_structure(2, 1)
        eroded = ndimage.binary_erosion(mask, structure=structure, iterations=open_radius)
        dilated = ndimage.binary_dilation(eroded, structure=structure, iterations=open_radius)
        return np.asarray(dilated)
    except ImportError:
        return mask


def cluster_depth(
    depth: np.ndarray,
    k: int = 5,
    subsample: int = 8,
    open_radius: int = 1,
    min_component_fraction: float = 0.02,
) -> ClusterResult:
    """Cluster a depth map into ``k`` depth-ordered layers.

    Args:
        depth: ``(H, W)`` float depth in ``[0, 1]`` (0 = far, 1 = near).
        k: number of clusters.
        subsample: stride for the k-means fit (speed); assignment uses all pixels.
        open_radius: morphological opening radius (px) for edge cleanup.
        min_component_fraction: connected components smaller than this fraction
            of the cluster area are removed.

    Returns:
        :class:`ClusterResult` with masks ordered far-to-near.
    """
    flat = depth.ravel()
    fit = flat[::subsample] if subsample > 1 else flat
    centers, _ = kmeans_1d(fit, k)

    # Assign every pixel to the nearest (depth-sorted) center.
    dists = np.abs(flat[:, None] - centers[None, :])
    labels = dists.argmin(axis=1).reshape(depth.shape)

    masks: list[np.ndarray] = []
    for i in range(k):
        mask = labels == i
        mask = _morphological_clean(mask, open_radius)
        mask = _remove_small_components(mask, min_component_fraction)
        masks.append(mask)

    # Cleanup can drop pixels (edges, specks). Re-assign any uncovered pixel to
    # the nearest-depth cluster so the masks fully tile the frame — this avoids a
    # spurious "background residual" being synthesized downstream.
    covered = np.zeros(depth.shape, dtype=bool)
    for m in masks:
        covered |= m
    gaps = ~covered
    if gaps.any():
        gap_labels = labels[gaps]
        for i in range(k):
            gi = gaps.copy()
            gi[gaps] = gap_labels == i
            masks[i] = masks[i] | gi

    return ClusterResult(labels=labels, centers=centers, masks=masks)
