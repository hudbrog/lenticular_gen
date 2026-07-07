"""Tests for depth-based clustering and its integration with assemble.

Uses synthetic depth maps with distinct modes so cluster boundaries are
predictable; no ML models needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lenticular.io_utils import save_depth
from lenticular.layers import BACKGROUND_LAYER_NAME
from lenticular.pipeline.cluster import cluster_depth, kmeans_1d
from lenticular.pipeline.separate import assemble_layers


@pytest.fixture()
def layered_depth(tmp_path):
    """A depth map with 4 distinct depth bands arranged as spatial regions.

    Layout (100x120):
      rows  0:25   → depth 0.15  (far background, top strip)
      rows 25:50   → depth 0.40  (mid background)
      rows 50:75   → depth 0.65  (character midground)
      rows 75:100  → depth 0.90  (foreground, bottom strip)
    Each band is a clean horizontal slab so cluster boundaries are sharp.
    """
    h, w = 100, 120
    depth = np.zeros((h, w), dtype=np.float64)
    depth[0:25, :] = 0.15
    depth[25:50, :] = 0.40
    depth[50:75, :] = 0.65
    depth[75:100, :] = 0.90
    path = tmp_path / "depth.png"
    save_depth(depth, path)
    return depth, path, (h, w)


class TestKMeans1D:
    def test_recovers_known_modes(self):
        # Four well-separated modes; k-means should find each.
        x = np.concatenate(
            [np.full(100, 0.1), np.full(100, 0.4), np.full(100, 0.6), np.full(100, 0.9)]
        )
        centers, labels = kmeans_1d(x, k=4)
        assert len(centers) == 4
        # Centers sorted ascending, near the true modes.
        np.testing.assert_allclose(np.sort(centers), [0.1, 0.4, 0.6, 0.9], atol=0.02)

    def test_labels_sorted_far_to_near(self):
        x = np.array([0.9, 0.1, 0.5, 0.9, 0.1, 0.5])
        centers, labels = kmeans_1d(x, k=3)
        # Label 0 = smallest depth (0.1), label 2 = largest (0.9).
        assert centers[0] < centers[1] < centers[2]
        assert labels[1] == 0  # x[1]=0.1 → farthest cluster
        assert labels[0] == 2  # x[0]=0.9 → nearest cluster


class TestClusterDepth:
    def test_four_clusters_match_bands(self, layered_depth):
        depth, _, shape = layered_depth
        result = cluster_depth(depth, k=4, open_radius=0, min_component_fraction=0)
        assert len(result.masks) == 4
        # Centers ascending.
        assert np.all(np.diff(result.centers) > 0)
        # Each band maps predominantly to one cluster, in order.
        for band_idx, (r0, r1) in enumerate([(0, 25), (25, 50), (50, 75), (75, 100)]):
            band_labels = result.labels[r0:r1, :]
            # The most common label in this band should equal band_idx.
            assert np.bincount(band_labels.ravel()).argmax() == band_idx

    def test_masks_partition_image(self, layered_depth):
        """Every pixel belongs to exactly one cluster (no overlap, no gaps)."""
        depth, _, _ = layered_depth
        result = cluster_depth(depth, k=4, open_radius=0, min_component_fraction=0)
        total = np.zeros(depth.shape, dtype=int)
        for m in result.masks:
            total += m.astype(int)
        # Cleanup + gap-fill: every pixel covered exactly once.
        np.testing.assert_array_equal(total, np.ones_like(total))

    def test_cleanup_removes_specks(self):
        """Isolated specks are removed by component filtering, then gap-filled."""
        depth = np.full((50, 50), 0.2, dtype=np.float64)
        depth[40:, :] = 0.8  # near band at bottom
        # Sprinkle isolated specks of "near" depth in the far region.
        depth[5, 5] = 0.9
        depth[10, 30] = 0.9
        result = cluster_depth(depth, k=2, open_radius=1, min_component_fraction=0.05)
        # The contiguous near band is present.
        assert result.masks[1][45, 25]
        # Masks still fully tile the frame after gap-fill.
        total = sum(m.astype(int) for m in result.masks)
        np.testing.assert_array_equal(total, np.ones_like(total))


class TestAssembleIntegration:
    def test_cluster_masks_assemble_ordered(self, layered_depth, tmp_path):
        """Cluster masks flow through assemble with correct ordering + focus."""
        depth, depth_path, shape = layered_depth
        result = cluster_depth(depth, k=4, open_radius=0, min_component_fraction=0)
        # Write cluster masks to a masks dir.
        from lenticular.io_utils import save_mask

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        for i, m in enumerate(result.masks):
            save_mask(m, masks_dir / f"layer_{i}.png")

        ls = assemble_layers(masks_dir, depth_path, shape)
        names = [layer.name for layer in ls]
        # 4 cluster layers + synthesized background residual (if any).
        assert len(names) >= 4
        # Ordered back-to-front by depth (centers ascending).
        depths = [layer.depth for layer in ls]
        assert depths == sorted(depths)
        # Focus = median-depth layer (no character mask present).
        median = float(np.median(depths))
        focus = ls.focus
        assert abs(focus.depth - median) == min(abs(d - median) for d in depths)

    def test_no_background_residual_when_clusters_cover_all(self, layered_depth, tmp_path):
        """If clusters cover the whole frame, no background layer is synthesized."""
        depth, depth_path, shape = layered_depth
        result = cluster_depth(depth, k=4, open_radius=0, min_component_fraction=0)
        from lenticular.io_utils import save_mask

        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()
        for i, m in enumerate(result.masks):
            save_mask(m, masks_dir / f"layer_{i}.png")
        ls = assemble_layers(masks_dir, depth_path, shape)
        # Clusters tile the image → no background layer.
        assert BACKGROUND_LAYER_NAME not in [layer.name for layer in ls]
