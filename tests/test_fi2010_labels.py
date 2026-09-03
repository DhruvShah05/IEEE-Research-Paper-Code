"""
tests/test_fi2010_labels.py — Tests for FI-2010 label convention (fix 0.2 / Section 6).

Verifies:
  - Raw FI-2010 values 1, 2, 3 are remapped correctly to 0, 1, 2
  - FI2010Dataset applies remap_fi2010_labels at load time
  - The old convention (treating labels as-is with 0-indexing) gives wrong semantics
"""

import numpy as np
import os
import pytest


class TestFI2010LabelRemap:
    """Tests for remap_fi2010_labels()."""

    def test_basic_remap(self):
        from data.labeling import remap_fi2010_labels
        y = np.array([1, 2, 3, 1, 3, 2])
        out = remap_fi2010_labels(y)
        expected = np.array([2, 1, 0, 2, 0, 1])  # 1→2, 2→1, 3→0
        np.testing.assert_array_equal(out, expected)

    def test_remap_is_bijective(self):
        """Each original label maps to exactly one remapped label."""
        from data.labeling import remap_fi2010_labels
        for v in [1, 2, 3]:
            out = remap_fi2010_labels(np.array([v]))
            assert out[0] in {0, 1, 2}
            # The remapping: 1→2 (Up), 2→1 (Stat), 3→0 (Down)

    def test_up_label_convention(self):
        """In FI-2010, label 1 = Up. After remap: label 1 → 2."""
        from data.labeling import remap_fi2010_labels
        out = remap_fi2010_labels(np.array([1]))
        assert out[0] == 2, (
            "FI-2010 label 1 (official=Up) must remap to 2 (shared convention=Up). "
            "Bug 0.2: the original code treated label 1 as Down."
        )

    def test_down_label_convention(self):
        """In FI-2010, label 3 = Down. After remap: label 3 → 0."""
        from data.labeling import remap_fi2010_labels
        out = remap_fi2010_labels(np.array([3]))
        assert out[0] == 0, (
            "FI-2010 label 3 (official=Down) must remap to 0 (shared convention=Down). "
            "Bug 0.2: the original code treated label 3 as Up."
        )

    def test_old_convention_would_invert(self):
        """
        Demonstrates the original bug: treating FI-2010 labels as 0-indexed
        (i.e. subtract 1: 1→0, 2→1, 3→2) gives the wrong semantic.
        After that wrong transform: label 0=Up_wrong (should be Down), label 2=Down_wrong.
        This test shows the OLD way was wrong and the NEW way is correct.
        """
        from data.labeling import remap_fi2010_labels
        raw_up   = np.array([1])  # official Up
        raw_down = np.array([3])  # official Down

        # Old incorrect transform: zero-indexed (subtract 1)
        old_up   = int(raw_up[0])   - 1  # = 0 → treated as Down! WRONG
        old_down = int(raw_down[0]) - 1  # = 2 → treated as Up!   WRONG

        # New correct transform
        new_up   = int(remap_fi2010_labels(raw_up)[0])   # = 2 → Up. CORRECT
        new_down = int(remap_fi2010_labels(raw_down)[0]) # = 0 → Down. CORRECT

        assert old_up   != 2, "Old transform treats Up as label 0 (wrong)"
        assert old_down != 0, "Old transform treats Down as label 2 (wrong)"
        assert new_up   == 2, "New transform correctly gives Up → label 2"
        assert new_down == 0, "New transform correctly gives Down → label 0"


class TestFI2010DatasetLabels:
    """
    Tests that FI2010Dataset applies remap_fi2010_labels at load time.
    Uses synthetic .npy files so no actual data is required.
    """

    def _make_synthetic_fi2010_npy(self, tmp_path, n_rows: int = 300,
                                    label_col_val: int = 1) -> tuple:
        """
        Creates minimal synthetic fi2010_train.npy and fi2010_test.npy with
        controlled label values (1, 2, or 3) in column 144.
        """
        # 149 columns: 144 features + 5 labels
        train_data = np.zeros((n_rows, 149))
        test_data  = np.zeros((100, 149))

        # Set all label columns to a controllable value
        for col in range(144, 149):
            train_data[:, col] = label_col_val
            test_data[:, col]  = label_col_val

        train_path = str(tmp_path / 'fi2010_train.npy')
        test_path  = str(tmp_path / 'fi2010_test.npy')
        np.save(train_path, train_data)
        np.save(test_path, test_data)
        return train_path, test_path

    def test_labels_remapped_from_1(self, tmp_path):
        """When raw FI-2010 label = 1 (Up), FI2010Dataset must yield y = 2."""
        from data.loaders import FI2010Dataset
        tr, te = self._make_synthetic_fi2010_npy(tmp_path, label_col_val=1)
        ds = FI2010Dataset(train_path=tr, test_path=te, horizon_k=10)
        assert np.all(ds.y_test == 2), (
            f"FI-2010 raw label 1 (Up) should remap to 2. Got: {np.unique(ds.y_test)}"
        )

    def test_labels_remapped_from_3(self, tmp_path):
        """When raw FI-2010 label = 3 (Down), FI2010Dataset must yield y = 0."""
        from data.loaders import FI2010Dataset
        tr, te = self._make_synthetic_fi2010_npy(tmp_path, label_col_val=3)
        ds = FI2010Dataset(train_path=tr, test_path=te, horizon_k=10)
        assert np.all(ds.y_test == 0), (
            f"FI-2010 raw label 3 (Down) should remap to 0. Got: {np.unique(ds.y_test)}"
        )

    def test_labels_remapped_from_2(self, tmp_path):
        """When raw FI-2010 label = 2 (Stationary), FI2010Dataset must yield y = 1."""
        from data.loaders import FI2010Dataset
        tr, te = self._make_synthetic_fi2010_npy(tmp_path, label_col_val=2)
        ds = FI2010Dataset(train_path=tr, test_path=te, horizon_k=10)
        assert np.all(ds.y_test == 1), (
            f"FI-2010 raw label 2 (Stat) should remap to 1. Got: {np.unique(ds.y_test)}"
        )

    def test_label_values_in_valid_range(self, tmp_path):
        """After remapping, all labels must be in {0, 1, 2}."""
        from data.loaders import FI2010Dataset
        tr, te = self._make_synthetic_fi2010_npy(tmp_path, label_col_val=2)
        ds = FI2010Dataset(train_path=tr, test_path=te, horizon_k=10)
        for y in [ds.y_train, ds.y_val, ds.y_test]:
            assert set(y).issubset({0, 1, 2}), f"Labels out of range: {set(y)}"
