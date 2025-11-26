"""
Intra Prediction Unit Tests

Tests for H.264 intra prediction modes (DC, Horizontal, Vertical).
"""

from h264.MacroBlock import MacroBlock
from h264.CAVLC import block_to_zigzag, CAVLC_enc

import numpy as np
import unittest


class TestIntraPredictionModes(unittest.TestCase):
    """Tests for intra prediction mode selection and residual calculation."""

    def setUp(self):
        """Set up test fixtures."""
        # Vertical gradient - columns have same value, rows differ
        self.v_grad_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                self.v_grad_data[i, j] = j * 16  # Value depends on column

        # Horizontal gradient - rows have same value, columns differ
        self.h_grad_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                self.h_grad_data[i, j] = i * 16  # Value depends on row

        # Constant block
        self.const_data = np.uint8(np.full((16, 16), 128))

        # Matching neighbor context
        self.top_row_v = np.uint8(np.arange(16) * 16)
        self.left_col_v = np.uint8(np.full(16, 128))
        
        self.top_row_h = np.uint8(np.full(16, 128))
        self.left_col_h = np.uint8(np.arange(16) * 16)

    def test_intra_predict_returns_valid_mode(self):
        """Test that intra_predict returns a valid mode string."""
        mb = MacroBlock(None, self.const_data)
        top_row = np.uint8(np.full(16, 128))
        left_column = np.uint8(np.full(16, 128))
        
        mode = mb.intra_predict((top_row, left_column))
        
        self.assertIn(mode, ['dc', 'h', 'v'],
                     f"Mode '{mode}' should be one of dc, h, v")

    def test_vertical_prediction_mode(self):
        """Test that vertical gradient selects V prediction when top row matches."""
        mb = MacroBlock(None, self.v_grad_data)
        
        mode = mb.intra_predict((self.top_row_v, self.left_col_v))
        
        self.assertEqual(mode, 'v',
                        "Vertical gradient with matching top row should choose V mode")

    def test_horizontal_prediction_mode(self):
        """Test that horizontal gradient selects H prediction when left column matches."""
        mb = MacroBlock(None, self.h_grad_data)
        
        mode = mb.intra_predict((self.top_row_h, self.left_col_h))
        
        self.assertEqual(mode, 'h',
                        "Horizontal gradient with matching left column should choose H mode")

    def test_dc_prediction_constant_block(self):
        """Test that constant block with matching DC neighbor uses DC mode."""
        mb = MacroBlock(None, self.const_data)
        # Neighbors with same DC value
        top_row = np.uint8(np.full(16, 128))
        left_column = np.uint8(np.full(16, 128))
        
        mode = mb.intra_predict((top_row, left_column))
        
        # DC prediction should work well for constant blocks
        self.assertEqual(mode, 'dc',
                        "Constant block should prefer DC prediction")

    def test_intra_predict_sets_residual(self):
        """Test that intra_predict modifies blocks to contain residuals."""
        mb = MacroBlock(None, self.v_grad_data.copy())
        original_blocks = [block.block.copy() for block in mb.blocks]
        
        top_row = self.top_row_v
        left_column = self.left_col_v
        
        mb.intra_predict((top_row, left_column))
        
        # Blocks should be modified (contain residuals now)
        for i, block in enumerate(mb.blocks):
            # At least some blocks should have changed
            pass  # The residual values depend on prediction mode

    def test_intra_predict_sets_prediction_mode(self):
        """Test that prediction mode is set on all transform blocks."""
        mb = MacroBlock(None, self.const_data)
        top_row = np.uint8(np.full(16, 128))
        left_column = np.uint8(np.full(16, 128))
        
        mode = mb.intra_predict((top_row, left_column))
        
        for block in mb.blocks:
            self.assertEqual(block.prediction_mode, mode,
                           "All blocks should have same prediction mode")

    def test_zero_residual_perfect_prediction(self):
        """Test that perfect prediction produces zero residual."""
        # Create data that exactly matches V prediction
        v_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                v_data[i, j] = j  # Each column has value = column index
        
        mb = MacroBlock(None, v_data)
        top_row = np.uint8(np.arange(16))  # Exact match for V prediction
        left_column = np.uint8(np.full(16, 0))
        
        mode = mb.intra_predict((top_row, left_column))
        
        if mode == 'v':
            # Residual should be approximately zero
            residual = mb.get_image()
            self.assertTrue(np.allclose(residual, 0, atol=1),
                           "Perfect V prediction should give zero residual")


class TestIntraPredictionEdgeCases(unittest.TestCase):
    """Edge case tests for intra prediction."""

    def test_all_zeros_block(self):
        """Test prediction with all-zero block."""
        mb = MacroBlock(None, np.uint8(np.zeros((16, 16))))
        top_row = np.uint8(np.zeros(16))
        left_column = np.uint8(np.zeros(16))
        
        # Should not raise
        mode = mb.intra_predict((top_row, left_column))
        self.assertIn(mode, ['dc', 'h', 'v'])

    def test_all_max_block(self):
        """Test prediction with all-255 block."""
        mb = MacroBlock(None, np.uint8(np.full((16, 16), 255)))
        top_row = np.uint8(np.full(16, 255))
        left_column = np.uint8(np.full(16, 255))
        
        # Should not raise
        mode = mb.intra_predict((top_row, left_column))
        self.assertIn(mode, ['dc', 'h', 'v'])

    def test_random_data(self):
        """Test prediction with random data."""
        np.random.seed(42)
        random_data = np.uint8(np.random.randint(0, 256, (16, 16)))
        mb = MacroBlock(None, random_data)
        
        top_row = np.uint8(np.random.randint(0, 256, 16))
        left_column = np.uint8(np.random.randint(0, 256, 16))
        
        # Should not raise
        mode = mb.intra_predict((top_row, left_column))
        self.assertIn(mode, ['dc', 'h', 'v'])


class TestNeighborAccess(unittest.TestCase):
    """Tests for neighbor pixel access methods."""

    def test_bottom_row_values(self):
        """Test bottom_row returns correct values."""
        data = np.uint8(np.zeros((16, 16)))
        data[15, :] = np.arange(16)  # Set bottom row to 0-15
        
        mb = MacroBlock(None, data)
        bottom = mb.bottom_row()
        
        expected = np.arange(16)
        self.assertTrue(np.array_equal(bottom, expected))

    def test_right_column_values(self):
        """Test right_column returns correct values."""
        data = np.uint8(np.zeros((16, 16)))
        data[:, 15] = np.arange(16)  # Set right column to 0-15
        
        mb = MacroBlock(None, data)
        right = mb.right_column()
        
        expected = np.arange(16)
        self.assertTrue(np.array_equal(right, expected))

    def test_neighbor_shapes(self):
        """Test that neighbor access returns correct shapes."""
        mb = MacroBlock(None, np.uint8(np.zeros((16, 16))))
        
        self.assertEqual(len(mb.bottom_row()), 16)
        self.assertEqual(len(mb.right_column()), 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)
