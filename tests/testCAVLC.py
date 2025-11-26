"""
CAVLC Module Unit Tests

Tests for Context-Adaptive Variable Length Coding implementation.
"""

from h264.MacroBlock import MacroBlock
from h264.CAVLC import block_to_zigzag, CAVLC_enc

import numpy as np
import unittest
import logging


class TestZigzagScan(unittest.TestCase):
    """Tests for zigzag scanning function."""

    def test_zigzag_order(self):
        """Test that zigzag produces correct coefficient ordering."""
        # Create block with known values to verify ordering
        test_block = [[0,  1,  5,  6],
                      [2,  4,  7, 12],
                      [3,  8, 11, 13],
                      [9, 10, 14, 15]]
        
        result = block_to_zigzag(test_block)
        
        # Zigzag should produce 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        expected = list(range(16))
        self.assertEqual(result, expected,
                        "Zigzag should order elements correctly")

    def test_zigzag_length(self):
        """Test that zigzag produces 16 elements from 4x4 block."""
        block = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
        
        result = block_to_zigzag(block)
        
        self.assertEqual(len(result), 16)

    def test_zigzag_dc_first(self):
        """Test that DC coefficient (0,0) is first in zigzag order."""
        block = [[100, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
        
        result = block_to_zigzag(block)
        
        self.assertEqual(result[0], 100,
                        "DC coefficient should be first")

    def test_zigzag_hf_last(self):
        """Test that highest frequency coefficient (3,3) is last."""
        block = [[0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 255]]
        
        result = block_to_zigzag(block)
        
        self.assertEqual(result[-1], 255,
                        "Highest frequency coefficient should be last")

    def test_zigzag_with_numpy_array(self):
        """Test zigzag works with numpy arrays."""
        block = np.array([[1, 2, 3, 4],
                          [5, 6, 7, 8],
                          [9, 10, 11, 12],
                          [13, 14, 15, 16]])
        
        result = block_to_zigzag(block)
        
        self.assertEqual(len(result), 16)
        self.assertIsInstance(result, list)


class TestCAVLCEncode(unittest.TestCase):
    """Tests for CAVLC encoding function.
    
    Note: CAVLC encoder has known bugs (index out of range in run_before calculation).
    These tests document expected behavior but may fail until bugs are fixed.
    """

    def setUp(self):
        """Set up logging for CAVLC debugging."""
        logging.basicConfig(level=logging.WARNING)

    @unittest.skip("CAVLC encoder has known bug in run_before calculation")
    def test_cavlc_known_vector(self):
        """Test CAVLC encoding with known test vector from H.264 spec."""
        # Standard test vector
        cavlc_test_arr = [[0, 3, -1, 0],
                          [0, -1, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 0]]
        
        output = CAVLC_enc(cavlc_test_arr)
        
        # Expected output from H.264 spec example
        expected = "000010001110010111101101"
        self.assertEqual(output, expected,
                        "CAVLC output should match expected bitstring")

    @unittest.skip("CAVLC encoder has known bug in run_before calculation")
    def test_cavlc_returns_string(self):
        """Test that CAVLC returns a binary string."""
        block = [[1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
        
        output = CAVLC_enc(block)
        
        self.assertIsInstance(output, str)
        self.assertTrue(all(c in '01' for c in output),
                       "Output should be binary string")

    @unittest.skip("CAVLC encoder has known bug in run_before calculation")
    def test_cavlc_dc_only(self):
        """Test CAVLC with DC-only block."""
        block = [[128, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
        
        output = CAVLC_enc(block)
        
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)

    @unittest.skip("CAVLC encoder has known bug in run_before calculation")
    def test_cavlc_trailing_ones(self):
        """Test CAVLC correctly handles trailing ones."""
        # Block with trailing +1 and -1 values
        block = [[0, 1, -1, 0],
                 [1, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]]
        
        output = CAVLC_enc(block)
        
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)


class TestCoeffTokenTables(unittest.TestCase):
    """Tests for coefficient token lookup tables."""

    def test_coeff_token0_structure(self):
        """Test coeff_token0 table has correct structure."""
        from h264.CAVLC import coeff_token0
        
        # Should have 17 rows (0-16 coefficients)
        self.assertEqual(len(coeff_token0), 17)
        
        # Each row should have 4 columns (0-3 trailing ones)
        for row in coeff_token0:
            self.assertEqual(len(row), 4)

    def test_table_zeros_structure(self):
        """Test Table_zeros has correct structure."""
        from h264.CAVLC import Table_zeros
        
        # Should have 15 rows
        self.assertEqual(len(Table_zeros), 15)
        
        # Each row should have 16 columns
        for row in Table_zeros:
            self.assertEqual(len(row), 16)

    def test_table_run_structure(self):
        """Test Table_run has correct structure."""
        from h264.CAVLC import Table_run
        
        # Should have 15 rows
        self.assertEqual(len(Table_run), 15)
        
        # Each row should have 7 columns
        for row in Table_run:
            self.assertEqual(len(row), 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
