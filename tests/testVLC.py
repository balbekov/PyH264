"""
VLC Module Unit Tests

Tests for Variable Length Coding (Exp-Golomb) implementation.
"""

import numpy as np
import unittest

import h264.VLC as VLC


class TestExpGolomb(unittest.TestCase):
    """Unit tests for Exponential-Golomb coding."""

    def setUp(self):
        """Initialize LUT before tests."""
        VLC.init_LUT()

    def test_lut_initialization(self):
        """Test that LUTs are properly initialized."""
        self.assertGreater(len(VLC.golomb_enc_LUT), 0,
                          "Encoding LUT should not be empty")
        self.assertGreater(len(VLC.golomb_dec_LUT), 0,
                          "Decoding LUT should not be empty")

    def test_encode_zero(self):
        """Test encoding of zero."""
        self.assertEqual(VLC.golomb_enc_LUT[0], '1',
                        "Zero should encode to '1'")

    def test_encode_positive_integers(self):
        """Test encoding of positive integers."""
        # Check specific known values
        # For exp-golomb: code_num = 2*n - 1 for positive n
        # Then encoded as: (leadingZeros) + 1 + (suffix)
        
        # 1 -> code_num=1 -> binary 10 -> 010
        self.assertIn('1', VLC.golomb_enc_LUT[1])
        
        # Verify all positive values have valid encodings
        for i in range(1, 100):
            code = VLC.golomb_enc_LUT[i]
            self.assertIsInstance(code, str)
            self.assertTrue(all(c in '01' for c in code),
                           f"Code for {i} should be binary string")

    def test_encode_negative_integers(self):
        """Test encoding of negative integers."""
        # For exp-golomb: code_num = -2*n for negative n
        
        for i in range(-100, 0):
            code = VLC.golomb_enc_LUT[i]
            self.assertIsInstance(code, str)
            self.assertTrue(all(c in '01' for c in code),
                           f"Code for {i} should be binary string")

    def test_decode_inverse_of_encode(self):
        """Test that decode is inverse of encode."""
        for value in range(-100, 101):
            encoded = VLC.golomb_enc_LUT[value]
            decoded = VLC.golomb_dec_LUT[encoded]
            self.assertEqual(decoded, value,
                           f"Decode should be inverse of encode for {value}")

    def test_prefix_free_codes(self):
        """Test that codes are prefix-free (no code is prefix of another)."""
        codes = list(VLC.golomb_enc_LUT.values())
        
        for i, code1 in enumerate(codes[:100]):  # Check subset for speed
            for code2 in codes[:100]:
                if code1 != code2:
                    self.assertFalse(code2.startswith(code1) and len(code2) > len(code1),
                                    f"'{code1}' should not be prefix of '{code2}'")

    def test_expgolomb_enc_block(self):
        """Test encoding a 4x4 block."""
        block = np.array([[0, 1, -1, 2],
                          [-2, 3, -3, 4],
                          [-4, 5, -5, 6],
                          [-6, 7, -7, 0]], dtype=np.int16)
        
        vlc = VLC.expgolomb_enc(4, block)
        
        self.assertIsInstance(vlc, str)
        self.assertGreater(len(vlc), 0)
        self.assertTrue(all(c in '01' for c in vlc))

    def test_expgolomb_dec_block(self):
        """Test decoding a VLC back to block."""
        original = np.array([[0, 1, -1, 2],
                             [-2, 3, -3, 4],
                             [-4, 5, -5, 6],
                             [-6, 7, -7, 0]], dtype=np.int16)
        
        vlc = VLC.expgolomb_enc(4, original)
        result, remaining, decoded = VLC.expgolomb_dec(4, vlc)
        
        self.assertEqual(result, 1, "Decoding should succeed")
        self.assertTrue(np.array_equal(decoded, original),
                       "Decoded block should match original")

    def test_expgolomb_roundtrip_zero_block(self):
        """Test roundtrip with all-zero block."""
        zero_block = np.zeros((4, 4), dtype=np.int16)
        
        vlc = VLC.expgolomb_enc(4, zero_block)
        result, remaining, decoded = VLC.expgolomb_dec(4, vlc)
        
        self.assertEqual(result, 1)
        self.assertTrue(np.array_equal(decoded, zero_block))

    def test_expgolomb_returns_remaining(self):
        """Test that decoder returns remaining bits correctly."""
        block = np.array([[1, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]], dtype=np.int16)
        
        vlc = VLC.expgolomb_enc(4, block)
        extra_bits = "1010101"
        vlc_with_extra = vlc + extra_bits
        
        result, remaining, decoded = VLC.expgolomb_dec(4, vlc_with_extra)
        
        self.assertEqual(result, 1)
        self.assertEqual(remaining, extra_bits,
                        "Decoder should return unused bits")

    def test_count_bits(self):
        """Test count_bits helper function."""
        # count_bits returns N-1 zeros for N-bit value
        self.assertEqual(VLC.count_bits("1"), "")      # 1 bit -> 0 zeros
        self.assertEqual(VLC.count_bits("10"), "0")    # 2 bits -> 1 zero
        self.assertEqual(VLC.count_bits("100"), "00")  # 3 bits -> 2 zeros
        self.assertEqual(VLC.count_bits("1000"), "000") # 4 bits -> 3 zeros


class TestVLCEdgeCases(unittest.TestCase):
    """Edge case tests for VLC module."""

    def setUp(self):
        """Initialize LUT before tests."""
        VLC.init_LUT()

    def test_large_positive_value(self):
        """Test encoding large positive values."""
        large_val = 1000
        code = VLC.golomb_enc_LUT[large_val]
        
        self.assertIsInstance(code, str)
        decoded = VLC.golomb_dec_LUT[code]
        self.assertEqual(decoded, large_val)

    def test_large_negative_value(self):
        """Test encoding large negative values."""
        large_neg = -1000
        code = VLC.golomb_enc_LUT[large_neg]
        
        self.assertIsInstance(code, str)
        decoded = VLC.golomb_dec_LUT[code]
        self.assertEqual(decoded, large_neg)

    def test_decode_error_handling(self):
        """Test decoder handles invalid VLC gracefully."""
        # Very long invalid sequence should trigger error handling
        invalid_vlc = "0" * 200  # No valid code starts with this many zeros
        
        result, remaining, block = VLC.expgolomb_dec(4, invalid_vlc)
        
        # Should return error code
        self.assertEqual(result, -1)

    def test_empty_vlc_handling(self):
        """Test decoder handles empty/short VLC."""
        empty_vlc = ""
        
        result, remaining, block = VLC.expgolomb_dec(4, empty_vlc)
        
        # Should return error (VLC underrun)
        self.assertEqual(result, -1)


class TestCodeEfficiency(unittest.TestCase):
    """Tests for coding efficiency properties."""

    def setUp(self):
        """Initialize LUT before tests."""
        VLC.init_LUT()

    def test_small_values_short_codes(self):
        """Test that smaller values have shorter codes."""
        # Zero should have shortest code
        zero_len = len(VLC.golomb_enc_LUT[0])
        one_len = len(VLC.golomb_enc_LUT[1])
        neg_one_len = len(VLC.golomb_enc_LUT[-1])
        
        self.assertLessEqual(zero_len, one_len)
        self.assertLessEqual(zero_len, neg_one_len)

    def test_code_length_growth(self):
        """Test that code length grows appropriately with value magnitude."""
        prev_len = 0
        for mag in [0, 1, 3, 7, 15, 31]:
            curr_len = len(VLC.golomb_enc_LUT[mag])
            self.assertGreaterEqual(curr_len, prev_len,
                                   "Code length should not decrease for larger values")
            prev_len = curr_len


if __name__ == '__main__':
    unittest.main(verbosity=2)

