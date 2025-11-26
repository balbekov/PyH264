"""
H264 Integration Tests

Tests for the complete H264 encoder/decoder pipeline including
TransformBlock, MacroBlock, and full frame processing.
"""

from h264.MacroBlock import MacroBlock
from h264.TransformBlock import TransformBlock
from h264.CAVLC import block_to_zigzag, CAVLC_enc
from h264 import H264 as codec

import numpy as np
from numpy import empty, isclose, array_equal
import logging
import unittest
import os


class TestTransformBlock(unittest.TestCase):
    """Unit tests for TransformBlock DCT/IDCT and quantization."""

    def setUp(self):
        """Set up test fixtures."""
        # Gradient pattern - good for testing DCT behavior
        self.grad_data = np.uint8(np.zeros((4, 4)))
        for i in range(4):
            for j in range(4):
                self.grad_data[i, j] = 100 - 8*i

        # Random data for general testing
        np.random.seed(42)  # Reproducible tests
        self.rand_data = np.uint8(np.random.randint(0, 256, (4, 4)))

        # Constant block - DC only
        self.const_data = np.uint8(np.full((4, 4), 128))

        # Checkerboard - high frequency content
        self.checker_data = np.uint8(np.zeros((4, 4)))
        for i in range(4):
            for j in range(4):
                self.checker_data[i, j] = 255 if (i + j) % 2 == 0 else 0

    def test_dct_idct_roundtrip(self):
        """Test that DCT followed by IDCT returns original data."""
        tb = TransformBlock(None, self.rand_data.copy())
        original = tb.block.copy()
        
        tb.dct()
        self.assertEqual(tb.state, "Frequency")
        
        tb.idct()
        self.assertEqual(tb.state, "Spatial")
        
        # Should be close to original (minor rounding errors expected)
        self.assertTrue(isclose(tb.block, original, rtol=1).all(),
                       "DCT/IDCT roundtrip should preserve data")

    def test_dct_idct_gradient(self):
        """Test DCT/IDCT with gradient pattern."""
        tb = TransformBlock(None, self.grad_data.copy())
        original = tb.block.copy()
        
        tb.dct()
        tb.idct()
        
        self.assertTrue(isclose(tb.block, original, rtol=1).all(),
                       "Gradient pattern should survive DCT/IDCT")

    def test_dct_idct_constant(self):
        """Test DCT/IDCT with constant block (DC only)."""
        tb = TransformBlock(None, self.const_data.copy())
        original = tb.block.copy()
        
        tb.dct()
        tb.idct()
        
        self.assertTrue(isclose(tb.block, original, atol=1).all(),
                       "Constant block should survive DCT/IDCT")

    def test_quantize_dequantize_roundtrip(self):
        """Test quantization/dequantization introduces acceptable loss."""
        tb = TransformBlock(None, self.grad_data.copy())
        original = tb.block.copy()
        
        tb.dct()
        tb.quantize()
        self.assertEqual(tb.state, "Quantized")
        
        tb.dequantize()
        self.assertEqual(tb.state, "Frequency")
        
        tb.idct()
        
        # Allow some loss due to quantization
        self.assertTrue(isclose(tb.block, original, rtol=5, atol=5).all(),
                       "Quantization should not destroy data completely")

    def test_vlc_roundtrip(self):
        """Test VLC encoding and decoding roundtrip."""
        tb = TransformBlock(None, self.rand_data.copy())
        original = tb.block.copy()
        
        vlc = tb.get_vlc()
        self.assertIsInstance(vlc, str)
        self.assertGreater(len(vlc), 0, "VLC should produce output")
        
        # Decode into new block
        tb2 = TransformBlock(None, np.zeros((4, 4)))
        tb2.set_vlc(vlc)
        
        # Should be close after lossy compression
        self.assertTrue(isclose(tb2.block, original, rtol=5, atol=5).all(),
                       "VLC roundtrip should preserve data approximately")

    def test_state_machine_enforcement(self):
        """Test that operations enforce correct state transitions."""
        tb = TransformBlock(None, self.rand_data.copy())
        
        # Can't IDCT from Spatial state
        with self.assertRaises(ValueError):
            tb.idct()
        
        tb.dct()
        
        # Can't DCT from Frequency state
        with self.assertRaises(ValueError):
            tb.dct()
        
        # Can't dequantize from Frequency state
        with self.assertRaises(ValueError):
            tb.dequantize()


class TestMacroBlock(unittest.TestCase):
    """Unit tests for MacroBlock operations."""

    def setUp(self):
        """Set up test fixtures."""
        # 16x16 stripe pattern
        self.stripe_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                self.stripe_data[i, j] = i * 16 + j

        # Constant macroblock
        self.const_data = np.uint8(np.full((16, 16), 128))

        # Gradient macroblock
        self.grad_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                self.grad_data[i, j] = i * 8

    def test_macroblock_construction(self):
        """Test MacroBlock correctly partitions into 16 TransformBlocks."""
        mb = MacroBlock(None, self.stripe_data)
        
        self.assertEqual(len(mb.blocks), 16, 
                        "MacroBlock should contain 16 TransformBlocks")
        
        for block in mb.blocks:
            self.assertIsInstance(block, TransformBlock)
            self.assertEqual(block.block.shape, (4, 4))

    def test_macroblock_image_roundtrip(self):
        """Test get_image returns original data."""
        mb = MacroBlock(None, self.stripe_data)
        recovered = mb.get_image()
        
        self.assertTrue(array_equal(self.stripe_data, recovered),
                       "MacroBlock get_image should return original data")

    def test_macroblock_vlc_roundtrip(self):
        """Test VLC encoding/decoding of entire macroblock."""
        mb = MacroBlock(None, self.stripe_data)
        
        vlc = mb.get_vlc()
        self.assertIsInstance(vlc, str)
        self.assertGreater(len(vlc), 0)
        
        mb2 = MacroBlock(None, None)
        remaining = mb2.set_vlc(vlc)
        
        # Data should be approximately preserved
        recovered = mb2.get_image()
        # Due to quantization, we allow significant tolerance
        diff = np.abs(self.stripe_data.astype(int) - recovered.astype(int))
        self.assertLess(diff.mean(), 30, 
                       "Average difference should be reasonable after VLC roundtrip")

    def test_bottom_row(self):
        """Test bottom_row accessor returns correct pixels."""
        mb = MacroBlock(None, self.stripe_data)
        bottom = mb.bottom_row()
        
        self.assertEqual(len(bottom), 16)
        # Bottom row of stripe_data is row 15
        expected = self.stripe_data[15, :]
        self.assertTrue(array_equal(bottom, expected))

    def test_right_column(self):
        """Test right_column accessor returns correct pixels."""
        mb = MacroBlock(None, self.stripe_data)
        right = mb.right_column()
        
        self.assertEqual(len(right), 16)
        # Right column of stripe_data is column 15
        expected = self.stripe_data[:, 15]
        self.assertTrue(array_equal(right, expected))

    def test_dct_all_blocks(self):
        """Test DCT operation on all blocks in macroblock."""
        mb = MacroBlock(None, self.const_data)
        mb.dct()
        
        for block in mb.blocks:
            self.assertEqual(block.state, "Frequency")

    def test_idct_all_blocks(self):
        """Test IDCT operation on all blocks in macroblock."""
        mb = MacroBlock(None, self.const_data)
        mb.dct()
        mb.idct()
        
        for block in mb.blocks:
            self.assertEqual(block.state, "Spatial")


class TestIntraPrediction(unittest.TestCase):
    """Tests for intra prediction modes."""

    def setUp(self):
        """Set up test fixtures for intra prediction."""
        # Vertical gradient - should favor V prediction
        self.v_grad_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                self.v_grad_data[i, j] = j * 16

        # Horizontal gradient - should favor H prediction
        self.h_grad_data = np.uint8(np.zeros((16, 16)))
        for i in range(16):
            for j in range(16):
                self.h_grad_data[i, j] = i * 16

        # Constant - should favor DC prediction
        self.const_data = np.uint8(np.full((16, 16), 128))

    def test_intra_predict_returns_mode(self):
        """Test intra_predict returns a valid prediction mode."""
        mb = MacroBlock(None, self.const_data)
        top_row = np.uint8(np.full(16, 128))
        left_column = np.uint8(np.full(16, 128))
        
        mode = mb.intra_predict((top_row, left_column))
        
        self.assertIn(mode, ['dc', 'h', 'v'],
                     "Prediction mode should be dc, h, or v")

    def test_intra_predict_vertical_gradient(self):
        """Test that vertical gradient data chooses V prediction."""
        mb = MacroBlock(None, self.v_grad_data)
        top_row = np.uint8(np.arange(16) * 16)  # Matches column pattern
        left_column = np.uint8(np.full(16, 128))
        
        mode = mb.intra_predict((top_row, left_column))
        
        # V prediction should produce smallest residual for vertical gradient
        self.assertEqual(mode, 'v',
                        "Vertical gradient should choose V prediction")


class TestH264Integration(unittest.TestCase):
    """Integration tests for H264 encoder/decoder."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level resources."""
        cls.test_file_path = "sequences/test.y4m"
        cls.has_test_file = os.path.exists(cls.test_file_path)

    def test_h264_initialization(self):
        """Test H264 object initialization."""
        h264 = codec.H264()
        
        self.assertEqual(h264.frame_counter, 0)
        self.assertEqual(len(h264.frames), 0)
        self.assertEqual(h264.width, 1280)
        self.assertEqual(h264.height, 720)

    def test_h264_custom_dimensions(self):
        """Test H264 with custom dimensions."""
        h264 = codec.H264(width=640, height=480)
        
        self.assertEqual(h264.width, 640)
        self.assertEqual(h264.height, 480)

    def test_load_pattern(self):
        """Test loading a test pattern."""
        h264 = codec.H264(width=64, height=64)  # Small for speed
        h264.load_pattern()
        
        self.assertEqual(len(h264.frames), 1)

    def test_load_pattern_custom(self):
        """Test loading pattern with custom generator."""
        h264 = codec.H264(width=64, height=64)
        h264.load_pattern(lambda x: x % 128)
        
        self.assertEqual(len(h264.frames), 1)

    def test_compress_inplace(self):
        """Test in-place compression."""
        h264 = codec.H264(width=64, height=64)
        h264.load_pattern()
        
        # Should not raise
        h264.compress_inplace()
        
        # Verify blocks are in quantized state
        for frame in h264.frames:
            for slice_obj in frame.slices:
                for mb in slice_obj.blocks:
                    for tb in mb.blocks:
                        self.assertEqual(tb.state, "Quantized")

    def test_decompress_inplace(self):
        """Test in-place decompression."""
        h264 = codec.H264(width=64, height=64)
        h264.load_pattern()
        
        h264.compress_inplace()
        h264.decompress_inplace()
        
        # Verify blocks are back in spatial state
        for frame in h264.frames:
            for slice_obj in frame.slices:
                for mb in slice_obj.blocks:
                    for tb in mb.blocks:
                        self.assertEqual(tb.state, "Spatial")

    def test_compress_frame_to_bitstream(self):
        """Test generating bitstream from frame."""
        h264 = codec.H264(width=64, height=64)
        h264.load_pattern()
        
        bitstream = h264.compress_frame(0)
        
        self.assertIsInstance(bitstream, str)
        self.assertGreater(len(bitstream), 0)
        # Bitstream should only contain '0' and '1'
        self.assertTrue(all(c in '01' for c in bitstream),
                       "Bitstream should only contain binary digits")

    def test_bitstream_roundtrip(self):
        """Test encoding to bitstream and decoding back."""
        h264_enc = codec.H264(width=64, height=64)
        h264_enc.load_pattern()
        
        bitstream = h264_enc.compress_frame(0)
        
        h264_dec = codec.H264(width=64, height=64)
        h264_dec.load_bitstream(bitstream)
        
        self.assertEqual(len(h264_dec.frames), 1)

    @unittest.skipUnless(os.path.exists("sequences/test.y4m"), 
                        "Test Y4M file not available")
    def test_load_y4m_video(self):
        """Test loading Y4M video file."""
        with open(self.test_file_path, "rb") as f:
            h264 = codec.H264()
            h264.load_video(f)
            
            self.assertEqual(len(h264.frames), 1)
            self.assertGreater(h264.width, 0)
            self.assertGreater(h264.height, 0)


class TestEdgeCases(unittest.TestCase):
    """Edge case and boundary condition tests."""

    def test_zero_block(self):
        """Test handling of all-zero block."""
        zero_data = np.uint8(np.zeros((4, 4)))
        tb = TransformBlock(None, zero_data)
        
        tb.dct()
        tb.quantize()
        vlc = tb.vlc_enc()
        
        self.assertIsInstance(vlc, str)

    def test_max_value_block(self):
        """Test handling of all-255 block."""
        max_data = np.uint8(np.full((4, 4), 255))
        tb = TransformBlock(None, max_data)
        
        tb.dct()
        tb.quantize()
        vlc = tb.vlc_enc()
        
        self.assertIsInstance(vlc, str)

    def test_empty_macroblock(self):
        """Test creating macroblock with None data."""
        mb = MacroBlock(None, None)
        
        self.assertEqual(len(mb.blocks), 16)
        # All blocks should be zero-initialized
        for block in mb.blocks:
            self.assertTrue(np.all(block.block == 0))


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
