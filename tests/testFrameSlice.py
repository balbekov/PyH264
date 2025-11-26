"""
Frame and Slice Module Unit Tests

Tests for Frame and Slice container classes.
"""

from h264.Frame import Frame
from h264.Slice import Slice
from h264.MacroBlock import MacroBlock

import numpy as np
import unittest


class TestSlice(unittest.TestCase):
    """Unit tests for Slice class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create 16-row slice data (one macroblock row)
        self.slice_data = np.uint8(np.zeros((16, 64)))
        for i in range(16):
            for j in range(64):
                self.slice_data[i, j] = (i * 4 + j) % 256

    def test_slice_construction_with_data(self):
        """Test Slice construction with image data."""
        slice_obj = Slice(None, self.slice_data, WIDTH=64)
        
        # Should create 4 macroblocks for 64-pixel width
        self.assertEqual(len(slice_obj.blocks), 4)
        
        for block in slice_obj.blocks:
            self.assertIsInstance(block, MacroBlock)

    def test_slice_construction_empty(self):
        """Test Slice construction without data."""
        slice_obj = Slice(None, None, WIDTH=64)
        
        # Should still create macroblocks
        self.assertEqual(len(slice_obj.blocks), 4)

    def test_slice_width_stored(self):
        """Test that slice stores width correctly."""
        slice_obj = Slice(None, None, WIDTH=128)
        
        self.assertEqual(slice_obj.width, 128)

    def test_slice_iterator(self):
        """Test that slice supports indexing."""
        slice_obj = Slice(None, self.slice_data, WIDTH=64)
        
        # Should be able to index macroblocks
        mb0 = slice_obj[0]
        self.assertIsInstance(mb0, MacroBlock)
        
        mb3 = slice_obj[3]
        self.assertIsInstance(mb3, MacroBlock)

    def test_slice_get_bits(self):
        """Test slice bitstream generation."""
        slice_obj = Slice(None, self.slice_data, WIDTH=64)
        
        bits = slice_obj.get_bits()
        
        self.assertIsInstance(bits, str)
        self.assertGreater(len(bits), 0)
        
        # Should end with slice sync marker
        self.assertTrue(bits.endswith('000000001'),
                       "Slice should end with sync marker")

    def test_slice_set_bits(self):
        """Test slice bitstream decoding."""
        # Encode a slice
        slice_enc = Slice(None, self.slice_data, WIDTH=64)
        bits = slice_enc.get_bits()
        
        # Decode into new slice
        slice_dec = Slice(None, None, WIDTH=64)
        remaining = slice_dec.set_bits(bits)
        
        # Should consume most/all bits
        self.assertLess(len(remaining), len(bits))

    def test_slice_str(self):
        """Test slice string representation."""
        slice_obj = Slice(None, self.slice_data, WIDTH=64)
        
        # Should not raise
        str_repr = str(slice_obj)
        self.assertIsInstance(str_repr, str)


class TestFrame(unittest.TestCase):
    """Unit tests for Frame class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create small frame for testing
        self.frame_data = np.uint8(np.zeros((64, 64)))
        for i in range(64):
            for j in range(64):
                self.frame_data[i, j] = (i + j) % 256

    def test_frame_construction_with_data(self):
        """Test Frame construction with image data."""
        frame = Frame(None, self.frame_data, WIDTH=64, HEIGHT=64)
        
        # 64 pixels / 16 per slice = 4 slices
        self.assertEqual(len(frame.slices), 4)
        
        self.assertEqual(frame.width, 64)
        self.assertEqual(frame.height, 64)

    def test_frame_construction_empty(self):
        """Test Frame construction without data."""
        frame = Frame(None, None, WIDTH=64, HEIGHT=64)
        
        # Should still create slices
        self.assertEqual(len(frame.slices), 4)

    def test_frame_default_dimensions(self):
        """Test Frame with default dimensions."""
        frame = Frame(None, None)
        
        self.assertEqual(frame.width, 1280)
        self.assertEqual(frame.height, 720)

    def test_frame_custom_dimensions(self):
        """Test Frame with custom dimensions."""
        frame = Frame(None, None, WIDTH=320, HEIGHT=240)
        
        self.assertEqual(frame.width, 320)
        self.assertEqual(frame.height, 240)
        # 240 / 16 = 15 slices
        self.assertEqual(len(frame.slices), 15)

    def test_frame_get_image(self):
        """Test frame image reconstruction."""
        frame = Frame(None, self.frame_data, WIDTH=64, HEIGHT=64)
        
        recovered = frame.get_image()
        
        self.assertEqual(recovered.shape, (64, 64))
        self.assertEqual(recovered.dtype, np.uint8)
        
        # Should approximately match original
        self.assertTrue(np.array_equal(recovered, self.frame_data),
                       "get_image should return original data")

    def test_frame_get_bits(self):
        """Test frame bitstream generation."""
        frame = Frame(None, self.frame_data, WIDTH=64, HEIGHT=64)
        
        bits = frame.get_bits()
        
        self.assertIsInstance(bits, str)
        self.assertGreater(len(bits), 0)
        self.assertTrue(all(c in '01' for c in bits))

    def test_frame_set_bits(self):
        """Test frame bitstream decoding."""
        # Encode frame
        frame_enc = Frame(None, self.frame_data, WIDTH=64, HEIGHT=64)
        bits = frame_enc.get_bits()
        
        # Decode into new frame
        frame_dec = Frame(None, None, WIDTH=64, HEIGHT=64)
        frame_dec.set_bits(bits)
        
        # Should have same structure
        self.assertEqual(len(frame_dec.slices), len(frame_enc.slices))

    def test_frame_slice_iterator(self):
        """Test iterating over frame slices."""
        frame = Frame(None, self.frame_data, WIDTH=64, HEIGHT=64)
        
        slice_count = 0
        for slice_obj in frame.slices:
            self.assertIsInstance(slice_obj, Slice)
            slice_count += 1
        
        self.assertEqual(slice_count, 4)


class TestFrameSliceIntegration(unittest.TestCase):
    """Integration tests for Frame and Slice interaction."""

    def test_frame_slice_macroblock_hierarchy(self):
        """Test correct hierarchy: Frame -> Slice -> MacroBlock."""
        frame_data = np.uint8(np.zeros((32, 32)))
        frame = Frame(None, frame_data, WIDTH=32, HEIGHT=32)
        
        # 2 slices (32/16)
        self.assertEqual(len(frame.slices), 2)
        
        for slice_obj in frame.slices:
            # 2 macroblocks per slice (32/16)
            self.assertEqual(len(slice_obj.blocks), 2)
            
            for mb in slice_obj.blocks:
                # 16 transform blocks per macroblock
                self.assertEqual(len(mb.blocks), 16)

    def test_roundtrip_preservation(self):
        """Test that encode/decode roundtrip preserves data approximately."""
        # Simple pattern
        frame_data = np.uint8(np.zeros((32, 32)))
        for i in range(32):
            for j in range(32):
                frame_data[i, j] = 128  # Constant value
        
        frame_enc = Frame(None, frame_data, WIDTH=32, HEIGHT=32)
        bits = frame_enc.get_bits()
        
        frame_dec = Frame(None, None, WIDTH=32, HEIGHT=32)
        frame_dec.set_bits(bits)
        
        recovered = frame_dec.get_image()
        
        # Check data is approximately preserved
        diff = np.abs(frame_data.astype(int) - recovered.astype(int))
        self.assertLess(diff.mean(), 20,
                       "Average difference should be small")


class TestSliceEdgeCases(unittest.TestCase):
    """Edge case tests for Slice class."""

    def test_single_macroblock_slice(self):
        """Test slice with single macroblock."""
        data = np.uint8(np.zeros((16, 16)))
        slice_obj = Slice(None, data, WIDTH=16)
        
        self.assertEqual(len(slice_obj.blocks), 1)

    def test_wide_slice(self):
        """Test slice with many macroblocks."""
        data = np.uint8(np.zeros((16, 256)))
        slice_obj = Slice(None, data, WIDTH=256)
        
        self.assertEqual(len(slice_obj.blocks), 16)


class TestFrameEdgeCases(unittest.TestCase):
    """Edge case tests for Frame class."""

    def test_single_slice_frame(self):
        """Test frame with single slice."""
        data = np.uint8(np.zeros((16, 32)))
        frame = Frame(None, data, WIDTH=32, HEIGHT=16)
        
        self.assertEqual(len(frame.slices), 1)

    def test_tall_frame(self):
        """Test frame with many slices."""
        data = np.uint8(np.zeros((256, 32)))
        frame = Frame(None, data, WIDTH=32, HEIGHT=256)
        
        self.assertEqual(len(frame.slices), 16)


if __name__ == '__main__':
    unittest.main(verbosity=2)

