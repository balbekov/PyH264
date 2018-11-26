from h264.MacroBlock import *
from h264.CAVLC import *
from h264 import H264 as codec

from numpy import *
import logging

import unittest

class H264IntegrationTests(unittest.TestCase):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    yuv_file = open("sequences/test.y4m", "rb")

    grad_data = uint8(empty((4, 4)))
    for i in range(4):
        for j in range(4):
            grad_data[i, j] = 100 - 8*i

    rand_data = uint8(empty((4, 4)))
    for i in range(4):
        for j in range(4):
            rand_data[i, j] = random.randint(0, 255)

    stripe_data = uint8(empty((16, 16)))
    for i in range(16):
        for j in range(16):
            stripe_data[i, j] = i*16 + j

    def testTransformBlock(self):

        # TransformBlock test, test to check DCT IDCT function
        print("Testing DCT IDCT")
        tb = TransformBlock(None, self.rand_data)
        tb.dct()
        tb.idct()
        assert isclose(tb.block, self.rand_data, rtol = 1).all()

    def testQuantizer(self):
        # TransformBlock test, test to check quantizer function
        print("Testing quantizer")

        tb = TransformBlock(None, self.grad_data)
        #tb.block = self.grad_data
        tb.dct()
        tb.quantize()
        tb.dequantize()
        tb.idct()
        assert isclose(tb.block, self.grad_data, rtol=5, atol=5).all()

    def testVLCiVLC(self):
        
        tb = TransformBlock(None, self.rand_data)
        # TransformBlock test, test to check VLC iVLC function
        print("Testing VLC iVLC")
        vlc = tb.get_vlc()
        #assert len(vlc) < 30 # check for common decency; coding DC value should be cheap
        tb.set_vlc(vlc)
        assert isclose(tb.block, self.rand_data, rtol=5, atol=5).all()

    def testMacroblock(self):

        # MacroBlock test; test that constructor and accessor are correct
        print("Testing macroblock")

        print(rand_data)
        mb = MacroBlock(None, self.rand_data)
        mb_image_data = mb.get_image()

        print(mb_image_data)

        assert array_equal(self.rand_data, mb_image_data)

    def testTransformBlockVLC(self):
        # Test bitstream generation
        print("Testing 4x4 Transformblock VLC coding")

        tb = TransformBlock(None, self.stripe_data)
        tb_frombits = TransformBlock(None, None)
        tb.dct()
        tb.quantize()
        tb = tb
        tb_frombits.set_vlc(tb.get_vlc())
        tb_image_data = tb_frombits.block
        #assert array_equal(tb_data, tb_image_data)

    def testMacroblockVLC(self):
        print("Testing Macroblock VLC coding")
        mb_frombits = MacroBlock(None, None)
        mb_frombits.set_vlc(mb.get_vlc())
        mb_image_data = mb_frombits.get_image()
        #assert array_equal(mb_data, mb_image_data)

# Test full frame encode
    def testFullFrame(self):
        print("Testing in place full frame compression / decompression")
        eH264 = codec.H264()
        eH264.load_video(self.yuv_file)
        #eH264.load_pattern()#lambda p: 0)
        eH264.show_frame(0)
        eH264.compress_inplace()
        eH264.decompress_inplace()
        eH264.show_frame(0)

    def testBitstream(self):
        print("Testing bitstream creation (via VLC)")
        # Test decompress
        eH264 = codec.H264()
        eH264.load_video(self.yuv_file)
        bitstream = eH264.compress_frame(0)

        print("Testing bitstream decompression (via VLC)")
        dH264 = codec.H264()
        #bitstream = bitstream[:9999] + '1' + bitstream[9999:]
        dH264.load_bitstream(bitstream)
        dH264.show_frame(0)

if __name__ == '__main__':
    unittest.main()