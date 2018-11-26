from h264.MacroBlock import *
from h264.CAVLC import *
from h264 import H264 as codec

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
yuv_file = open("sequences/test.y4m", "rb")

# Test full frame encode
print("Testing in place full frame compression / decompression")
eH264 = codec.H264()
eH264.load_video(yuv_file)
#eH264.load_pattern()#lambda p: 0)
#eH264.show_frame(0)
#eH264.compress_inplace()
#eH264.decompress_inplace()
#eH264.show_frame(0)

print("Testing bitstream creation (via VLC)")
# Test decompress 
bitstream = eH264.compress_frame(0)
print(len(bitstream))
# print("Testing bitstream decompression (via VLC)")
# dH264 = codec.H264()
# #bitstream = bitstream[:9999] + '1' + bitstream[9999:]
# dH264.load_bitstream(bitstream)
# dH264.show_frame(0)

print("Done!")
