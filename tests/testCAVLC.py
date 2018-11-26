from h264.MacroBlock import *
from h264.CAVLC import *
from h264 import H264 as codec

from numpy import *
import unittest
import logging

# Issues:
# You try iterating over the coeffs, but hte ceoffs are just the regular coeffs, not the T1s
# Which need to be encoded with zeros too
# and remember -- the run_before runs from LF to HF, but the coeff order is HF to LF
class TestCAVLC(unittest.TestCase):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    def testCAVLCEncode(self):
        # CAVLC zigzag scan
        cavlc_test_arr = [[0, 3, -1, 0],
                          [0, -1, 1, 0],
                          [1, 0, 0, 0],
                          [0, 0, 0, 0]]
        cavlc_zigzag_arr = block_to_zigzag(cavlc_test_arr)
        #assert(cavlc_zigzag_arr == [0,3,0,1,-1,-1,0,1,0,0,0,0,0,0,0,0])

        # CAVLC encode
        cavlc_output = []
        cavlc_output = CAVLC_enc(cavlc_test_arr)

        # For this check the predicted macroblock must be empty
        self.assertEqual(cavlc_output, "000010001110010111101101")

if __name__ == '__main__':
    unittest.main()
