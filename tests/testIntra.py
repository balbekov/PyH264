from h264.MacroBlock import *
from h264.CAVLC import *
from h264 import H264 as codec

from numpy import *
import unittest

class TestIntraCompression(unittest.TestCase):

    def testMBIntra(self):
        #yuv_file = open("Etalon/sequences/test.y4m", "rb")

        # Create the neighboring rows
        mb_data = uint8(empty((16, 16)))
        top_row = uint8(empty(16))
        left_column = uint8(empty(16))

        # Populate the neighboring rows and macroblock with a test pattern
        for i in range(16):
            top_row[i] = i
            left_column[i] = 15

        for i in range(16):
            for j in range(16):
                mb_data[i,j] = j

        mb = MacroBlock(None, mb_data)

        print(mb.bottom_row())
        print(mb.right_column())
        mb_image_data = mb.get_image()
        mode = mb.intra_predict((top_row, left_column))
        mb_image_data_intra = mb.get_image()

        # For this check the predicted macroblock must be empty
        self.assertEqual(sum(mb_image_data_intra), 0)

if __name__ == '__main__':
    unittest.main()