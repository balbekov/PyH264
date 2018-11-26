from builtins import range
from math import cos

from numpy import *
from PIL import Image

from h264.Frame import Frame

import logging

class H264:
    'H264 coder object with memory for several frames. Operates \
     on file pointer; generates bitstream into output FIFO'
    def __init__(self):
        # Strictly incrementing frame counter
        self.frame_counter = 0
        self.frames = []

 # Compresses the frame specified by frame_id, outputs bitstream
    def compress_frame(self, frame_id):
        frame = self.frames[frame_id]

        return frame.get_bits()

    # Load a test pattern (the X coordinate of each pixel is the value (with wrap))
    def load_pattern(self, pattern = lambda p: min(p, 255)):
        test_data = empty((720,1280))
        for i in range(1280):
            for j in range(720):
                test_data[j,i] =pattern(i)
        self.frames.append(Frame(self, test_data))

    def load_bitstream(self, vlc):
        frame = Frame(self, None)
        frame.set_bits(vlc)
        self.frames.append(frame)

    # Load a video from a YUV4MPEG format file pointer
    def load_video(self, yuv_file):
        # Read the header
        header = yuv_file.read(10)
        print(header)
        assert "YUV4MPEG2 " in str(header)

        # Read the container params
        params = ""
        watchdog_count = 100
        while params[-5:] != "FRAME":
            params += str(yuv_file.read(1).decode("utf-8"))
            watchdog_count -= 1
            assert watchdog_count > 0

        #print(params)

        # Read the frame params
        frame_params = " "
        watchdog_count = 100
        while frame_params[-1] != "\x0a":
            frame_params += str(yuv_file.read(1).decode("utf-8"))
            watchdog_count -= 1
            assert watchdog_count > 0

        # Convert byte sequence to string
        frame_params = ''.join(frame_params)

        # Debug info
        print("Container parameters: ",  params)
        print("Frame parameters: ", frame_params)

        # Start populating the frame (Row-major full plane, 4:2:0, YCbCr order)
        y_frame = self.grab_plane(yuv_file, 1280, 720)
        cr_frame = self.grab_plane(yuv_file, (int) (1280/2), (int) (720/2))
        cb_frame = self.grab_plane(yuv_file, (int) (1280/2), (int) (720/2))
        self.frames.append(Frame(self, y_frame))

    def show_frame(self, id):
        frame = self.frames[id].get_image()
        im = Image.fromarray(frame, "L")
        im.show()
        
    # Given a YUV4MPEG2 file pointer with seek marker at FRAME start,
    # Read a single plane with dimensions WIDTHxHEIGHT and return a UINT8 NP matrix
    def grab_plane(self, yuv_file, width, height):
        # The cast to uint8 is necessary for PIL to display the image
        frame = uint8(empty([height, width]))

        # Iterate over the plane, raster (row major) order, reading 1 byte per plane
        for y in range(height):
            for x in range(width):
                byte = yuv_file.read(1)
                # if(x < 10 and y < 10):
                #     print(yuv_file.tell())
                #     print(byte)
                frame[y, x] = int.from_bytes(byte, byteorder='little')

        return frame


    # Compress in place
    def compress_inplace(self):
        for frame in self.frames:
            for i, slice in enumerate(frame.slices):
                logging.info("Compressed %i slices of %i", i, len(frame.slices))
                for mb in slice.blocks:
                    for tb in mb.blocks:
                        tb.dct()
                        tb.quantize()

    # Decompress in place
    def decompress_inplace(self):
         for frame in self.frames:
            for i, slice in enumerate(frame.slices):
                logging.info("Decompressed %i slicess of %i", i, len(frame.slices))
                for mb in slice.blocks:
                    for tb in mb.blocks:
                        tb.dequantize()
                        tb.idct()       
