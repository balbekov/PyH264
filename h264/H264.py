from builtins import range
from math import cos

import numpy as np
from numpy import empty
from PIL import Image

from h264.Frame import Frame

import logging

class H264:
    'H264 coder object with memory for several frames. Operates \
     on file pointer; generates bitstream into output FIFO'
    def __init__(self, width=1280, height=720):
        # Strictly incrementing frame counter
        self.frame_counter = 0
        self.frames = []
        # Default dimensions (can be overridden by loaded video)
        self.width = width
        self.height = height

 # Compresses the frame specified by frame_id, outputs bitstream
    def compress_frame(self, frame_id):
        frame = self.frames[frame_id]

        return frame.get_bits()

    # Load a test pattern (the X coordinate of each pixel is the value (with wrap))
    def load_pattern(self, pattern=lambda p: min(p, 255)):
        test_data = empty((self.height, self.width))
        for i in range(self.width):
            for j in range(self.height):
                test_data[j, i] = pattern(i)
        self.frames.append(Frame(self, test_data, WIDTH=self.width, HEIGHT=self.height))

    def load_bitstream(self, vlc):
        frame = Frame(self, None, WIDTH=self.width, HEIGHT=self.height)
        frame.set_bits(vlc)
        self.frames.append(frame)

    # Load a video from a YUV4MPEG format file pointer
    def load_video(self, yuv_file):
        # Read the header
        header = yuv_file.read(10)
        logging.debug("Y4M Header: %s", header)
        assert "YUV4MPEG2 " in str(header)

        # Read the container params
        params = ""
        watchdog_count = 100
        while params[-5:] != "FRAME":
            params += str(yuv_file.read(1).decode("utf-8"))
            watchdog_count -= 1
            assert watchdog_count > 0

        # Parse width and height from params (format: "W1280 H720 ...")
        for param in params.split():
            if param.startswith('W'):
                self.width = int(param[1:])
            elif param.startswith('H'):
                self.height = int(param[1:])

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
        logging.debug("Container parameters: %s", params)
        logging.debug("Frame parameters: %s", frame_params)
        logging.info("Video dimensions: %dx%d", self.width, self.height)

        # Start populating the frame (Row-major full plane, 4:2:0, YCbCr order)
        y_frame = self.grab_plane(yuv_file, self.width, self.height)
        cr_frame = self.grab_plane(yuv_file, self.width // 2, self.height // 2)
        cb_frame = self.grab_plane(yuv_file, self.width // 2, self.height // 2)
        self.frames.append(Frame(self, y_frame, WIDTH=self.width, HEIGHT=self.height))

    def load_image(self, image_path):
        """Load an image file (jpg, png, bmp, gif, tiff) as a single frame.
        
        The image is converted to grayscale (Y channel only) and padded
        to be divisible by 16 for macroblock alignment.
        
        Args:
            image_path: Path to image file
        """
        # Load image and convert to grayscale
        img = Image.open(image_path)
        img_gray = img.convert('L')
        
        # Get dimensions
        orig_width, orig_height = img_gray.size
        
        # Pad to be divisible by 16 (macroblock size)
        self.width = ((orig_width + 15) // 16) * 16
        self.height = ((orig_height + 15) // 16) * 16
        
        # Create padded array
        frame_data = np.zeros((self.height, self.width), dtype=np.uint8)
        
        # Copy image data
        img_array = np.array(img_gray, dtype=np.uint8)
        frame_data[:orig_height, :orig_width] = img_array
        
        # Pad edges by repeating border pixels
        if orig_width < self.width:
            frame_data[:orig_height, orig_width:] = img_array[:, -1:].repeat(
                self.width - orig_width, axis=1)
        if orig_height < self.height:
            frame_data[orig_height:, :] = frame_data[orig_height-1:orig_height, :].repeat(
                self.height - orig_height, axis=0)
        
        logging.info("Loaded image: %dx%d (padded to %dx%d)", 
                    orig_width, orig_height, self.width, self.height)
        
        self.frames.append(Frame(self, frame_data, WIDTH=self.width, HEIGHT=self.height))

    def show_frame(self, frame_id):
        frame = self.frames[frame_id].get_image()
        im = Image.fromarray(frame, "L")
        im.show()
        
    # Given a YUV4MPEG2 file pointer with seek marker at FRAME start,
    # Read a single plane with dimensions WIDTHxHEIGHT and return a UINT8 NP matrix
    def grab_plane(self, yuv_file, width, height):
        # The cast to uint8 is necessary for PIL to display the image
        frame = np.uint8(empty([height, width]))

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
