from h264.Slice import Slice
from numpy import *
import logging

''' docstring
'''
class Frame:
    'Binary (VLC) coded collection of multiple slices'

    def __init__(self, parent, new_frame, WIDTH=1280, HEIGHT=720):
        self.width = WIDTH
        self.height = HEIGHT
        self.slices = []
        self.parent = parent
        if new_frame != None:
            self.load_image(new_frame)
        else:
            self.slices = [Slice(self, None) for i in range(0, HEIGHT, 16)]

    '''
    Load a Numpy UINT8 matrix into this frame
    Using one slice per 16 lines
    '''
    def load_image(self, new_frame):
        for y in range(0, new_frame.shape[0], 16):
            self.slices.append(Slice(self, new_frame[y:y+16, 0:new_frame.shape[1]]))

    def get_image(self):
        image = uint8(empty((self.height, self.width)))
        # Iterate through slice -> macroblock -> transformblock layers
        # Y coordinate increments by 16 for every slice, and 4 for every transformblock
        for i, slice in enumerate(self.slices):
            for j, mb in enumerate(slice):
                image[(i*16):(i*16)+16, (j*16):(j*16)+16] = mb.get_image()

        return image

    def get_bits(self):
        frame_bits = []

        for i, frame_slice in enumerate(self.slices):
            slice_bits = frame_slice.get_bits()
            slice_bits = ''.join(slice_bits)
            frame_bits.append(slice_bits)
            #print(frame_slice)
            logging.info("Finished %i slices of %i", i, len(self.slices))
            logging.info("Slice size: %i", len(slice_bits))
        frame_bits = ''.join(frame_bits)
        logging.info("Total size %i bits", len(frame_bits))

        return frame_bits

    def set_bits(self, vlc):
        for i, slice in enumerate(self.slices):
            vlc = slice.set_bits(vlc)
            print("Finished slice ", i, " , ", len(vlc), " bits remaining")

        if len(vlc) > 0:
            print("Warning! Not all VLC bits used in frame processing")

    # Get the residuals of this Frame when exposed to a predicted Frame
    def get_residuals(self, pred_frame):
        return None

    def __str__(self):
        for frame_slice in slices:
            print(frame_slice)
