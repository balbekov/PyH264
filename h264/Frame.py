from h264.Slice import Slice
import numpy as np
import logging

''' docstring
'''
class Frame:
    'Binary (VLC) coded collection of multiple slices'

    def __init__(self, parent, new_frame, WIDTH=1280, HEIGHT=720, qp=26, mb_size=16, tb_size=4):
        self.width = WIDTH
        self.height = HEIGHT
        self.slices = []
        self.parent = parent
        self.qp = qp
        self.mb_size = mb_size
        self.tb_size = tb_size
        if new_frame is not None:
            self.load_image(new_frame)
        else:
            self.slices = [Slice(self, None, WIDTH=WIDTH, qp=qp, mb_size=mb_size, tb_size=tb_size) 
                          for i in range(0, HEIGHT, mb_size)]

    '''
    Load a Numpy UINT8 matrix into this frame
    Using one slice per mb_size lines
    '''
    def load_image(self, new_frame):
        for y in range(0, new_frame.shape[0], self.mb_size):
            self.slices.append(Slice(self, new_frame[y:y+self.mb_size, 0:new_frame.shape[1]], 
                                     WIDTH=self.width, qp=self.qp, mb_size=self.mb_size, tb_size=self.tb_size))

    def get_image(self):
        image = np.uint8(np.empty((self.height, self.width)))
        # Iterate through slice -> macroblock -> transformblock layers
        # Y coordinate increments by mb_size for every slice
        for i, slice in enumerate(self.slices):
            for j, mb in enumerate(slice):
                image[(i*self.mb_size):(i*self.mb_size)+self.mb_size, 
                      (j*self.mb_size):(j*self.mb_size)+self.mb_size] = mb.get_image()

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
            logging.debug("Finished slice %d, %d bits remaining", i, len(vlc))

        if len(vlc) > 0:
            logging.warning("Not all VLC bits used in frame processing")

    # Get the residuals of this Frame when exposed to a predicted Frame
    def get_residuals(self, pred_frame):
        return None

    def __str__(self):
        for frame_slice in slices:
            print(frame_slice)
