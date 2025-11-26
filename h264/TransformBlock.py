import numpy as np
from numpy import zeros, around, asarray, matrix, transpose
from math import cos

import h264.DCT as DCT
import h264.VLC as VLC
import h264.CAVLC as CAVLC

# Using internal 16 bit types
class TransformBlock():
    'Transform Block Class (4x4 or 8x8 (unsupported)). DCT kernel type'

    qmat =  asarray(matrix("12 12 14 16; 12 14 16 20; 14 16 20 25; 16 20 25 32"))
    #qmat =  asarray(matrix("1 1 1 1; 1 1 1 1; 1 1 1 1; 1 1 1 1"))
    #qmat =  uint8(asarray(matrix("255 255 255 255; 255 255 255 255; 255 255 255 255; 255 255 255 255")))
    #qmat =  asarray(matrix("32 32 64 64; 32 32 64 64; 64 64 64 64; 64 64 64 64"))
    
    coses = []

    # For the encode direction, create a block with an optional preload and size 
    def __init__(self, parent, new_block, kernel_size=4):
        self.parent = parent
        if new_block is not None:
            self.block = new_block
            self.state = "Spatial"
        else:
            self.block = zeros((kernel_size, kernel_size))

        assert self.block.shape == (kernel_size, kernel_size)

        self.prediction_mode = "PCM"
        self.kernel_size = kernel_size
        # Preinitialize the integer transform cosines LUT
        if not len(self.coses):
            self.coses = DCT.cosines(kernel_size)

    def __str__(self):
        return str(self.block)

    def dct(self):
        # DCT(block) = (cosines .* block) .* (cosines^-1)
        if self.state != "Spatial":
            raise ValueError("Block must be in spatial mode for DCT")
        self.block = np.float64(self.block)
        # Important to round; a cast by itself will cause severe aliasing
        self.block = around(self.coses.dot(self.block).dot(transpose(self.coses)))
        self.block = np.int16(self.block)

        self.state = "Frequency"

    def idct(self):
        # IDCT(block) = ((cosines^-1) .* block) .* cosines
        if self.state != "Frequency":
            raise ValueError("Block must be in frequency mode for IDCT")
        self.block = transpose(self.coses).dot(self.block).dot(self.coses)
        self.block = np.uint8(around(self.block))
        self.state = "Spatial"

    def quantize(self):
        if self.state != "Frequency":
            raise ValueError("Block must be in frequency mode for quantization")
        self.block = np.int16(around(self.block / self.qmat))
        self.state = "Quantized"

    def dequantize(self):
        if self.state != "Quantized":
            raise ValueError("Block must be in quantized mode for dequantization")
        self.block = np.int16(around(self.block * self.qmat))
        self.state = "Frequency"

    def vlc_enc(self):
        #return CAVLC.CAVLC_enc(self.block)
        return VLC.expgolomb_enc(self.kernel_size, self.block)

    def vlc_dec(self, vlc):
        return VLC.expgolomb_dec(self.kernel_size, vlc)

    # Accessor method for VLC (to allow redirection of VLC type)
    def get_vlc(self):
        assert self.state != "Frequency"
        if self.state == "Spatial":
            self.dct()
            self.quantize()
        return self.vlc_enc()

    # Return the remaining VLC string pointer up the chain
    def set_vlc(self, vlc):
        (result, rem_vlc, self.block) = self.vlc_dec(vlc)
        # If we encountered a VLC failure, invalidate the parent slice
        if result < 0:
            self.parent.valid = 0
        self.state = "Quantized"
        self.dequantize()
        self.idct()
        return rem_vlc
