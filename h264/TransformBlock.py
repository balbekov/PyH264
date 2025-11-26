import numpy as np
from numpy import zeros, around, asarray, matrix, transpose
from math import cos

import h264.DCT as DCT
import h264.VLC as VLC
import h264.CAVLC as CAVLC

# Using internal 16 bit types
class TransformBlock():
    'Transform Block Class (4x4 or 8x8). DCT kernel type with configurable QP'

    # Base quantization matrices for 4x4 and 8x8
    qmat_base_4x4 = asarray(matrix("12 12 14 16; 12 14 16 20; 14 16 20 25; 16 20 25 32"))
    qmat_base_8x8 = asarray(matrix(
        "16 16 16 16 17 18 21 24; "
        "16 16 16 16 17 19 22 25; "
        "16 16 17 18 20 22 25 29; "
        "16 16 18 21 24 27 31 36; "
        "17 17 20 24 30 35 41 47; "
        "18 19 22 27 35 44 54 65; "
        "21 22 25 31 41 54 70 88; "
        "24 25 29 36 47 65 88 115"
    ))
    
    # Class-level cosine LUT cache (keyed by kernel_size)
    coses_cache = {}

    # For the encode direction, create a block with an optional preload and size 
    def __init__(self, parent, new_block, kernel_size=4, qp=26):
        self.parent = parent
        self.qp = qp
        self.kernel_size = kernel_size
        
        if new_block is not None:
            self.block = new_block
            self.state = "Spatial"
        else:
            self.block = zeros((kernel_size, kernel_size))

        assert self.block.shape == (kernel_size, kernel_size)

        self.prediction_mode = "PCM"
        
        # Preinitialize the integer transform cosines LUT (cached by kernel size)
        if kernel_size not in TransformBlock.coses_cache:
            TransformBlock.coses_cache[kernel_size] = DCT.cosines(kernel_size)
        self.coses = TransformBlock.coses_cache[kernel_size]
        
        # Compute the effective quantization matrix scaled by QP
        # QP scaling: scale = 2^((qp - 12) / 6)
        # Lower QP = less quantization = higher quality
        # Higher QP = more quantization = more compression
        qp_scale = 2 ** ((qp - 12) / 6.0)
        if kernel_size == 4:
            self.qmat = self.qmat_base_4x4 * qp_scale
        elif kernel_size == 8:
            self.qmat = self.qmat_base_8x8 * qp_scale
        else:
            # For other sizes, generate a simple scaling matrix
            base_val = 16 * qp_scale
            self.qmat = np.full((kernel_size, kernel_size), base_val)

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
        return VLC.expgolomb_dec(self.kernel_size, vlc, self.qp)

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
