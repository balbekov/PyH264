import numpy as np
from numpy import floor, concatenate, empty, zeros
from h264.TransformBlock import TransformBlock
import logging as logger

# Using internal 16 bit types
class MacroBlock():
    'Macroblock Class (Composed of TransformBlocks). Supports configurable mb_size and tb_size'
    # Transform blocks laid out in raster order

    # For the encode direction, create a block with an optional preload and size 
    def __init__(self, parent, new_block, mb_size=16, tb_size=4, qp=26):
        self.valid = 1
        self.parent = parent
        self.mb_size = mb_size
        self.tb_size = tb_size
        self.qp = qp
        
        # Number of transform blocks per dimension and total
        self.blocks_per_dim = mb_size // tb_size
        self.num_blocks = self.blocks_per_dim ** 2
        
        if new_block is not None:
            assert(len(new_block) == mb_size)
            self.blocks = [
                TransformBlock(self, 
                              new_block[int(floor(i/self.blocks_per_dim))*tb_size:int(floor(i/self.blocks_per_dim))*tb_size+tb_size, 
                                       (i%self.blocks_per_dim)*tb_size:(i%self.blocks_per_dim)*tb_size+tb_size],
                              kernel_size=tb_size, qp=qp) 
                for i in range(self.num_blocks)
            ]
        else:
            self.blocks = [TransformBlock(self, zeros((tb_size, tb_size)), kernel_size=tb_size, qp=qp) 
                          for i in range(self.num_blocks)]

    # For the decode direction, create a new MacroBlock from a VLC
    # def __init__(self, vlc = None):
    #     self.block = zeros(4,4)
    #     # Pregenerate cosines (this might be slow)
    #     self.coses = self.cosines(kernel_size)

    def __str__(self):
        return str(self.blocks)

    def dct(self):
        for block in self.blocks:
            block.dct()

    def idct(self):
        for block in self.blocks:
            block.idct()

    def vlc_dec(self, VLC):
        return VLC.expgolomb_dec(self.kernel_size, VLC)

    # Return the right column of the macroblock
    def right_column(self):
        return concatenate([self.blocks[i].block[0:self.tb_size, self.tb_size-1] 
                           for i in range(self.blocks_per_dim-1, self.num_blocks, self.blocks_per_dim)])

    # Return the bottom row of the macroblock
    def bottom_row(self):
        start_idx = self.num_blocks - self.blocks_per_dim
        return concatenate([self.blocks[i].block[self.tb_size-1, 0:self.tb_size] 
                           for i in range(start_idx, self.num_blocks)])

    # Given the surrounding pixels (one row above, one column before)
    # Evaluate if the supported prediction modes help, and encode the block with that mode
    # THIS IS A FULL MACROBLOCK INTRA COMPRESSION MODE
    def intra_predict(self, neighbors):
        # Unpack the neighbors tuple
        top_row = neighbors[0]
        left_column = neighbors[1]
 
        # Initialize empty predicted macroblocks
        dc_pred = empty((self.mb_size, self.mb_size))
        h_pred = empty((self.mb_size, self.mb_size))
        v_pred = empty((self.mb_size, self.mb_size))

        dc_residual = [empty((self.tb_size, self.tb_size))] * self.num_blocks
        h_residual = [empty((self.tb_size, self.tb_size))] * self.num_blocks
        v_residual = [empty((self.tb_size, self.tb_size))] * self.num_blocks

        # Create the intra predictions
        for x in range(self.mb_size):
            for y in range(self.mb_size):
                dc_pred[y,x] = top_row[0]
                h_pred[y, x] = left_column[y]
                v_pred[y, x] = top_row[x]

        #dc_pred = empty((mb_size,mb_size)).fill(top_row[0])

        # Evaluate the residuals for SNR
        for x in range(self.blocks_per_dim):
            for y in range(self.blocks_per_dim):
                tb = self.tb_size
                dc_residual[y*self.blocks_per_dim+x] = dc_pred[x*tb:x*tb+tb, y*tb:y*tb+tb] - self.blocks[y+x*self.blocks_per_dim].block
                h_residual[y*self.blocks_per_dim+x] = h_pred[x*tb:x*tb+tb, y*tb:y*tb+tb] - self.blocks[y+x*self.blocks_per_dim].block
                v_residual[y*self.blocks_per_dim+x] = v_pred[x*tb:x*tb+tb, y*tb:y*tb+tb] - self.blocks[y+x*self.blocks_per_dim].block
        
        # Pick the smallest value (if above threshold) for the entire macroblock
        dc_snr = [(id, abs(dc_x.mean())) for id, dc_x in enumerate(dc_residual)] 
        dc_sum = sum(x[1] for x in dc_snr)
        #dc_best = min(dc_snr, key = lambda t: t[1])

        h_snr = [(id, abs(h_x.mean())) for id, h_x in enumerate(h_residual)]
        h_sum = sum(x[1] for x in h_snr)
        #h_best = min(h_snr, key = lambda t: t[1])

        v_snr = [(id, abs(v_x.mean())) for id, v_x in enumerate(v_residual)]
        v_sum = sum(x[1] for x in v_snr)
        #v_best = min(v_snr, key = lambda t: t[1])
        
        logger.info("DC SNR: %f H_SNR: %f V_SNR: %f", dc_sum, h_sum, v_sum)

        best_modes = [("dc", dc_sum), ("v", v_sum), ("h", h_sum)]
        best_mode = min(best_modes, key = lambda t: t[1])

        # Create reverse mapping from the best mode to the blocks of the best mode
        def best_blocks(x):
            return {
                'dc': dc_residual,
                'h': h_residual,
                'v' : v_residual
            }[x]

        # Set the prediction mode and insert the residual to be coded
        for x in range(self.blocks_per_dim):
            for y in range(self.blocks_per_dim):
                self.blocks[x*self.blocks_per_dim+y].block = best_blocks(best_mode[0])[y*self.blocks_per_dim+x]
                self.blocks[x*self.blocks_per_dim+y].prediction_mode = best_mode[0]
                #logger.info("Picking mode %s for block %i", best_mode[x*blocks_per_dim+y], )

        return best_mode[0]

    # Accessor method for VLC (to allow redirection of VLC type)
    def get_vlc(self):
        mb_bits = ""

        for block in self.blocks:
            mb_bits += block.get_vlc()

        return mb_bits

    # Setter method to set VLC, returns unconsumed VLC
    def set_vlc(self, vlc):
        for i in range(self.num_blocks):
            vlc = self.blocks[i].set_vlc(vlc)

        return vlc

    # Expose the Iterator of the self.blocks object
    def __getitem__(self, index):
            result = self.blocks[index]
            return result

    def get_image(self):
        mb_image = np.uint8(empty((self.mb_size, self.mb_size)))
        for i, block in enumerate(self.blocks):
            row = int(floor(i / self.blocks_per_dim))
            col = i % self.blocks_per_dim
            mb_image[row*self.tb_size:row*self.tb_size+self.tb_size, 
                     col*self.tb_size:col*self.tb_size+self.tb_size] = block.block

        return mb_image