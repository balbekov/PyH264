from numpy import *
from h264.TransformBlock import *
import logging as logger

# Using internal 16 bit types
class MacroBlock():
    'Macroblock Class (Composed of 4x4 TransformBlocks). Other partitions unsupported'
    # 4x4 laid out in raster order:
    #         0    1   2   3
    #         4    5   6   7 ... 16

    # For the encode direction, create a block with an optional preload and size 
    def __init__(self, parent, new_block, mb_size = 16):
        self.valid = 1
        self.parent = 1
        if(new_block != None):
            assert(len(new_block) == 16)
            self.blocks = [TransformBlock(self, new_block[floor(i/4)*4:floor(i/4)*4+4, (i%4)*4:(i%4)*4+4]) for i in range(16)]
        else:
            self.blocks = [TransformBlock(self, zeros((4, 4))) for i in range(16)]

    # For the decode direction, create a new MacroBlock from a VLC
    # def __init__(self, vlc = None):
    #     self.block = zeros(4,4)
    #     # Pregenerate cosines (this might be slow)
    #     self.coses = self.cosines(kernel_size)

    def __str__(self):
        return str(self.blocks)

    def dct(self):
        for block in self.blocks:
            blocks.dct()

    def idct(self):
        for block in self.blocks:
            blocks.idct()

    def vlc_dec(self, VLC):
        return VLC.expgolomb_dec(self.kernel_size, VLC)

    # Return the bottom row of the macroblock
    def right_column(self):
        return concatenate([self.blocks[i].block[0:4,3] for i in range(3, 16, 4)])

    # Return the right row of the macroblock
    def bottom_row(self):
        return concatenate([self.blocks[i].block[3,0:4] for i in range(12,16)])

    # Given the surrounding pixels (one row above, one column before)
    # Evaluate if the supported prediction modes help, and encode the block with that mode
    # THIS IS A FULL MACROBLOCK INTRA COMPRESSION MODE
    def intra_predict(self, neighbors):
        # Unpack the neighbors tuple
        top_row = neighbors[0]
        left_column = neighbors[1]
 
        # Initialize empty predicted macroblocks
        dc_pred = empty((16,16))
        h_pred = empty((16,16))
        v_pred = empty((16,16))

        dc_residual = [empty((4,4))] * 16
        h_residual = [empty((4,4))] * 16
        v_residual = [empty((4,4))] * 16

        # Create the intra predictions
        for x in range(16):
            for y in range(16):
                dc_pred[y,x] = top_row[0]
                h_pred[y, x] = left_column[y]
                v_pred[y, x] = top_row[x]

        #dc_pred = empty((16,16)).fill(top_row[0])

        # Evaluate the residuals for SNR
        for x in range(4):
            for y in range(4):
                dc_residual[y*4+x]  = dc_pred[x*4:x*4+4, y*4:y*4+4] - self.blocks[y+x*4].block
                h_residual[y*4+x]   = h_pred[x*4:x*4+4, y*4:y*4+4] - self.blocks[y+x*4].block
                v_residual[y*4+x]   = v_pred[x*4:x*4+4, y*4:y*4+4] - self.blocks[y+x*4].block
        
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
        for x in range(4):
            for y in range(4):
                self.blocks[x*4+y].block = best_blocks(best_mode[0])[y*4+x]
                self.blocks[x*4+y].prediction_mode = best_mode[0]
                #logger.info("Picking mode %s for block %i", best_mode[x*4+y], )

        return best_mode[0]

    # Accessor method for VLC (to allow redirection of VLC type)
    def get_vlc(self):
        mb_bits = ""

        for block in self.blocks:
            mb_bits += block.get_vlc()

        return mb_bits

    # Setter method to set VLC, returns unconsumed VLC
    def set_vlc(self, vlc):
        for i in range(16):
            vlc = self.blocks[i].set_vlc(vlc)

        return vlc

    # Expose the Iterator of the self.blocks object
    def __getitem__(self, index):
            result = self.blocks[index]
            return result

    def get_image(self):
        mb_image = uint8(empty((16,16)))
        for i,block in enumerate(self.blocks):
            mb_image[floor(i/4)*4:floor(i/4)*4+4, (i%4)*4:(i%4)*4+4] = block.block

        return mb_image