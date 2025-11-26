from h264.MacroBlock import MacroBlock
from PIL import Image
import logging

class Slice:
    'Binary (VLC) coded collection of multiple macroblocks'



    def __init__(self, parent, new_slice, WIDTH=1280, HEIGHT=720, qp=26, mb_size=16, tb_size=4):
        self.width = WIDTH
        self.height = HEIGHT
        self.blocks = []
        self.parent = parent
        self.qp = qp
        self.mb_size = mb_size
        self.tb_size = tb_size
        # Optional preload step
        if new_slice is not None:
            self.load_blocks(new_slice)
        else: 
            for x in range(0, WIDTH, mb_size):
                self.blocks.append(MacroBlock(self, None, mb_size=mb_size, tb_size=tb_size, qp=qp))      

    # For an input slice numpy array, create transform blocks
    def load_blocks(self, new_slice):
        for x in range(0, new_slice.shape[1], self.mb_size):
            self.blocks.append(MacroBlock(self, new_slice[0:self.mb_size, x:x+self.mb_size], 
                                          mb_size=self.mb_size, tb_size=self.tb_size, qp=self.qp)) 


    def __str__(self):
        txt =  ""
        for block in self.blocks:
            txt += str(block)

        return txt

    def get_bits(self):
        slice_bits = []
        for block in self.blocks:
            #block.intra_predict( (block.bottom_row(), block.right_column()) )
            #block.intra_predict()
            block_bits = block.get_vlc()
            slice_bits.append(block_bits)

        slice_bits = ''.join(slice_bits)

        # Append the slice synchronization marker
        slice_bits = slice_bits + '000000001'

        return slice_bits

    # Uses bits to read in VLC into macroblocks, send remainder back
    def set_bits(self, vlc):
        for i, block in enumerate(self.blocks):
            vlc = block.set_vlc(vlc)
            logging.debug("Set VLC for MB: %d", i)
            # If we encountered a block decoding error, terminate the slice
            if block.valid == 0:
                return vlc  # The VLC decoder already read to the end of the slice

        # Make sure the slice synchronization marker is present
        if '000000001' not in vlc[0:9]:
            logging.warning("Slice synchronization marker missing")
        
        return vlc[9:]
        
    # Expose the Iterator of the self.slices object
    def __getitem__(self, index):
            result = self.blocks[index]
            return result