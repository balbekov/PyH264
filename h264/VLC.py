from numpy import *

golomb_enc_LUT = {}
golomb_dec_LUT = {}

def init_LUT():
    global golomb_dec_LUT
    global golomb_enc_LUT
    golomb_enc_LUT[0] = '1'
    for i in range(1,2**15):
        # bin() returns a value prefixed with 0b; [2:] slices it off
        # Note that the (2*i)-1 term is intended to support signed integers
        # (all positive integers are mapped to 2x-1)
        golomb_enc_LUT[i] = count_bits( bin( ((2*i)-1)+1 )[2:]) + bin( ((2*i)-1)+1 )[2:]
    for i in range(-(2**15), 0):
        # Similarly, negative integers are mapped to -2x
        golomb_enc_LUT[i] = count_bits( bin( ((2*-i))+1 )[2:] ) + bin( ((2*-i))+1 )[2:]
    golomb_dec_LUT = {v: k for k, v in golomb_enc_LUT.items()}

# Count the number of bits, subtract one, write that number of zeros
def count_bits(value):
        return ''.join(['0' for s in range(len(value)-1)])

def expgolomb_enc(kernel_size, block):
    if(len(golomb_enc_LUT) == 0):
        init_LUT()
    vlc = []

    for i in range(kernel_size):
        for j in range(kernel_size):
            lut_value = golomb_enc_LUT[block[i][j]] 
            vlc.append(lut_value)
            #print("Generating VLC for ", i, ",", j, "(", block[i][j], ")", " :", lut_value)
    vlc = ''.join(vlc)
    
    return vlc


def cavlc_enc():
    vlc = []

    for i in range(kernel_size):
        for j in range(kernel_size):
            vlc.append(lut_value)
            #print("Generating VLC for ", i, ",", j, "(", block[i][j], ")", " :", lut_value)
    vlc = ''.join(vlc)

    return vlc

def expgolomb_dec(kernel_size, vlc):
    if(len(golomb_dec_LUT) == 0):
        init_LUT()
    i = 0
    j = 0

    # Sliding window of the VLC
    window = []
    block = int16(full((4,4), 127, dtype=int16))

    # Slice characters off the VLC, see if they match an element in our LUT
    # Code is prefix free and uniquely decodable.
    bitcount = 0
    for el in vlc:
        window.append(el)
        if(len(window) == 100):
            print("Warning! VLC decoding error. Consuming until slice marker.")
            
            # Skip to next slice synchronization marker
            slice_marker = vlc.find('000000001')
            if(slice_marker > 0):
                return (-1, vlc[slice_marker:], block)

        bitcount += 1
        if "".join(window) in golomb_dec_LUT:
            block[i][j] = golomb_dec_LUT["".join(window)]
            if(j == 3):
                i = i + 1
            j = (j + 1) % 4
            window = []
        if(i > 3):
            return (1, vlc[bitcount:], block)

    print("Error decoding block. VLC underrun.")
    return (-1, [], block)

def cavlc_enc(block):
    return block

def cavlc_dec(block):
    return block
