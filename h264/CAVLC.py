from numpy import *
import logging as logger

# Coeff_token tables; index by Num_Coeffs in first dimension, Trailing Ones in second
# Tables decoded by number of coefficients (nC) in Upper and Left blocks: 0: 0 <= x <2, 1: 2 <= x < 4, 2: 4 <= x < 8; 3: x >= 8
# U > 0 & L > 0 -> nC = avg(U, L), else nC = U + L

coeff_token0 =  [['1'                 , ''                  , ''                 , ''                ],
                 ['000101'            , '01'                , ''                 , ''                ],
                 ['00000111'          , '000100'            , '001'              , ''                ],
                 ['000000111'         , '00000110'          , '0000101'          , '00011'           ],
                 ['0000000111'        , '000000110'         , '00000101'         , '000011'          ],
                 ['00000000111'       , '0000000110'        , '000000101'        , '0000100'         ],
                 ['0000000001111'     , '00000000110'       , '0000000101'       , '00000100'        ],
                 ['0000000001011'     , '0000000001110'     , '00000000101'      , '000000100'       ],
                 ['0000000001000'     , '0000000001010'     , '0000000001101'    , '0000000100'      ],
                 ['00000000001111'    , '00000000001110'    , '0000000001001'    , '00000000100'     ],
                 ['00000000001011'    , '00000000001010'    , '00000000001101'   , '0000000001100'   ],
                 ['000000000001111'   , '000000000001110'   , '00000000001001'   , '00000000001100'  ],
                 ['000000000001011'   , '000000000001010'   , '000000000001101'  , '00000000001000'  ],
                 ['0000000000001111'  , '000000000000001'   , '000000000001001'  , '000000000001100' ],
                 ['0000000000001011'  , '0000000000001110'  , '0000000000001101' , '000000000001000' ],
                 ['0000000000000111'  , '0000000000001010'  , '0000000000001001' , '0000000000001100'],
                 ['0000000000000100'  , '0000000000000110'  , '0000000000000101' , '0000000000001000']]

coeff_token2 = [['11'             , ''               , ''               , ''              ],
                ['001011'         , '10'             , ''               , ''              ],
                ['000111'         , '00111'          , '011'            , ''              ],
                ['0000111'        , '001010'         , '001001'         , '0101'          ],
                ['00000111'       , '000110'         , '000101'         , '0100'          ],
                ['00000100'       , '0000110'        , '0000101'        , '00110'         ],
                ['000000111'      , '00000110'       , '00000101'       , '001000'        ],
                ['00000001111'    , '000000110'      , '000000101'      , '000100'        ],
                ['00000001011'    , '00000001110'    , '00000001101'    , '0000100'       ],
                ['000000001111'   , '00000001010'    , '00000001001'    , '000000100'     ],
                ['000000001011'   , '000000001110'   , '000000001101'   , '00000001100'   ],
                ['000000001000'   , '000000001010'   , '000000001001'   , '00000001000'   ],
                ['0000000001111'  , '0000000001110'  , '0000000001101'  , '000000001100'  ],
                ['0000000001011'  , '0000000001010'  , '0000000001001'  , '0000000001100' ],
                ['0000000000111'  , '00000000001011' , '0000000000110'  , '0000000001000' ],
                ['00000000001001' , '00000000001000' , '00000000001010' , '0000000000001' ],
                ['00000000000111' , '00000000000110' , '00000000000101' , '00000000000100']]


coeff_token3 = [['1111'       , ''           , ''           , ''          ],
                ['001111'     , '1110'       , ''           , ''          ],
                ['001011'     , '01111'      , '1101'       , ''          ],
                ['001000'     , '01100'      , '01110'      , '1100'      ],
                ['0001111'    , '01010'      , '01011'      , '1011'      ],
                ['0001011'    , '01000'      , '01001'      , '1010'      ],
                ['0001001'    , '001110'     , '001101'     , '1001'      ],
                ['0001000'    , '001010'     , '001001'     , '1000'      ],
                ['00001111'   , '0001110'    , '0001101'    , '01101'     ],
                ['00001011'   , '00001110'   , '0001010'    , '001100'    ],
                ['000001111'  , '00001010'   , '00001101'   , '0001100'   ],
                ['000001011'  , '000001110'  , '00001001'   , '00001100'  ],
                ['000001000'  , '000001010'  , '000001101'  , '00001000'  ],
                ['0000001101' , '000000111'  , '000001001'  , '000001100' ],
                ['0000001001' , '0000001100' , '0000001011' , '0000001010'],
                ['0000000101' , '0000001000' , '0000000111' , '0000000110'],
                ['0000000001' , '0000000100' , '0000000011' , '0000000010']]

coeff_token4 = [['000011' , ''        , ''       , ''      ],
                ['000000' , '000001'  , ''       , ''      ],
                ['000100' , '000101'  , '000110' , ''      ],
                ['001000' , '001001'  , '001010' , '001011'],
                ['001100' , '001101'  , '001110' , '001111'],
                ['010000' , '010001'  , '010010' , '010011'],
                ['010100' , '010101'  , '010110' , '010111'],
                ['011000' , '011001'  , '011010' , '011011'],
                ['011100' , '011101'  , '011110' , '011111'],
                ['100000' , '100001'  , '100010' , '100011'],
                ['100100' , '100101'  , '100110' , '100111'],
                ['101000' , '101001'  , '101010' , '101011'],
                ['101100' , '101101'  , '101110' , '101111'],
                ['110000' , '110001'  , '110010' , '110011'],
                ['110100' , '110101'  , '110110' , '110111'],
                ['111000' , '111001'  , '111010' , '111011'],
                ['111100' , '111101'  , '111110' , '111111']]

Table_zeros =  [['1'      , '011'    , '010'   , '0011' , '0010' , '00011' , '00010' ,  '000011' , '000010' ,  '0000011' , '0000010', '00000011' , '00000010' , '000000011' , '000000010', '000000001'],
                ['111'    , '110'    , '101'   , '100'  , '011'  , '0101'  , '0100'  ,  '0011'   , '0010'   ,  '00011'   , '00010'  , '000011'   , '000010'   , '000001'    , '000001'   , ''         ],
                ['0101'   , '111'    , '110'   , '101'  , '0100' , '0011'  , '100'   ,  '011'    , '0010'   ,  '00011'   , '00010'  , '000001'   , '00001'    , '000000'    , ''         , ''         ],
                ['00011'  , '111'    , '0101'  , '0100' , '110'  , '101'   , '100'   ,  '0011'   , '011'    ,  '0010'    , '00010'  , '00001'    , '00000'    , ''          , ''         , ''         ],
                ['0101'   , '0100'   , '0011'  , '111'  , '110'  , '101'   , '100'   ,  '011'    , '0010'   ,  '00001'   , '0001'   , '00000'    , ''         , ''          , ''         , ''         ],
                ['000001' , '00001'  , '111'   , '110'  , '101'  , '100'   , '011'   ,  '010'    , '0001'   ,  '001'     , '000000' , ''         , ''         , ''          , ''         , ''         ],
                ['000001' , '00001'  , '101'   , '100'  , '011'  , '11'    , '010'   ,  '0001'   , '001'    ,  '000000'  , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['000001' , '0001'   , '00001' , '011'  , '11'   , '10'    , '010'   ,  '001'    , '000000' ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['000001' , '000000' , '0001'  , '11'   , '10'   , '001'   , '01'    ,  '00001'  , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],               
                ['00001'  , '00000'  , '001'   , '11'   , '10'   , '01'    , '0001'  ,  ''       , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['0000'   , '0001'   , '001'   , '010'  , '1'    , '011'   , ''      ,  ''       , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['0000'   , '0001'   , '01'    , '1'    , '001'  , ''      , ''      ,  ''       , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['000'    , '001'    , '1'     , '01'   , ''     , ''      , ''      ,  ''       , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['00'     , '01'     , '1'     , ''     , ''     , ''      , ''      ,  ''       , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ],
                ['0'      , '1'      , ''      , ''     , ''     , ''      , ''      ,  ''       , ''       ,  ''        , ''       , ''         , ''         , ''          , ''         , ''         ]]

Table_run =    [['1' , '1'  , '11' , '11'  , '11'  , '11'  , '111'        ],
                ['0' , '01' , '10' , '10'  , '10'  , '000' , '110'        ],
                [''  , '00' , '01' , '01'  , '011' , '001' , '101'        ],
                [''  , ''   , '00' , '001' , '010' , '011' , '100'        ],
                [''  , ''   , ''   , '000' , '001' , '010' , '011'        ],
                [''  , ''   , ''   , ''    , '000' , '101' , '010'        ],
                [''  , ''   , ''   , ''    , ''    , '100' , '001'        ],
                [''  , ''   , ''   , ''    , ''    , ''    , '0001'       ],
                [''  , ''   , ''   , ''    , ''    , ''    , '00001'      ],
                [''  , ''   , ''   , ''    , ''    , ''    , '000001'     ],
                [''  , ''   , ''   , ''    , ''    , ''    , '0000001'    ],
                [''  , ''   , ''   , ''    , ''    , ''    , '00000001'   ],
                [''  , ''   , ''   , ''    , ''    , ''    , '000000001'  ],
                [''  , ''   , ''   , ''    , ''    , ''    , '0000000001' ],
                [''  , ''   , ''   , ''    , ''    , ''    , '00000000001']]

def block_to_zigzag(input_array):
    # This is the block scan mapping for H264 (Converts 4x4 NDARRAY into 1x16 list)
    # Arranges high frequency coefficients to end of array
    # Done combinationally in a single cycle in hardware, or via registered loop iteration for speed
    # in (X,Y) column-row notation
    mapping_sequence = [(0,0) , (1,0) , (0,1) , (0,2) , \
                        (1,1) , (2,0) , (3,0) , (2,1) , \
                        (1,2) , (0,3) , (1,3) , (2,2) , \
                        (3,1) , (3,2) , (2,3) , (3,3)]
    
    enc_sequence = [input_array[x][y] for (y,x) in mapping_sequence]
    
    return enc_sequence

###
# Given a 4x4 input array, output a CAVLC coded string 
###
def CAVLC_enc(input_array):
    coeffs = []
    level_codes = []
    trailing_ones = []
    trailing_ones_count = 0
    zeros = []
    run = []
    output_stream = []

    # Data start mark (marks the coefficient where first nonzero coeff occurs)
    data_start_mark = 15

    # Convert the raster scan array to zigzag mode
    zigzag = block_to_zigzag(input_array)
    zigzag = [(idx, coeff) for idx, coeff in enumerate(zigzag)]
    zigzag.reverse()

    print(zigzag)
    # For each element, sort it into the appropriate bucket
    for idx, el in zigzag:
        if el == 0:
            zeros.append(idx)
        if el == 1 or el == -1:
            data_start_mark = idx
            trailing_ones_count += 1
            if len(trailing_ones) < 3:
                trailing_ones.append((idx, el))
            else:
                coeffs.append((idx, el))
        if abs(el) > 1:
            coeffs.append((idx, el))
    
    data_start_mark =  max( [max([x[0] for x in trailing_ones]), max([x[0] for x in trailing_ones])] )

    # Sanitize the zeros array to not include uncoded zeros
    # Zeros ID list has to be reversed -- zeros are assembled from end of array (HF first)
    zeros = [x for x in zeros if x < data_start_mark]
  
    #print("zeroes {0} coeffs {1}".format(zeros, coeffs))
    
    # Generate the coeff_token from LUT as fxn of num_coeffs and trailing_ones
    #print("T1s", trailing_ones)
    #print("coeffs", coeffs)
    #print("zeros", zeros)
    num_coeffs = len(coeffs) + len(trailing_ones)
    coeff_token = coeff_token0[num_coeffs][len(trailing_ones)]

    #print("Num coeffs: ", num_coeffs, "| T1s: ", trailing_ones_count, "| coeff_token = ", coeff_token)

    # Generate the levels encoding: 0, 3, 6, 12, 24, 48
    # H264 require bumping up the suffix length immediately for a complex array
    if(num_coeffs > 10 and trailing_ones_count < 3):
        suffix_length = 1
    else: 
        suffix_length = 0

    logger.debug("TotalCoeff (0 <= TotalCoeff <= 16): %i", num_coeffs)
    logger.debug("T1 (0 <= T1 <= 3): %i", len(trailing_ones))
    logger.debug("coeff_token: %s", coeff_token)
    logger.debug("Level VLCs:")

   
    for idx, coeff in coeffs:
        
        # Transform the coefficient into a levelcode
        # 0 is NOT a valid encoded coefficient :) 
        # Positive values are encoded as even
        # Negative as odd
        if coeff > 0:
            level_code = 2*coeff - 2
        elif coeff < 0:
            level_code = -2*coeff - 1
        else:
            level_code = 0
            print("Trying to encode 0 coeff. Halt")
            assert(0)

        # Saturate the prefixlength to a maximum of 15
        if (level_code >> suffix_length) >= 15:
            suffix_size = 12
        else:
            suffix_size = suffix_length

        # Generate the suffix 
        level_suffix_mask = int16((0x1 << suffix_size) - 1)
        level_suffix = level_code & level_suffix_mask
        level_suffix = bin(level_suffix)[2:]
        
        # Supress the suffix if suffixlength is 0
        level_suffix = [] if not suffix_size else level_suffix

        level_prefix_raw = level_code >> suffix_size
        level_prefix = (['0'] * level_prefix_raw) + ['1']

        # if (i == i_trailing + 1) and (i_trailing<3): 
        #     i_level_code = i_level_code - 2; 

        # Increment the suffix length if we are transitioning to a larger codebook
        threshold = ((2 ** (suffix_length-1) * 3) )
        if(abs(coeff) > threshold):
            suffix_length += 1 
        elif(suffix_length == 0):
            suffix_length = 1

        level_codes.append("".join(level_prefix) + "".join(level_suffix))
        #logger.debug("Level VLC[%i] suffixLength: %i level_prefix: %s level_suffix: %s", idx, suffix_length, "".join(level_prefix), "".join(level_suffix))

    # Now code the zeros
    print("len zeros" + str(len(zeros)))
    print("num coeffs" + str(num_coeffs))

    # Be careful; Table_zeros is 0 indexed for zero count but not for coeff count
    total_zeros = Table_zeros[num_coeffs-1][len(zeros)]
    
    # The run_before variable refers to the coeffs that are >1
    # It increments the position of the coeff in the decoded string
    # Not really about the zeros -- it's about coeffs and the distance between them
    zerosLeft = len(zeros)
    
    # For zero encoding, trailing ones count as coeffs
    coeffs += trailing_ones
    coeffs.sort(key = lambda i: i[0], reverse=True)
    
    prev_idx = data_start_mark
    
    for i, coeff in enumerate(coeffs):
        if(i < len(coeffs)):
            run_before = coeff[0] - coeffs[i+1][0]
        else:
            run_before = zerosLeft

        code = Table_run[run_before][min(zerosLeft,6)]
        print("Coding coeff(%i): %i" % (idx, coeff[1]))
        print("code " + str(code) +  ": run_before " + str(run_before) + " zeros left " + str(zerosLeft) )
        run.append(code)
        if(run_before):
            zerosLeft -= run_before

    print("total zeros " + str(total_zeros))
    print("run" + str(run))

    return "".join(generate_cavlc_bits(coeff_token, level_codes, trailing_ones, total_zeros, run))

def generate_cavlc_bits(coeff_token, level_codes, trailing_ones, total_zeros, run):
    output = []

    # Encode the coeff token
    output.append(coeff_token)

    # Encode the sign bits for trailing ones (1 for +1 0 for -1)
    for (idx,el) in trailing_ones:
        output.append('0') if el == 1 else output.append('1')

    # Encode the bits for the level codes
    for el in level_codes:
        output.append(el)

    # Encode the zeros values
    output.append(total_zeros)

    for el in run:
        output.append(el)

    return output
    

def CAVLC_enc_deprecated(input_array):

    encoded_sequence = []

    # Block scan the input array
    sequence = sequence_code(input_array)

    # Get number of nonzero coefficients
    coeff_count = 0
    T1s_count = 0
    T1_mode = 1
    T1s = []
    levels = []
    run_before = [0] * 16
    totalZeroes = 0
    suffixLength = 0 # Note that we don't support the H264 bump of suffixlength for coeffcount > 10

    for idx, el in enumerate(sequence.reverse()):
        if el != 0:
            # This logic supports the trailing ones mode
            if (el == 1 or el == -0) and T1s_count < 3 and T1_mode:
                T1s += 1
                if(el == 1):
                   encoded_sequence.append(0)
                if(el == -1):
                    encoded_sequence.append(1)

                coeff_count += 1
            else:
                # Once we encounter the first nonzero element, or more than 3 are encoded, we enter 
                # coefficient exclusive mode
                T1_mode = 0
                coeff_count += 1
                encoded_sequence.append(level_LUT(el, suffixLength))

        # Otherwise this is a zero, mark its location if it's after the first coefficient
        else:
                if(coeff_count > 0):
                    run_before[coeff_count] = run_before[coeff_count] + 1
                    totalZeroes += 1
        # Now assemble the output array: {coeff_count, T1s, levels, totalzeroes, run_before}
    
    return []

def CAVLC_dev(sequence):
    return []