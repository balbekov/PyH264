# PyH264

Experimental implementation of H264 in pure Python. This is intended to be a learning codec, and it is intentionally inefficient.

This uses very inefficient data structures (for example, each macroblock is an object) for educational use. As a result, each frame takes several seconds to encode.

What's supported:

Fixed 16x16 macroblocks
Quantization, with configurable Qp
Intra frame compression

What's currently not supported:
Inter frame compression
I-frames
Motion prediction
Transport stream generation (bitmap only)