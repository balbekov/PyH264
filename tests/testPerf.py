"""
Performance Benchmark Tests

Simple benchmarks for encoding/decoding performance.
These are not unit tests - they measure timing.
"""

from h264.MacroBlock import MacroBlock
from h264.TransformBlock import TransformBlock
from h264.Frame import Frame
from h264 import H264 as codec

import numpy as np
import time
import logging

logging.basicConfig(level=logging.WARNING)


def benchmark_transform_block():
    """Benchmark TransformBlock DCT/IDCT operations."""
    data = np.uint8(np.random.randint(0, 256, (4, 4)))
    iterations = 1000
    
    # DCT benchmark
    start = time.time()
    for _ in range(iterations):
        tb = TransformBlock(None, data.copy())
        tb.dct()
    dct_time = time.time() - start
    
    # IDCT benchmark
    tb = TransformBlock(None, data)
    tb.dct()
    start = time.time()
    for _ in range(iterations):
        tb_copy = TransformBlock(None, data)
        tb_copy.dct()
        tb_copy.idct()
    idct_time = time.time() - start
    
    print(f"TransformBlock DCT: {iterations} iterations in {dct_time:.3f}s ({iterations/dct_time:.0f} ops/sec)")
    print(f"TransformBlock DCT+IDCT: {iterations} iterations in {idct_time:.3f}s ({iterations/idct_time:.0f} ops/sec)")


def benchmark_macroblock():
    """Benchmark MacroBlock VLC operations."""
    data = np.uint8(np.random.randint(0, 256, (16, 16)))
    iterations = 100
    
    # VLC encode benchmark
    start = time.time()
    for _ in range(iterations):
        mb = MacroBlock(None, data.copy())
        vlc = mb.get_vlc()
    encode_time = time.time() - start
    
    # VLC decode benchmark
    mb = MacroBlock(None, data)
    vlc = mb.get_vlc()
    start = time.time()
    for _ in range(iterations):
        mb_dec = MacroBlock(None, None)
        mb_dec.set_vlc(vlc)
    decode_time = time.time() - start
    
    print(f"MacroBlock VLC encode: {iterations} iterations in {encode_time:.3f}s ({iterations/encode_time:.0f} ops/sec)")
    print(f"MacroBlock VLC decode: {iterations} iterations in {decode_time:.3f}s ({iterations/decode_time:.0f} ops/sec)")


def benchmark_frame(width=64, height=64):
    """Benchmark Frame compression."""
    print(f"\nFrame benchmark ({width}x{height}):")
    
    # Create test frame
    h264 = codec.H264(width=width, height=height)
    h264.load_pattern()
    
    # Compress benchmark
    start = time.time()
    h264.compress_inplace()
    compress_time = time.time() - start
    
    # Decompress benchmark
    start = time.time()
    h264.decompress_inplace()
    decompress_time = time.time() - start
    
    pixels = width * height
    print(f"  Compress: {compress_time:.3f}s ({pixels/compress_time:.0f} pixels/sec)")
    print(f"  Decompress: {decompress_time:.3f}s ({pixels/decompress_time:.0f} pixels/sec)")


def benchmark_bitstream(width=64, height=64):
    """Benchmark bitstream generation."""
    print(f"\nBitstream benchmark ({width}x{height}):")
    
    h264 = codec.H264(width=width, height=height)
    h264.load_pattern()
    
    start = time.time()
    bitstream = h264.compress_frame(0)
    encode_time = time.time() - start
    
    bits = len(bitstream)
    pixels = width * height
    compression_ratio = (pixels * 8) / bits  # Assuming 8 bits per pixel original
    
    print(f"  Encode: {encode_time:.3f}s")
    print(f"  Bitstream size: {bits} bits ({bits/8:.0f} bytes)")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Bits per pixel: {bits/pixels:.2f}")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("PyH264 Performance Benchmarks")
    print("=" * 60)
    
    print("\n--- TransformBlock Benchmarks ---")
    benchmark_transform_block()
    
    print("\n--- MacroBlock Benchmarks ---")
    benchmark_macroblock()
    
    print("\n--- Frame Benchmarks ---")
    benchmark_frame(64, 64)
    benchmark_frame(128, 128)
    
    print("\n--- Bitstream Benchmarks ---")
    benchmark_bitstream(64, 64)
    
    print("\n" + "=" * 60)
    print("Benchmarks complete")
    print("=" * 60)


if __name__ == '__main__':
    main()
