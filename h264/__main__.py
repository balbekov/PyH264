#!/usr/bin/env python3
"""
PyH264 Command Line Interface

Usage:
    python -m h264 encode <input> [-o output] [--show]
    python -m h264 decode <input> [-o output] [--show]
    python -m h264 test [--width W] [--height H] [--show]
    python -m h264 info <input>
"""

import argparse
import sys
import os
import logging

from h264.H264 import H264


def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )


def cmd_encode(args):
    """Encode an image/video to H264 bitstream."""
    encoder = H264()
    
    # Load input file
    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == '.y4m':
        with open(input_path, 'rb') as f:
            encoder.load_video(f)
        logging.info(f"Loaded Y4M video: {encoder.width}x{encoder.height}")
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        encoder.load_image(input_path)
        logging.info(f"Loaded image: {encoder.width}x{encoder.height}")
    else:
        logging.error(f"Unsupported format: {ext}")
        logging.error("Supported formats: .y4m, .jpg, .jpeg, .png, .bmp, .gif, .tiff")
        return 1
    
    # Show original if requested
    if args.show:
        logging.info("Displaying original frame...")
        encoder.show_frame(0)
    
    # Compress to bitstream
    logging.info("Encoding frame...")
    bitstream = encoder.compress_frame(0)
    
    # Calculate stats
    pixels = encoder.width * encoder.height
    bits = len(bitstream)
    bpp = bits / pixels
    ratio = (pixels * 8) / bits
    
    logging.info(f"Bitstream: {bits} bits ({bits // 8} bytes)")
    logging.info(f"Bits per pixel: {bpp:.2f}")
    logging.info(f"Compression ratio: {ratio:.2f}x")
    
    # Save output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(bitstream)
        logging.info(f"Saved bitstream to: {args.output}")
    else:
        # Default output name
        output_path = os.path.splitext(input_path)[0] + '.h264bits'
        with open(output_path, 'w') as f:
            f.write(bitstream)
        logging.info(f"Saved bitstream to: {output_path}")
    
    return 0


def cmd_decode(args):
    """Decode H264 bitstream to image."""
    # Read bitstream
    with open(args.input, 'r') as f:
        bitstream = f.read()
    
    logging.info(f"Loaded bitstream: {len(bitstream)} bits")
    
    # Decode
    decoder = H264(width=args.width, height=args.height)
    decoder.load_bitstream(bitstream)
    
    logging.info("Decoded frame successfully")
    
    # Show if requested
    if args.show:
        decoder.show_frame(0)
    
    # Save output
    if args.output:
        from PIL import Image
        frame = decoder.frames[0].get_image()
        img = Image.fromarray(frame, 'L')
        img.save(args.output)
        logging.info(f"Saved decoded image to: {args.output}")
    
    return 0


def cmd_test(args):
    """Run test pattern encode/decode."""
    width = args.width
    height = args.height
    
    logging.info(f"Creating test pattern: {width}x{height}")
    
    # Create encoder with test pattern
    encoder = H264(width=width, height=height)
    encoder.load_pattern()
    
    if args.show:
        logging.info("Displaying original pattern...")
        encoder.show_frame(0)
    
    # Compress
    logging.info("Compressing...")
    encoder.compress_inplace()
    
    # Decompress
    logging.info("Decompressing...")
    encoder.decompress_inplace()
    
    if args.show:
        logging.info("Displaying reconstructed pattern...")
        encoder.show_frame(0)
    
    # Stats
    bitstream = encoder.compress_frame(0)
    pixels = width * height
    bits = len(bitstream)
    
    logging.info(f"Test complete!")
    logging.info(f"  Dimensions: {width}x{height}")
    logging.info(f"  Pixels: {pixels}")
    logging.info(f"  Bitstream: {bits} bits")
    logging.info(f"  Bits/pixel: {bits/pixels:.2f}")
    
    return 0


def cmd_info(args):
    """Show information about input file."""
    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()
    
    print(f"File: {input_path}")
    print(f"Size: {os.path.getsize(input_path)} bytes")
    
    if ext == '.y4m':
        with open(input_path, 'rb') as f:
            header = f.read(100).decode('utf-8', errors='ignore')
        print(f"Format: YUV4MPEG2")
        print(f"Header: {header[:50]}...")
    elif ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']:
        from PIL import Image
        img = Image.open(input_path)
        print(f"Format: {img.format}")
        print(f"Dimensions: {img.width}x{img.height}")
        print(f"Mode: {img.mode}")
    elif ext == '.h264bits':
        with open(input_path, 'r') as f:
            bits = f.read()
        print(f"Format: H264 bitstream")
        print(f"Bits: {len(bits)}")
    else:
        print(f"Format: Unknown ({ext})")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='PyH264 - Educational H.264 Encoder/Decoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m h264 encode image.png -o output.h264bits
  python -m h264 encode video.y4m --show
  python -m h264 decode output.h264bits -o decoded.png --width 640 --height 480
  python -m h264 test --width 128 --height 128 --show
  python -m h264 info image.png
        """
    )
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Encode command
    encode_parser = subparsers.add_parser('encode', help='Encode image/video to bitstream')
    encode_parser.add_argument('input', help='Input file (y4m, jpg, png, bmp)')
    encode_parser.add_argument('-o', '--output', help='Output bitstream file')
    encode_parser.add_argument('--show', action='store_true', help='Display frame')
    
    # Decode command
    decode_parser = subparsers.add_parser('decode', help='Decode bitstream to image')
    decode_parser.add_argument('input', help='Input bitstream file')
    decode_parser.add_argument('-o', '--output', help='Output image file')
    decode_parser.add_argument('--width', type=int, default=1280, help='Frame width')
    decode_parser.add_argument('--height', type=int, default=720, help='Frame height')
    decode_parser.add_argument('--show', action='store_true', help='Display frame')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test pattern')
    test_parser.add_argument('--width', type=int, default=64, help='Pattern width')
    test_parser.add_argument('--height', type=int, default=64, help='Pattern height')
    test_parser.add_argument('--show', action='store_true', help='Display frames')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show file information')
    info_parser.add_argument('input', help='Input file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    setup_logging(args.verbose)
    
    if args.command == 'encode':
        return cmd_encode(args)
    elif args.command == 'decode':
        return cmd_decode(args)
    elif args.command == 'test':
        return cmd_test(args)
    elif args.command == 'info':
        return cmd_info(args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

