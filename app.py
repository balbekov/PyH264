#!/usr/bin/env python3
"""
PyH264 Dashboard - Flask-based web interface for the educational H.264 codec
"""

import os
import io
import base64
import time
import logging
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np

from h264.H264 import H264
from h264.DCT import dct2

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)


def image_to_base64(img_array, mode='L'):
    """Convert numpy array to base64 encoded PNG."""
    img = Image.fromarray(img_array, mode)
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def get_dct_visualization(block_4x4):
    """Generate a visualization of DCT coefficients."""
    dct_coeffs = dct2(block_4x4.astype(np.float64))
    # Normalize to 0-255 for visualization
    dct_norm = np.abs(dct_coeffs)
    if dct_norm.max() > 0:
        dct_norm = (dct_norm / dct_norm.max() * 255).astype(np.uint8)
    else:
        dct_norm = np.zeros_like(dct_coeffs, dtype=np.uint8)
    # Scale up for visibility
    dct_scaled = np.repeat(np.repeat(dct_norm, 16, axis=0), 16, axis=1)
    return dct_scaled


def generate_ctu_visualization(frame_data, mb_size, tb_size):
    """Generate a visualization of the Coding Tree Unit structure.
    
    Creates an RGBA image showing the original frame with:
    - Macroblock boundaries as 1px cyan boxes
    - Transform block boundaries as 1px violet boxes
    
    Args:
        frame_data: Grayscale image as numpy array
        mb_size: Macroblock size (8, 16, or 32)
        tb_size: Transform block size (4 or 8)
    
    Returns:
        Base64 encoded PNG image
    """
    height, width = frame_data.shape
    
    # Create RGBA image from grayscale
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = frame_data  # R
    rgba[:, :, 1] = frame_data  # G
    rgba[:, :, 2] = frame_data  # B
    rgba[:, :, 3] = 255          # A (fully opaque)
    
    # Colors for grid lines (RGBA)
    mb_color = (34, 211, 238, 255)   # Cyan for macroblocks
    tb_color = (167, 139, 250, 255)  # Violet for transform blocks
    
    # Draw transform block boundaries first (underneath MB boundaries)
    for y in range(0, height, tb_size):
        rgba[y, :, :] = tb_color
    for x in range(0, width, tb_size):
        rgba[:, x, :] = tb_color
    
    # Draw macroblock boundaries on top (thicker visual emphasis)
    for y in range(0, height, mb_size):
        rgba[y, :, :] = mb_color
    for x in range(0, width, mb_size):
        rgba[:, x, :] = mb_color
    
    # Draw right and bottom edges
    if height > 0:
        rgba[height - 1, :, :] = mb_color
    if width > 0:
        rgba[:, width - 1, :] = mb_color
    
    return image_to_base64(rgba, mode='RGBA')


@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')


@app.route('/api/encode', methods=['POST'])
def encode_image():
    """Encode an uploaded image and return statistics."""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Get encoding parameters from form data
        qp = int(request.form.get('qp', 26))
        mb_size = int(request.form.get('mb_size', 16))
        tb_size = int(request.form.get('tb_size', 4))
        
        # Validate parameters
        qp = max(0, min(51, qp))
        mb_size = mb_size if mb_size in [8, 16, 32] else 16
        tb_size = tb_size if tb_size in [4, 8] else 4
        if mb_size % tb_size != 0:
            tb_size = 4  # Fall back to safe default
        
        # Load image from upload
        img = Image.open(file.stream)
        img_gray = img.convert('L')
        orig_width, orig_height = img_gray.size
        
        # Create encoder with parameters
        encoder = H264(qp=qp, mb_size=mb_size, tb_size=tb_size)
        
        # Pad dimensions to multiples of mb_size
        width = ((orig_width + mb_size - 1) // mb_size) * mb_size
        height = ((orig_height + mb_size - 1) // mb_size) * mb_size
        encoder.width = width
        encoder.height = height
        
        # Create padded frame
        frame_data = np.zeros((height, width), dtype=np.uint8)
        img_array = np.array(img_gray, dtype=np.uint8)
        frame_data[:orig_height, :orig_width] = img_array
        
        # Pad edges
        if orig_width < width:
            frame_data[:orig_height, orig_width:] = img_array[:, -1:].repeat(
                width - orig_width, axis=1)
        if orig_height < height:
            frame_data[orig_height:, :] = frame_data[orig_height-1:orig_height, :].repeat(
                height - orig_height, axis=0)
        
        # Load frame
        from h264.Frame import Frame
        encoder.frames.append(Frame(encoder, frame_data, WIDTH=width, HEIGHT=height,
                                    qp=qp, mb_size=mb_size, tb_size=tb_size))
        
        # Original image base64
        original_b64 = image_to_base64(frame_data)
        
        # Time the encoding
        start_time = time.time()
        
        # Compress in place
        encoder.compress_inplace()
        
        # Get bitstream
        bitstream = encoder.compress_frame(0)
        
        encode_time = time.time() - start_time
        
        # Get compressed statistics
        pixels = width * height
        bits = len(bitstream)
        bytes_size = bits // 8
        bpp = bits / pixels
        ratio = (pixels * 8) / bits if bits > 0 else 0
        
        # Decompress
        encoder.decompress_inplace()
        
        # Get reconstructed image
        reconstructed = encoder.frames[0].get_image()
        reconstructed_b64 = image_to_base64(reconstructed)
        
        # Calculate PSNR
        mse = np.mean((frame_data.astype(float) - reconstructed.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
        else:
            psnr = float('inf')
        
        # Get DCT visualization for first macroblock
        first_block = frame_data[0:4, 0:4]
        dct_viz = get_dct_visualization(first_block)
        dct_b64 = image_to_base64(dct_viz)
        
        # Get difference image
        diff = np.abs(frame_data.astype(np.int16) - reconstructed.astype(np.int16))
        diff_scaled = np.clip(diff * 4, 0, 255).astype(np.uint8)  # Amplify for visibility
        diff_b64 = image_to_base64(diff_scaled)
        
        # Structure info
        num_slices = len(encoder.frames[0].slices)
        num_macroblocks = sum(len(list(s)) for s in encoder.frames[0].slices)
        blocks_per_mb = (mb_size // tb_size) ** 2
        num_transform_blocks = num_macroblocks * blocks_per_mb
        
        # Generate CTU visualization
        ctu_b64 = generate_ctu_visualization(frame_data, mb_size, tb_size)
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'reconstructed': reconstructed_b64,
            'dct_sample': dct_b64,
            'difference': diff_b64,
            'ctu': ctu_b64,
            'stats': {
                'original_width': orig_width,
                'original_height': orig_height,
                'padded_width': width,
                'padded_height': height,
                'pixels': pixels,
                'bitstream_bits': bits,
                'bitstream_bytes': bytes_size,
                'bits_per_pixel': round(bpp, 3),
                'compression_ratio': round(ratio, 2),
                'psnr_db': round(psnr, 2) if psnr != float('inf') else 'Perfect',
                'encode_time_ms': round(encode_time * 1000, 1),
                'num_slices': num_slices,
                'num_macroblocks': num_macroblocks,
                'num_transform_blocks': num_transform_blocks,
                'qp': qp,
                'mb_size': mb_size,
                'tb_size': tb_size
            }
        })
        
    except Exception as e:
        logging.exception("Error encoding image")
        return jsonify({'error': str(e)}), 500


@app.route('/api/test-pattern', methods=['POST'])
def test_pattern():
    """Generate and encode a test pattern."""
    try:
        width = int(request.json.get('width', 64))
        height = int(request.json.get('height', 64))
        pattern_type = request.json.get('pattern', 'gradient')
        
        # Get encoding parameters
        qp = int(request.json.get('qp', 26))
        mb_size = int(request.json.get('mb_size', 16))
        tb_size = int(request.json.get('tb_size', 4))
        
        # Validate parameters
        qp = max(0, min(51, qp))
        mb_size = mb_size if mb_size in [8, 16, 32] else 16
        tb_size = tb_size if tb_size in [4, 8] else 4
        if mb_size % tb_size != 0:
            tb_size = 4  # Fall back to safe default
        
        # Ensure dimensions are multiples of mb_size
        width = ((width + mb_size - 1) // mb_size) * mb_size
        height = ((height + mb_size - 1) // mb_size) * mb_size
        
        # Cap at reasonable size for web demo
        width = min(width, 256)
        height = min(height, 256)
        
        # Generate pattern
        if pattern_type == 'gradient':
            pattern_func = lambda p: min(p, 255)
        elif pattern_type == 'checkerboard':
            pattern_func = lambda x, y: 255 if ((x // mb_size) + (y // mb_size)) % 2 == 0 else 0
        elif pattern_type == 'circles':
            cx, cy = width // 2, height // 2
            pattern_func = lambda x, y: int(255 * (1 - min(1, ((x-cx)**2 + (y-cy)**2) / (min(cx, cy)**2))))
        else:
            pattern_func = lambda p: min(p, 255)
        
        # Create encoder with parameters
        encoder = H264(width=width, height=height, qp=qp, mb_size=mb_size, tb_size=tb_size)
        
        # Generate custom pattern
        frame_data = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                if pattern_type in ['checkerboard', 'circles']:
                    frame_data[y, x] = pattern_func(x, y)
                else:
                    frame_data[y, x] = pattern_func(x)
        
        from h264.Frame import Frame
        encoder.frames.append(Frame(encoder, frame_data, WIDTH=width, HEIGHT=height,
                                    qp=qp, mb_size=mb_size, tb_size=tb_size))
        
        original_b64 = image_to_base64(frame_data)
        
        start_time = time.time()
        encoder.compress_inplace()
        bitstream = encoder.compress_frame(0)
        encode_time = time.time() - start_time
        
        encoder.decompress_inplace()
        reconstructed = encoder.frames[0].get_image()
        reconstructed_b64 = image_to_base64(reconstructed)
        
        pixels = width * height
        bits = len(bitstream)
        
        # Calculate PSNR
        mse = np.mean((frame_data.astype(float) - reconstructed.astype(float)) ** 2)
        psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
        
        # Difference image
        diff = np.abs(frame_data.astype(np.int16) - reconstructed.astype(np.int16))
        diff_scaled = np.clip(diff * 4, 0, 255).astype(np.uint8)
        diff_b64 = image_to_base64(diff_scaled)
        
        # Generate CTU visualization
        ctu_b64 = generate_ctu_visualization(frame_data, mb_size, tb_size)
        
        # Structure info
        num_slices = len(encoder.frames[0].slices)
        num_macroblocks = sum(len(list(s)) for s in encoder.frames[0].slices)
        blocks_per_mb = (mb_size // tb_size) ** 2
        num_transform_blocks = num_macroblocks * blocks_per_mb
        
        return jsonify({
            'success': True,
            'original': original_b64,
            'reconstructed': reconstructed_b64,
            'difference': diff_b64,
            'ctu': ctu_b64,
            'stats': {
                'width': width,
                'height': height,
                'pixels': pixels,
                'bitstream_bits': bits,
                'bitstream_bytes': bits // 8,
                'bits_per_pixel': round(bits / pixels, 3),
                'compression_ratio': round((pixels * 8) / bits, 2) if bits > 0 else 0,
                'psnr_db': round(psnr, 2) if psnr != float('inf') else 'Perfect',
                'encode_time_ms': round(encode_time * 1000, 1),
                'num_slices': num_slices,
                'num_macroblocks': num_macroblocks,
                'num_transform_blocks': num_transform_blocks,
                'qp': qp,
                'mb_size': mb_size,
                'tb_size': tb_size
            }
        })
        
    except Exception as e:
        logging.exception("Error with test pattern")
        return jsonify({'error': str(e)}), 500


@app.route('/api/info')
def codec_info():
    """Return information about the codec architecture."""
    return jsonify({
        'name': 'PyH264',
        'description': 'Educational H.264 video codec in pure Python',
        'architecture': {
            'hierarchy': ['H264 (Encoder)', 'Frame', 'Slice', 'MacroBlock (16×16)', 'TransformBlock (4×4)'],
            'encoding_pipeline': [
                'Load - Parse input image/video',
                'Partition - Split into slices → macroblocks → 4×4 blocks',
                'Predict - Apply intra prediction (DC/H/V modes)',
                'Transform - Apply 4×4 DCT to residual blocks',
                'Quantize - Reduce precision based on Qp',
                'Entropy Code - Exp-Golomb encode coefficients'
            ],
            'decoding_pipeline': [
                'Entropy Decode - Parse Exp-Golomb VLC',
                'Dequantize - Restore coefficient magnitudes',
                'Inverse Transform - Apply inverse DCT',
                'Reconstruct - Add prediction to residual'
            ]
        },
        'features': {
            'supported': [
                '16×16 macroblocks with 4×4 transform blocks',
                'DCT and inverse DCT',
                'Quantization with configurable Qp',
                'Intra-frame prediction (DC, H, V modes)',
                'Exp-Golomb variable length coding',
                'Multiple input formats (Y4M, JPEG, PNG, etc.)'
            ],
            'not_yet_supported': [
                'Inter-frame compression (P/B frames)',
                'Motion estimation and compensation',
                'CABAC entropy coding',
                'NAL unit / transport stream',
                'Deblocking filter'
            ]
        }
    })


if __name__ == '__main__':
    print("=" * 60)
    print("  PyH264 Dashboard")
    print("  Educational H.264 Video Codec")
    print("=" * 60)
    print("\n  Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, host='127.0.0.1', port=5000)

