# PyH264

An educational implementation of H.264 video codec in pure Python. Designed for learning and understanding video compression concepts, not for production use.

This implementation prioritizes clarity over performance, making it ideal for:
- Learning H.264 fundamentals
- Understanding DCT, quantization, and entropy coding
- Prototyping HDL (hardware) implementations
- Experimenting with video compression algorithms

## Features

### Supported
- Configurable macroblock sizes (8x8, 16x16, 32x32)
- Configurable transform block sizes (4x4, 8x8)
- Web-based GUI dashboard for interactive encoding
- DCT (Discrete Cosine Transform) and inverse DCT
- Quantization with configurable Qp
- Intra-frame prediction (DC, Horizontal, Vertical modes)
- Exp-Golomb variable length coding (VLC)
- Multiple input formats: Y4M, JPEG, PNG, BMP, GIF, TIFF
- Command-line interface
- Configurable video dimensions

### Not Yet Supported
- Inter-frame compression (P/B frames)
- Motion estimation and compensation
- CABAC entropy coding
- NAL unit / transport stream generation
- Deblocking filter

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Command Line Interface

```bash
# Encode an image to H264 bitstream
python -m h264 encode image.png -o output.h264bits

# Encode with preview
python -m h264 encode photo.jpg --show

# Decode bitstream back to image
python -m h264 decode output.h264bits -o decoded.png --width 640 --height 480

# Run test pattern
python -m h264 test --width 128 --height 128 --show

# Show file information
python -m h264 info image.png
```

### Supported Input Formats
- **Images**: JPEG, PNG, BMP, GIF, TIFF
- **Video**: YUV4MPEG2 (.y4m)

## Web GUI

The project includes an interactive web dashboard for visualizing the H.264 encoding process.

### Running the GUI

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the Flask server
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

### GUI Features

- **Image Upload**: Drag and drop or browse to upload any image
- **Test Patterns**: Generate gradient, checkerboard, or circle patterns
- **Encoding Parameters**:
  - **QP (0-51)**: Quantization parameter - lower values = higher quality, larger files
  - **MB Size (8/16/32)**: Macroblock size in pixels
  - **TB Size (4/8)**: Transform block size in pixels
- **Live Visualization**:
  - Original image
  - Reconstructed image (after encode/decode)
  - Difference image (4× amplified)
  - CTU Structure (macroblock and transform block grid overlay)
- **Statistics**: Compression ratio, bits per pixel, PSNR, encode time
- **Zoom**: Hover over images to see pixel-level detail

Parameters automatically re-encode when changed, allowing real-time exploration of quality vs compression tradeoffs.

## Python API

```python
from h264.H264 import H264

# Load from image file (auto-detects dimensions)
encoder = H264()
encoder.load_image("photo.jpg")

# Or load from Y4M video
with open("video.y4m", "rb") as f:
    encoder.load_video(f)

# Or create a test pattern
encoder = H264(width=640, height=480)
encoder.load_pattern()

# Generate bitstream
bitstream = encoder.compress_frame(0)

# Or compress/decompress in place
encoder.compress_inplace()
encoder.decompress_inplace()

# Display a frame
encoder.show_frame(0)
```

## Project Structure

```
PyH264/
├── app.py               # Flask web GUI server
├── templates/
│   └── index.html       # Web dashboard UI
├── static/uploads/      # Uploaded images (gitignored)
├── h264/
│   ├── H264.py          # Main encoder/decoder class
│   ├── Frame.py         # Frame container (collection of slices)
│   ├── Slice.py         # Slice container (collection of macroblocks)
│   ├── MacroBlock.py    # 16x16 macroblock (16 transform blocks)
│   ├── TransformBlock.py # 4x4 transform block with DCT/quantization
│   ├── DCT.py           # DCT transform functions
│   ├── VLC.py           # Exp-Golomb variable length coding
│   └── CAVLC.py         # Context-adaptive VLC (partial)
├── tests/
│   ├── testH264.py      # Integration tests
│   ├── testDCT.py       # DCT unit tests
│   ├── testVLC.py       # VLC unit tests
│   ├── testCAVLC.py     # CAVLC unit tests
│   ├── testIntra.py     # Intra prediction tests
│   ├── testFrameSlice.py # Frame/Slice tests
│   └── testPerf.py      # Performance benchmarks
├── sequences/           # Test video files
├── requirements.txt
└── README.md
```

## Running Tests

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests with unittest
python -m unittest discover tests/

# Run with pytest (more verbose)
python -m pytest tests/ -v

# Run specific test file
python -m unittest tests.testH264

# Run performance benchmarks
python tests/testPerf.py
```

## Architecture

The codec follows a hierarchical structure:

```
H264 (Encoder)
  └── Frame (720p = 45 slices)
        └── Slice (1 row of macroblocks)
              └── MacroBlock (16x16 pixels = 16 transform blocks)
                    └── TransformBlock (4x4 pixels)
```

### Encoding Pipeline
1. **Load** - Parse YUV4MPEG2 or generate test pattern
2. **Partition** - Split frame into slices → macroblocks → 4x4 blocks
3. **Predict** - Apply intra prediction (DC/H/V modes)
4. **Transform** - Apply 4x4 DCT to residual blocks
5. **Quantize** - Reduce precision based on Qp
6. **Entropy Code** - Exp-Golomb encode coefficients

### Decoding Pipeline
1. **Entropy Decode** - Parse Exp-Golomb VLC
2. **Dequantize** - Restore coefficient magnitudes
3. **Inverse Transform** - Apply inverse DCT
4. **Reconstruct** - Add prediction to residual

## Performance

This implementation is intentionally slow for educational clarity:
- Each macroblock is a Python object
- Transforms use explicit loops (not optimized NumPy)
- No parallelization

Typical performance: ~1000 pixels/second (a 720p frame takes several seconds)

## Dependencies

- Python 3.8+
- NumPy >= 1.20.0
- Pillow >= 8.0.0
- pytest >= 7.0.0 (for testing)

## License

Educational use. See LICENSE file for details.

## Contributing

Contributions welcome! Areas that need work:
- Complete CAVLC encoder implementation
- Add inter-frame prediction
- Implement motion estimation
- Add NAL unit packaging
