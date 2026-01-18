import zlib
import struct


from pathlib import Path

import kagglehub

from dotenv import load_dotenv


load_dotenv()

def download_dataset():
    path = Path(kagglehub.dataset_download("ananthu017/emotion-detection-fer"))
    return path


def parse_grayscale_png(data: bytes):
    # PNG Signature
    assert data[:8] == b'\x89PNG\r\n\x1a\n', "Does not match PNG signature."

    offset = 8
    chunks = {}
    idat_data = b''

    # Parse data chunks
    while offset < len(data):
        length = struct.unpack('>I', data[offset:offset+4])[0]
        chunk_type = data[offset+4:offset+8]
        chunk_data = data[offset+8:offset+8+length]
        
        if chunk_type == b'IHDR':
            # IHDR holds image metadata (Width: 4 bytes, Height: 4 bytes, Bit depth: 1 byte, Color type: 1 byte, and more)
            width, height, bit_depth, color_type = struct.unpack('>IIBB', chunk_data[:10])
            if color_type != 0:
                raise ValueError("This function only supports Grayscale (Color Type 0)")
        
        elif chunk_type == b'IDAT':
            # There can be multiple IDAT chunks; concatenate them
            idat_data += chunk_data
        
        elif chunk_type == b'IEND':
            break
            
        # Move to next chunk: length(4) + type(4) + data(length) + crc(4)
        offset += length + 12

    # Decompress IDAT data with zlib
    decompressed = zlib.decompress(idat_data)
    
    pixels = []
    stride = width + 1  # width + 1 byte for the filter type per row
    filter_types = set()
    
    for y in range(height):
        row_start = y * stride
        filter_type = decompressed[row_start]
        row_data = decompressed[row_start + 1 : row_start + stride]
        
        if filter_type == 0:
            pixels.append(list(row_data))
        elif filter_type == 1: # Sub
            bpp = 1  # bytes per pixel for grayscale
            recon = bytearray(len(row_data))
            for i in range(len(row_data)):
                left = recon[i - bpp] if i >= bpp else 0
                recon[i] = (row_data[i] + left) % 256
            pixels.append(list(recon))
        elif filter_type == 2: # Up
            prior = pixels[y - 1] if y > 0 else [0] * width
            recon = bytearray(len(row_data))
            for i in range(len(row_data)):
                up = prior[i]
                recon[i] = (row_data[i] + up) % 256
            pixels.append(list(recon))
        elif filter_type == 3: # Average
            bpp = 1  # bytes per pixel for grayscale
            prior = pixels[y - 1] if y > 0 else [0] * width
            recon = bytearray(len(row_data))
            for i in range(len(row_data)):
                left = recon[i - bpp] if i >= bpp else 0
                up = prior[i]
                avg = (left + up) // 2
                recon[i] = (row_data[i] + avg) % 256
            pixels.append(list(recon))
        elif filter_type == 4: # Paeth
            bpp = 1  # bytes per pixel for grayscale
            prior = pixels[y - 1] if y > 0 else [0] * width
            recon = bytearray(len(row_data))
            for i in range(len(row_data)):
                left = recon[i - bpp] if i >= bpp else 0
                up = prior[i]
                left_prior = pixels[y - 1][i - bpp] if i >= bpp and y > 0 else 0
                paeth = left + up - left_prior
                recon[i] = (row_data[i] + paeth) % 256
            pixels.append(list(recon))

    import matplotlib.pyplot as plt
    plt.imshow(pixels, cmap='gray')
    plt.show()

    return pixels

def base10_to_hex(n: int):
    return hex(n)[2:].upper()
