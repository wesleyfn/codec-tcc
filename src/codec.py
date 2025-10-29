import numpy as np
import pydicom
import os, io
import random
import struct
from datetime import datetime
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, DeflatedExplicitVRLittleEndian
import zlib
import subprocess
import tempfile

def save_dicom_file(dicom_dataset: FileDataset, file_path: str):
    """Saves a DICOM dataset to a file."""
    dicom_dataset.save_as(file_path, write_like_original=False)
    print(f"\n>>>> DICOM file saved to: {file_path}\n")

def create_dicom_dataset(image_array: np.ndarray) -> FileDataset:
    """Creates a simple DICOM dataset from a numpy array."""
    max_val = image_array.max()
    
    log_val = np.log2(float(max_val) + 1.0)
    bits_stored = int(np.ceil(log_val))
    bits_stored = max(1, bits_stored)

    if image_array.ndim != 2: raise ValueError("Image must be 2D (grayscale).")

    if image_array.dtype not in [np.uint8, np.uint16]:
        raise ValueError("Image array must be uint8 or uint16.")

    SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    dicom_dataset = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    dicom_dataset.is_little_endian = True
    dicom_dataset.is_implicit_VR = False

    dicom_dataset.PatientName = "STEGO^"
    dicom_dataset.PatientID = "123456"
    dicom_dataset.StudyInstanceUID = generate_uid()
    dicom_dataset.SeriesInstanceUID = generate_uid()
    dicom_dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dicom_dataset.SOPClassUID = SOP_CLASS_UID

    now = datetime.now()
    dicom_dataset.StudyDate = now.strftime("%Y%m%d")
    dicom_dataset.StudyTime = now.strftime("%H%M%S")
    dicom_dataset.SeriesDate = now.strftime("%Y%m%d")
    dicom_dataset.ContentDate = now.strftime("%Y%m%d")
    dicom_dataset.ContentTime = now.strftime("%H%M%S")

    dicom_dataset.Modality = "OT"
    dicom_dataset.InstanceNumber = "1"
    dicom_dataset.SeriesNumber = "1"

    dicom_dataset.Rows, dicom_dataset.Columns = image_array.shape
    dicom_dataset.SamplesPerPixel = 1
    dicom_dataset.PhotometricInterpretation = "MONOCHROME2"
    dicom_dataset.PixelRepresentation = 0

    bits_allocated = image_array.dtype.itemsize * 8
    dicom_dataset.BitsAllocated = bits_allocated
    dicom_dataset.BitsStored = min(bits_stored, bits_allocated)
    dicom_dataset.HighBit = dicom_dataset.BitsStored - 1

    dicom_dataset.PixelData = image_array.tobytes()
    
    return dicom_dataset

def compress_image_data(image_array: np.ndarray, codec: str) -> bytes:
    """Compresses image data using the specified codec."""
    print(f"   - Compressing with {codec.upper()}...")
       
    if codec == 'jxl':
        with tempfile.TemporaryDirectory() as td:
            temp_input_png = os.path.join(td, 'in.png')
            temp_output_jxl = os.path.join(td, 'out.jxl')
            pil_img = Image.fromarray(image_array)
            pil_img.save(temp_input_png)
            cmd = ['cjxl', temp_input_png, temp_output_jxl, '-d', '0', '-e', '9']
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with open(temp_output_jxl, 'rb') as f:
                    return f.read()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"cjxl failed: {e.stderr.decode(errors='ignore')}")

    elif codec in ['j2k', 'jls']:
        with tempfile.TemporaryDirectory() as td:
            temp_input = os.path.join(td, 'in.dcm')
            temp_output = os.path.join(td, 'out.dcm')
            ds_uncompressed = create_dicom_dataset(image_array)
            ds_uncompressed.save_as(temp_input)
            cmd = ['gdcmconv', '--j2k' if codec == 'j2k' else '--jpegls', temp_input, temp_output]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                with open(temp_output, 'rb') as f:
                    return f.read()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"gdcmconv failed: {e.stderr.decode(errors='ignore')}")

    elif codec == 'png':
        dicom_dataset = create_dicom_dataset(image_array)
        dicom_dataset.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
        buffer = io.BytesIO()
        dicom_dataset.save_as(buffer)
        return buffer.getvalue()

    else:
        raise ValueError(f"Unsupported codec: '{codec}'")

def decompress_image_data(compressed_bytes: bytes, codec: str) -> np.ndarray:
    """Decompresses image bytes based on the specified codec."""
    if codec == 'jxl':
        temp_in, temp_out = 'temp_decompress.jxl', 'temp_decompress.png'
        try:
            with open(temp_in, 'wb') as f:
                f.write(compressed_bytes)
            cmd = ['djxl.exe', temp_in, temp_out]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with Image.open(temp_out) as img: 
                return np.array(img)
        finally:
            if os.path.exists(temp_in): os.remove(temp_in)
            if os.path.exists(temp_out): os.remove(temp_out)

    elif codec in ['j2k', 'jls', 'png']:
        dicom_dataset = pydicom.dcmread(io.BytesIO(compressed_bytes), force=True)
        return dicom_dataset.pixel_array
        
    else:
        raise ValueError(f"Unsupported codec: '{codec}'")

def read_dicom_file(file_path: str) -> FileDataset:
    """Reads a DICOM file and returns the dataset."""
    return pydicom.dcmread(file_path)

def merge_bit_planes(global_planes: list, local_planes: list) -> np.ndarray:
    """Merges global and local bit planes into a single image array."""
    if not local_planes:
        raise ValueError("The list of local planes cannot be empty.")
    
    sample_plane = local_planes[0]
    num_local_planes = len(local_planes)
    total_bits = len(global_planes) + num_local_planes
    dtype = np.uint16 if total_bits > 8 else np.uint8

    global_image = np.zeros(sample_plane.shape, dtype=dtype)
    local_image = np.zeros(sample_plane.shape, dtype=dtype)

    for i, plane in enumerate(global_planes):
        shift = i + num_local_planes
        global_image |= (plane.astype(dtype) << shift)

    for i, plane in enumerate(local_planes):
        local_image |= (plane.astype(dtype) << i)

    return global_image | local_image

def convert_message_to_bits(message: str) -> str:
    """Converts a UTF-8 string message to a bit string."""
    encoded_bytes = message.encode('utf-8')
    return np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8)).tobytes()


def create_message_segments(local_planes: list, message_bits: str) -> (list, list, list):
    """Distributes the message into weighted segments for each plane."""
    num_local_planes = len(local_planes)
    total_bits = len(message_bits)
    weights = [(num_local_planes - i) ** 2 for i in range(num_local_planes)]
    total_weight = sum(weights)
    
    if total_weight == 0:
        distributed_sizes = [total_bits // num_local_planes] * num_local_planes
    else:
        distributed_sizes = [max(1, int((w / total_weight) * total_bits)) for w in weights]

    excess = sum(distributed_sizes) - total_bits
    if excess != 0:
        for i in range(abs(excess)):
            distributed_sizes[-(i + 1)] -= np.sign(excess)
    
    final_adjustment = total_bits - sum(distributed_sizes)
    distributed_sizes[0] += final_adjustment
    
    all_segments = {i: message_bits[sum(distributed_sizes[:i]):sum(distributed_sizes[:i+1])] for i in range(num_local_planes)}
        
    segment_indices = list(range(num_local_planes))
    random.seed(42)
    random.shuffle(segment_indices)
    
    processing_segments = [all_segments[idx] for idx in segment_indices]
    final_lengths = [len(all_segments[i]) for i in range(num_local_planes)]

    return processing_segments, final_lengths, segment_indices

def embed_message_in_planes(local_planes: list, message_bits: str, allowed_indices: np.ndarray, image_shape: tuple, start_offset: int = 0, align_across_planes: bool = False):
    """
    Embeds the message sequentially using allowed pixels and returns an embedding map.
    Map values: 0=unused, 1=used/no flip, 2=used/flipped.
    """
    h, w = image_shape

    if len(message_bits) > len(allowed_indices):
        raise ValueError(f"Message ({len(message_bits)} bits) is larger than image capacity ({len(allowed_indices)} bits).")

    segments, segments_lengths, segment_indices = create_message_segments(local_planes, message_bits)

    stego_planes = [p.copy() for p in local_planes]
    embedding_map = np.zeros((h, w), dtype=np.uint8)
    total_used = 0
    current_start_offset_in_allowed = start_offset

    for i, dest_idx in enumerate(segment_indices):
        segment = segments[i]
        stego_plane = stego_planes[dest_idx]
        num_bits = segments_lengths[dest_idx]

        if num_bits == 0:
            continue

        indices_to_use = allowed_indices[current_start_offset_in_allowed : current_start_offset_in_allowed + num_bits]
        y_coords, x_coords = np.unravel_index(indices_to_use, (h, w))

        original_pixels = stego_plane[y_coords, x_coords]
        msg_bits_arr = np.fromiter(segment, dtype=np.uint8, count=num_bits)

        stego_pixels = (original_pixels & 0xFE) | msg_bits_arr
        stego_plane[y_coords, x_coords] = stego_pixels

        xor_values = original_pixels ^ stego_pixels
        embedding_map[y_coords, x_coords] = 1 + xor_values
        
        total_used += num_bits

        if not align_across_planes:
            current_start_offset_in_allowed += num_bits

    return stego_planes, embedding_map, total_used, segments_lengths, segment_indices

def calculate_entropy(data_array: np.ndarray) -> float:
    """Calculates the Shannon entropy of an array."""
    counts = np.bincount(data_array.ravel())
    probabilities = counts[counts > 0] / data_array.size
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_mutual_information(bit_plane: np.ndarray, image_array: np.ndarray) -> float:
    """Calculates the mutual information between a bit plane and the image."""
    if not hasattr(calculate_mutual_information, '_cache'):
        calculate_mutual_information._cache = {}
    cache_key = (hash(bit_plane.tobytes()), hash(image_array.tobytes()))
    if cache_key in calculate_mutual_information._cache:
        return calculate_mutual_information._cache[cache_key]
    
    if bit_plane.min() == bit_plane.max() or image_array.min() == image_array.max():
        result = 0.0
        calculate_mutual_information._cache[cache_key] = result
        return result

    h_x = calculate_entropy(bit_plane)
    h_y = calculate_entropy(image_array)

    max_val = int(image_array.max())
    combined_indices = bit_plane.ravel().astype(np.int32) * (max_val + 1) + image_array.ravel().astype(np.int32)
    h_xy = calculate_entropy(combined_indices)
    
    mi = h_x + h_y - h_xy
    result = max(0.0, mi)
    calculate_mutual_information._cache[cache_key] = result
    return result

def decompose_image_adaptively(image_array: np.ndarray, beta: float = 0.8, nbits: int = None) -> (list, list):
    """Adaptively decomposes the image into global and local bit planes."""
    nbits = image_array.dtype.itemsize * 8 if nbits is None else nbits
    print(f"   - Effective bit depth: {nbits}")
    bit_planes = [(image_array >> i) & 1 for i in range(nbits)]
    total_info = calculate_entropy(image_array)
    target_info = beta * total_info
    print(f"   - Total image information: {total_info:.4f}")
    print(f"   - Retention target ({beta*100}%): {target_info:.4f}")
    
    cumulative_info = 0.0
    s = 1
    for i in range(nbits):
        mi = calculate_mutual_information(bit_planes[i], image_array)
        cumulative_info += mi
        if cumulative_info >= target_info:
            s = i + 1
            break
    
    local_planes = bit_planes[:s]
    global_planes = bit_planes[s:]
    return global_planes, local_planes

def create_steganography_header(codec: str, s: int, segments_lengths: list, segments_indices: list, bitmaps_blob_size: int, 
                  width: int, height: int, start_offset: int, align_across_planes: bool,
                  block_size: int, threshold_factor: float) -> bytes:
    """Creates the header for the steganography container file."""
    codec_map = {'png': 1, 'j2k': 2, 'jls': 3, 'jxl': 4}
    codec_id = codec_map.get(codec.lower(), 0)
    align_flag = 1 if align_across_planes else 0

    header_format = '>BBBBHHH B f'
    header_parts = [1, codec_id, s, align_flag, width, height, start_offset, block_size, threshold_factor]

    header_format += f'{s}H'
    header_parts.extend(segments_lengths)

    header_format += f'{s}B'
    header_parts.extend(segments_indices)

    header_format += 'I'
    header_parts.append(bitmaps_blob_size)
    
    packed_header = struct.pack(header_format, *header_parts)
    
    final_header = struct.pack('>I', len(packed_header)) + packed_header
    
    print(" HEADER CREATED:")
    print(f"   - Block Size: {block_size}")
    print(f"   - Threshold Factor: {threshold_factor}")
    return final_header

def parse_steganography_file(filepath: str):
    codec_map = {1: 'png', 2: 'j2k', 3: 'jls', 4: 'jxl'}
    with open(filepath, 'rb') as f:
        if f.read(4) != b'STGC':
            raise ValueError("Invalid file or incorrect signature.")
        
        header_length = struct.unpack('>I', f.read(4))[0]
        header_data = f.read(header_length)
        
        base_format = '>BBBBHHH B f'
        base_size = struct.calcsize(base_format)
        version, codec_id, s, align_flag, width, height, start_offset, block_size, threshold_factor = struct.unpack(base_format, header_data[:base_size])
        
        cursor = base_size
        seg_lengths_format = f'>{s}H'
        seg_lengths_size = struct.calcsize(seg_lengths_format)
        segments_lengths = list(struct.unpack(seg_lengths_format, header_data[cursor:cursor+seg_lengths_size]))
        cursor += seg_lengths_size
        
        seg_indices_format = f'>{s}B'
        seg_indices_size = struct.calcsize(seg_indices_format)
        segments_indices = list(struct.unpack(seg_indices_format, header_data[cursor:cursor+seg_indices_size]))
        cursor += seg_indices_size
        
        bitmaps_blob_size = struct.unpack('>I', header_data[cursor:cursor+4])[0]
        
        bitmaps_data = f.read(bitmaps_blob_size)
        stego_image_data = f.read()

        metadata = {
            'version': version, 'codec': codec_map.get(codec_id, 'unknown'), 's': s,
            'align_flag': align_flag, 'width': width, 'height': height,
            'start_offset': start_offset, 'segments_lengths': segments_lengths,
            'segments_indices': segments_indices,
            'block_size': block_size, 'threshold_factor': threshold_factor
        }
    return metadata, bitmaps_data, stego_image_data

def create_steganography_container(filename: str, header_bytes: bytes, bitmap_bytes: bytes, stego_compressed: bytes) -> int:
    """Creates the final steganography container file."""
    with open(filename, "wb") as f:
        f.write(b"STGC")
        f.write(header_bytes)
        f.write(bitmap_bytes)
        f.write(stego_compressed)
    return os.path.getsize(filename)

def extract_message_and_restore_planes(stego_planes: list, embedding_map: np.ndarray, metadata: dict) -> (str, list):
    """Extracts the message and restores the original planes using the embedding map."""
    num_local_planes = metadata['s']
    h, w = metadata['height'], metadata['width']
    align_across_planes = metadata.get('align_flag', 0)
    current_start_offset_in_allowed = metadata.get('start_offset', 0)

    allowed_indices = np.where(embedding_map.ravel() > 0)[0]
    xor_map = (embedding_map == 2).astype(np.uint8)

    message_segments = [''] * num_local_planes
    restored_planes = [p.copy() for p in stego_planes]
    
    segment_indices = metadata['segments_indices']
    segments_lengths = metadata['segments_lengths']

    for i, dest_plane_idx in enumerate(segment_indices):
        stego_plane = stego_planes[dest_plane_idx]
        plane_to_restore = restored_planes[dest_plane_idx]
        num_bits = segments_lengths[dest_plane_idx]
        
        if num_bits == 0:
            continue

        indices_to_use = allowed_indices[current_start_offset_in_allowed : current_start_offset_in_allowed + num_bits]
        y_coords, x_coords = np.unravel_index(indices_to_use, (h, w))
        
        extracted_bits = stego_plane[y_coords, x_coords] & 1
        message_segments[dest_plane_idx] = ''.join(map(str, extracted_bits))
        
        xor_diff = xor_map[y_coords, x_coords]
        original_lsb = extracted_bits ^ xor_diff
        plane_to_restore[y_coords, x_coords] = (plane_to_restore[y_coords, x_coords] & 0xFE) | original_lsb
        
        if not align_across_planes:
            current_start_offset_in_allowed += num_bits

    all_bits = ''.join(message_segments)
    message_bytes = bytearray(int(all_bits[i:i+8], 2) for i in range(0, len(all_bits), 8) if len(all_bits[i:i+8]) == 8)
    
    message = message_bytes.decode('utf-8', errors='replace')
    
    return message, restored_planes

def extract_local_bit_planes(stego_array: np.ndarray, num_local_planes: int) -> list:
    """Extracts the local bit planes from a stego image array."""
    return [(stego_array >> i) & 1 for i in range(num_local_planes)]

def decode_steganography_container(filepath: str, output_prefix: str = "decoded"):
    """
    Decodes a .bin file, extracting the message and recovering the original image.
    """
    print(f"\n[..] Decoding file: {filepath}")
    
    metadata, embedding_map_blob, stego_image_data = parse_steganography_file(filepath)
    num_local_planes = metadata['s']
    codec = metadata['codec']
    print(f"   - Codec detected: {codec}")
    print(f"   - Local planes (s): {num_local_planes}")

    stego_array = decompress_image_data(stego_image_data, codec)

    print("[..] Loading embedding map from binary file...")
    w, h = metadata['width'], metadata['height']
    embedding_map_bytes = zlib.decompress(embedding_map_blob)
    embedding_map = np.frombuffer(embedding_map_bytes, dtype=np.uint8).reshape((h, w))
    print("   - Map loaded successfully.")

    nbits = stego_array.dtype.itemsize * 8
    all_stego_planes = [(stego_array >> i) & 1 for i in range(nbits)]
    stego_local_planes = all_stego_planes[:num_local_planes]
    global_planes = all_stego_planes[num_local_planes:]

    print("[..] Extracting message and restoring planes...")
    message, restored_local_planes = extract_message_and_restore_planes(
        stego_local_planes, embedding_map, metadata
    )

    message_file = f"{output_prefix}_message.txt"
    with open(message_file, 'w', encoding='utf-8') as f:
        f.write(message)
    print(f"[OK] Message saved to: {message_file}")
    
    print("[..] Reconstructing original image...")
    restored_image_array = merge_bit_planes(global_planes, restored_local_planes)

    print("[..] Creating DICOM file for the original image...")
    dicom_dataset = create_dicom_dataset(restored_image_array)
    dicom_file = f"{output_prefix}_image.dcm"
    save_dicom_file(dicom_dataset, dicom_file)
    
    return message, dicom_dataset

def create_embedding_capacity_map(image_array: np.ndarray, block_size: int = 8, threshold_factor: float = 1.0) -> (np.ndarray, np.ndarray):
    """Analyzes the image to find complex regions suitable for embedding."""
    print(f"   - Creating capacity map with {block_size}x{block_size} blocks...")
    h, w = image_array.shape

    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded_image = np.pad(image_array, ((0, h_pad), (0, w_pad)), 'edge')
    padded_h, padded_w = padded_image.shape
    
    blocks = padded_image.reshape(padded_h // block_size, block_size, padded_w // block_size, block_size).transpose(0, 2, 1, 3)
    block_stds = np.std(blocks, axis=(2, 3))
    non_zero_stds = block_stds[block_stds > 0]
    
    if non_zero_stds.size == 0:
        raise ValueError("Image is completely flat; adaptive steganography is not possible.")

    adaptive_threshold = np.mean(non_zero_stds) * threshold_factor
    print(f"   - Adaptive standard deviation threshold calculated: {adaptive_threshold:.4f}")

    complex_blocks_map = block_stds > adaptive_threshold
    capacity_map_padded = np.kron(complex_blocks_map, np.ones((block_size, block_size), dtype=np.uint8))
    capacity_map = capacity_map_padded[:h, :w]
    allowed_indices = np.where(capacity_map.ravel() == 1)[0]
    
    total_capacity = len(allowed_indices)
    print(f"   - Total capacity found: {total_capacity} bits ({total_capacity / (h*w) * 100:.2f}% of the image)")
    
    return capacity_map, allowed_indices

def save_embedding_map_as_image(embedding_map: np.ndarray, file_path: str):
    """Saves the embedding map as a PNG image with distinct colors for visualization."""
    h, w = embedding_map.shape
    rgb_map = np.zeros((h, w, 3), dtype=np.uint8)
    
    rgb_map[embedding_map == 0] = [0, 0, 0]      # Black: Not used
    rgb_map[embedding_map == 1] = [128, 128, 128] # Gray: Used, not flipped
    rgb_map[embedding_map == 2] = [255, 255, 255] # White: Used, flipped
    
    img = Image.fromarray(rgb_map)
    img.save(file_path)
    print(f"[OK] Embedding map saved to: {file_path}")

def run_steganography(input_dicom_file, message, output_dir, base_filename, beta, block_size, threshold_factor, codec='jxl', align_across_planes=False, start_offset=0):
    """
    Encapsulates the entire steganography logic into a parameterizable function.
    Returns the size of the generated binary file and the path of the recovered DICOM.
    """
    print(f"Running with Beta={beta}, BlockSize={block_size}, Threshold={threshold_factor}")
    
    image_array = read_dicom_file(input_dicom_file).pixel_array
    message_bits = convert_message_to_bits(message)

    global_planes, local_planes = decompose_image_adaptively(image_array, beta=beta)
    num_local_planes = len(local_planes)

    capacity_map, allowed_indices = create_embedding_capacity_map(image_array, block_size=block_size, threshold_factor=threshold_factor)

    capacity_map_img = Image.fromarray((capacity_map * 255).astype(np.uint8))
    capacity_map_path = os.path.join(output_dir, f"{base_filename}_capacity_map.png")
    capacity_map_img.save(capacity_map_path)
    print(f"[OK] Capacity map saved to: {capacity_map_path}")

    if len(message_bits) > len(allowed_indices):
        print(f"[!] WARNING: Message ({len(message_bits)} bits) too large for capacity ({len(allowed_indices)} bits). Skipping this combination.")
        return None, None

    stego_planes, embedding_map, _, segments_lengths, segment_indices = embed_message_in_planes(
        local_planes, message_bits, allowed_indices, image_array.shape, 
        start_offset=start_offset, align_across_planes=align_across_planes
    )
    
    embedding_map_path = os.path.join(output_dir, f"{base_filename}_embedding_map.png")
    save_embedding_map_as_image(embedding_map, embedding_map_path)

    stego_image_array = merge_bit_planes(global_planes, stego_planes)
    
    stego_dcm_path = os.path.join(output_dir, f"{base_filename}_stego_image.dcm")
    stego_ds = create_dicom_dataset(stego_image_array)
    save_dicom_file(stego_ds, stego_dcm_path)
    print(f"[OK] Intermediate stego image saved to: {stego_dcm_path}")

    compressed_bytes = compress_image_data(stego_image_array, codec)
    bitmaps_blob = zlib.compress(embedding_map.tobytes(), level=9)
    bitmaps_blob_size = len(bitmaps_blob)
    
    height, width = stego_image_array.shape
    header = create_steganography_header(
        codec=codec, s=num_local_planes, segments_lengths=segments_lengths,
        segments_indices=segment_indices, bitmaps_blob_size=bitmaps_blob_size,
        width=width, height=height, start_offset=start_offset,
        align_across_planes=align_across_planes,
        block_size=block_size, threshold_factor=threshold_factor
    )
    
    output_bin_file = os.path.join(output_dir, f"{base_filename}.bin")
    file_size = create_steganography_container(output_bin_file, header, bitmaps_blob, compressed_bytes)
    
    output_prefix = os.path.join(output_dir, f"{base_filename}_decoded")
    _, recovered_dicom_ds = decode_steganography_container(output_bin_file, output_prefix=output_prefix)
    recovered_dicom_path = f"{output_prefix}_image.dcm"
    
    print(f"[OK] Generation complete: {output_bin_file} ({file_size} bytes)")
    return file_size, recovered_dicom_path

def main():
    """Original main function for standalone script usage."""
    input_dicom_file = "images/dx_8b/111.dcm"
    if not os.path.exists(input_dicom_file):
        print(f"[Error] File not found: {input_dicom_file}")
        return

    try:
        message = "This is a test message for steganography! v3 for the final test. " * 24
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        beta = 0.3
        block_size = 4
        threshold_factor = 2
        base_filename = f"output_beta{beta}_bs{block_size}_tf{threshold_factor}"

        run_steganography(
            input_dicom_file, message, output_dir, base_filename,
            beta, block_size, threshold_factor, codec='jxl'
        )

    except Exception as e:
        print(f"[Error] An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()

