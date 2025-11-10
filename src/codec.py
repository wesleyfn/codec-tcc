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
import logging
import json

logger = logging.getLogger(__name__)

def save_dicom_file(dicom_dataset: FileDataset, file_path: str):
    """Saves a DICOM dataset to a file, ensuring compliant format."""
    dicom_dataset.save_as(file_path, enforce_file_format=True)
    logger.info(f"\t✔ DICOM file saved to: {file_path}")

# ==============================================================================
# DICOM METADATA HANDLING
# ==============================================================================

def serialize_dicom_value(value):
    """Converts DICOM values to JSON-serializable types."""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, pydicom.uid.UID):
        return str(value)
    elif isinstance(value, pydicom.multival.MultiValue):
        return [serialize_dicom_value(x) for x in value]
    elif isinstance(value, pydicom.valuerep.DSfloat):
        return float(value)
    elif isinstance(value, pydicom.valuerep.IS):
        return int(value)
    elif isinstance(value, pydicom.valuerep.PersonName):
        return str(value)
    elif isinstance(value, bytes):
        # Skip binary data that's too large
        if len(value) > 1000:  # Skip large binary fields
            return f"<binary_data_{len(value)}_bytes>"
        else:
            return value.hex()
    elif hasattr(value, '__str__'):
        return str(value)
    else:
        return f"<unserializable_{type(value).__name__}>"

def extract_dicom_metadata(dicom_dataset: FileDataset) -> str:
    """
    Extracts all DICOM metadata as a JSON string for embedding.
    Preserves the original metadata structure.
    """
    metadata_dict = {}
    
    # Extract all elements that are not pixel data
    for elem in dicom_dataset:
        # Skip pixel data and large binary fields
        if elem.tag == (0x7FE0, 0x0010):  # PixelData
            continue
        if elem.tag.group > 0x0008:  # Skip private tags for simplicity
            continue
            
        if hasattr(elem, 'value') and elem.value is not None:
            try:
                serialized_value = serialize_dicom_value(elem.value)
                if serialized_value is not None:
                    # Use tag string as key, e.g., "(0008,0020)"
                    metadata_dict[str(elem.tag)] = {
                        'value': serialized_value,
                        'VR': elem.VR if hasattr(elem, 'VR') else 'UN',
                        'name': elem.name if hasattr(elem, 'name') else 'Unknown'
                    }
            except Exception as e:
                logger.warning(f"Could not serialize tag {elem.tag}: {e}")
    
    # Add critical tags with descriptive names for easier recovery
    critical_tags = {}
    critical_fields = [
        'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
        'StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID',
        'StudyID', 'StudyDate', 'StudyTime', 'StudyDescription',
        'SeriesNumber', 'SeriesDescription', 'Modality',
        'InstanceNumber', 'ImagePositionPatient', 'ImageOrientationPatient',
        'PixelSpacing', 'SliceThickness', 'Manufacturer', 'ManufacturerModelName'
    ]
    
    for field in critical_fields:
        if hasattr(dicom_dataset, field):
            try:
                value = getattr(dicom_dataset, field)
                critical_tags[field] = serialize_dicom_value(value)
            except Exception as e:
                logger.warning(f"Could not extract critical tag {field}: {e}")
    
    metadata_dict['_critical_tags'] = critical_tags
    metadata_json = json.dumps(metadata_dict, indent=2)
    
    logger.info(f"\t✔ Extracted {len(metadata_dict)} DICOM metadata tags ({len(metadata_json)} bytes)")
    return metadata_json

def restore_dicom_metadata(dicom_dataset: FileDataset, metadata_json: str) -> FileDataset:
    """
    Restores DICOM metadata from JSON string back to dataset.
    """
    try:
        metadata_dict = json.loads(metadata_json)
        
        # Restore critical tags first
        if '_critical_tags' in metadata_dict:
            critical_tags = metadata_dict.pop('_critical_tags')
            for tag_name, value in critical_tags.items():
                if value is not None and hasattr(dicom_dataset, tag_name):
                    try:
                        # For critical tags, we try to preserve the original type
                        current_value = getattr(dicom_dataset, tag_name)
                        if isinstance(current_value, pydicom.valuerep.PersonName) and isinstance(value, str):
                            setattr(dicom_dataset, tag_name, pydicom.valuerep.PersonName(value))
                        elif isinstance(current_value, pydicom.uid.UID) and isinstance(value, str):
                            setattr(dicom_dataset, tag_name, pydicom.uid.UID(value))
                        else:
                            setattr(dicom_dataset, tag_name, value)
                    except Exception as e:
                        logger.warning(f"Could not restore critical tag {tag_name}: {e}")
        
        # Restore all other tags
        restored_count = 0
        for tag_str, tag_info in metadata_dict.items():
            try:
                # Convert tag string back to tuple
                tag = eval(tag_str)  # Safe in this context as it's from our own serialization
                if tag in dicom_dataset:
                    # For now, we set the value directly
                    # In a more sophisticated implementation, you'd handle VR types properly
                    if isinstance(tag_info, dict) and 'value' in tag_info:
                        dicom_dataset[tag].value = tag_info['value']
                        restored_count += 1
            except Exception as e:
                logger.debug(f"Could not restore tag {tag_str}: {e}")
        
        logger.info(f"\t✔ Restored {restored_count + len(critical_tags)} DICOM metadata tags")
        return dicom_dataset
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        raise

def create_clean_dicom_dataset(image_array: np.ndarray) -> FileDataset:
    """
    Creates a clean DICOM dataset with minimal metadata for stego image.
    This is what will be visible in the container - without the real metadata.
    """
    max_val = image_array.max()
    bits_stored = int(np.ceil(np.log2(float(max_val) + 1.0))) if max_val > 0 else 1
    bits_stored = max(1, bits_stored)

    SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    
    # Minimal fake metadata - this is the decoy
    ds.PatientName = "ANONYMIZED^PATIENT"
    ds.PatientID = "000000"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SOP_CLASS_UID

    now = datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.Modality = "OT"
    ds.InstanceNumber = "1"
    ds.SeriesNumber = "1"
    ds.Rows, ds.Columns = image_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0

    bits_allocated = image_array.dtype.itemsize * 8
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = min(bits_stored, bits_allocated)
    ds.HighBit = ds.BitsStored - 1
    ds.PixelData = image_array.tobytes()
    
    logger.info("\t✔ Created clean DICOM dataset with decoy metadata")
    return ds

# ==============================================================================
# IMAGE COMPRESSION / DECOMPRESSION
# ==============================================================================

def compress_image_data(image_array: np.ndarray, codec: str) -> bytes:
    """Compresses image data using the specified codec."""
    logger.info(f"\t- Compressing with {codec.upper()}...")
    
    if codec == 'jxl':
        with tempfile.TemporaryDirectory() as td:
            temp_input_png = os.path.join(td, 'in.png')
            temp_output_jxl = os.path.join(td, 'out.jxl')
            Image.fromarray(image_array).save(temp_input_png)
            cmd = ['cjxl', temp_input_png, temp_output_jxl, '-d', '0', '-e', '4']
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                with open(temp_output_jxl, 'rb') as f:
                    return f.read()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"cjxl failed: {e.stderr.decode(errors='ignore')}")

    elif codec in ['j2k', 'jls']:
        with tempfile.TemporaryDirectory() as td:
            temp_input = os.path.join(td, 'in.dcm')
            temp_output = os.path.join(td, 'out.dcm')
            # Use clean dataset for compression
            temp_ds = create_clean_dicom_dataset(image_array)
            temp_ds.save_as(temp_input)
            cmd = ['gdcmconv', '--j2k' if codec == 'j2k' else '--jpegls', temp_input, temp_output]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                with open(temp_output, 'rb') as f:
                    return f.read()
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"gdcmconv failed: {e.stderr.decode(errors='ignore')}")

    elif codec == 'png':
        ds = create_clean_dicom_dataset(image_array)
        ds.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
        buffer = io.BytesIO()
        ds.save_as(buffer)
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
            cmd = ['djxl', temp_in, temp_out]
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

# ==============================================================================
# BIT/MESSAGE MANIPULATION
# ==============================================================================

def convert_message_to_bits(message: str) -> np.ndarray:
    """Converts a UTF-8 string message to a numpy array of bits (0s and 1s)."""
    return np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))

def convert_bits_to_message(bits: np.ndarray) -> str:
    """Converts bits back to string message."""
    # Ensure we have complete bytes
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - len(bits) % 8), mode='constant')
    
    bytes_array = np.packbits(bits).tobytes()
    return bytes_array.decode('utf-8', errors='replace').rstrip('\x00')

def create_message_segments(local_planes: list, message_bits: np.ndarray) -> tuple[list, list, list]:
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
    if distributed_sizes:
        distributed_sizes[0] += final_adjustment
    
    all_segments = {}
    cursor = 0
    for i in range(num_local_planes):
        num = distributed_sizes[i]
        all_segments[i] = message_bits[cursor : cursor + num]
        cursor += num
        
    segment_indices = list(range(num_local_planes))
    random.seed(42)
    random.shuffle(segment_indices)
    
    processing_segments = [all_segments[idx] for idx in segment_indices]
    final_lengths = [len(all_segments[i]) for i in range(num_local_planes)]

    return processing_segments, final_lengths, segment_indices

def embed_message_in_planes(local_planes: list, message_bits: np.ndarray, allowed_indices: np.ndarray, image_shape: tuple, start_offset: int = 0, align_across_planes: bool = False):
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
        stego_pixels = (original_pixels & 0xFE) | segment
        stego_plane[y_coords, x_coords] = stego_pixels

        xor_values = original_pixels ^ stego_pixels
        embedding_map[y_coords, x_coords] = 1 + xor_values
        
        total_used += num_bits
        if not align_across_planes:
            current_start_offset_in_allowed += num_bits

    return stego_planes, embedding_map, total_used, segments_lengths, segment_indices

def extract_message_and_restore_planes(stego_planes: list, embedding_map: np.ndarray, metadata: dict) -> tuple[str, list]:
    """Extracts the message and restores the original planes using the embedding map."""
    num_local_planes = metadata['s']
    h, w = metadata['height'], metadata['width']
    align_across_planes = metadata.get('align_flag', 0)
    current_start_offset_in_allowed = metadata.get('start_offset', 0)

    allowed_indices = np.where(embedding_map.ravel() > 0)[0]
    xor_map = (embedding_map == 2).astype(np.uint8)

    total_bits = sum(metadata['segments_lengths'])
    all_bits_array = np.empty(total_bits, dtype=np.uint8)
    restored_planes = [p.copy() for p in stego_planes]
    
    for i, dest_plane_idx in enumerate(metadata['segments_indices']):
        num_bits = metadata['segments_lengths'][dest_plane_idx]
        if num_bits == 0:
            continue

        indices_to_use = allowed_indices[current_start_offset_in_allowed : current_start_offset_in_allowed + num_bits]
        y_coords, x_coords = np.unravel_index(indices_to_use, (h, w))
        
        extracted_bits = stego_planes[dest_plane_idx][y_coords, x_coords] & 1
        
        segment_start = sum(metadata['segments_lengths'][:dest_plane_idx])
        all_bits_array[segment_start : segment_start + num_bits] = extracted_bits
        
        xor_diff = xor_map[y_coords, x_coords]
        original_lsb = extracted_bits ^ xor_diff
        restored_planes[dest_plane_idx][y_coords, x_coords] = (restored_planes[dest_plane_idx][y_coords, x_coords] & 0xFE) | original_lsb
        
        if not align_across_planes:
            current_start_offset_in_allowed += num_bits

    message = convert_bits_to_message(all_bits_array)
    
    return message, restored_planes

# ==============================================================================
# IMAGE ANALYSIS
# ==============================================================================

def calculate_entropy(data_array: np.ndarray) -> float:
    """Calculates the Shannon entropy of an array."""
    counts = np.bincount(data_array.ravel())
    probabilities = counts[counts > 0] / data_array.size
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_mutual_information(bit_plane: np.ndarray, image_array: np.ndarray) -> float:
    """Calculates the mutual information between a bit plane and the image."""
    if bit_plane.min() == bit_plane.max() or image_array.min() == image_array.max():
        return 0.0

    h_x = calculate_entropy(bit_plane)
    h_y = calculate_entropy(image_array)

    max_val = int(image_array.max())
    combined_indices = bit_plane.ravel().astype(np.int32) * (max_val + 1) + image_array.ravel().astype(np.int32)
    h_xy = calculate_entropy(combined_indices)
    
    mi = h_x + h_y - h_xy
    return max(0.0, mi)

def decompose_image_adaptively(image_array: np.ndarray, beta: float = 0.8, nbits: int = None) -> tuple[list, list]:
    """Adaptively decomposes the image into global and local bit planes."""
    nbits = image_array.dtype.itemsize * 8 if nbits is None else nbits
    logger.info(f"\t- Effective bit depth: {nbits}")
    bit_planes = [(image_array >> i) & 1 for i in range(nbits)]
    total_info = calculate_entropy(image_array)
    logger.info(f"\t- Total image information (entropy): {total_info:.4f}")
    target_info = beta * total_info
    logger.info(f"\t- Retention target ({beta*100:.1f}%): {target_info:.4f}")
    
    cumulative_info = 0.0
    s = 0
    for i in range(nbits):
        mi = calculate_mutual_information(bit_planes[i], image_array)
        cumulative_info += mi
        if cumulative_info >= target_info:
            s = i + 1
            break
    
    logger.info(f"\t- Separation point 's' found at {s} (i.e., {s} LSB planes are local)")
    local_planes = bit_planes[:s]
    global_planes = bit_planes[s:]
    return global_planes, local_planes

def create_embedding_capacity_map(image_array: np.ndarray, block_size: int = 8, threshold_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Analyzes the image to find complex regions suitable for embedding."""
    logger.info(f"\t- Creating capacity map with {block_size}x{block_size} blocks (threshold factor: {threshold_factor})...")
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
    logger.info(f"\t- Adaptive threshold calculated: {adaptive_threshold:.4f}")

    capacity_map = np.kron(block_stds > adaptive_threshold, np.ones((block_size, block_size), dtype=np.uint8))[:h, :w]
    allowed_indices = np.where(capacity_map.ravel() == 1)[0]
    
    total_capacity = len(allowed_indices)
    capacity_percent = total_capacity / (h * w) * 100
    logger.info(f"\t- Capacity found: {total_capacity} bits ({capacity_percent:.2f}% of image)")
    
    return capacity_map, allowed_indices

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

# ==============================================================================
# HEADER & CONTAINER MANAGEMENT
# ==============================================================================

def create_steganography_header(codec: str, s: int, segments_lengths: list, segments_indices: list, bitmaps_blob_size: int, 
                  width: int, height: int, start_offset: int, align_across_planes: bool,
                  block_size: int, threshold_factor: float) -> bytes:
    """Creates the header for the steganography container file."""
    codec_map = {'png': 1, 'j2k': 2, 'jls': 3, 'jxl': 4}
    codec_id = codec_map.get(codec.lower(), 0)
    align_flag = 1 if align_across_planes else 0

    header_format = f'>BBBBHHH B f{s}H{s}BI'
    header_parts = [1, codec_id, s, align_flag, width, height, start_offset, block_size, threshold_factor, 
                    *segments_lengths, *segments_indices, bitmaps_blob_size]
    
    packed_header = struct.pack(header_format, *header_parts)
    return struct.pack('>I', len(packed_header)) + packed_header

def parse_steganography_file(filepath: str):
    """Parses the header of a steganography container file."""
    codec_map = {1: 'png', 2: 'j2k', 3: 'jls', 4: 'jxl'}
    with open(filepath, 'rb') as f:
        if f.read(4) != b'STGC':
            raise ValueError("Invalid file: incorrect signature.")
        
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
            'align_flag': bool(align_flag), 'width': width, 'height': height,
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

# ==============================================================================
# MAIN WORKFLOWS WITH STEGANOGRAPHY
# ==============================================================================

def run_steganography(input_dicom_file, output_dir, base_filename, beta, block_size, threshold_factor, codec='jxl', align_across_planes=False, start_offset=0):
    """
    Hides the original DICOM metadata within the image pixels.
    Creates a stego container with clean metadata.
    """
    print('\n')
    logger.info(f"STARTING STEGANOGRAPHY ENCODING\n{'='*100}")
    logger.info(f"Parameters: Beta={beta}, BlockSize={block_size}, Threshold={threshold_factor}, Codec={codec}")
    
    # 1. Read original DICOM and extract metadata
    logger.info("[1/5] Reading source DICOM and extracting metadata...")
    original_dicom = pydicom.dcmread(input_dicom_file)
    image_array = original_dicom.pixel_array
    
    # Extract original metadata as the secret message
    secret_message = extract_dicom_metadata(original_dicom)
    message_bits = convert_message_to_bits(secret_message)
    
    logger.info(f"\t- Original image: {image_array.shape}, {image_array.dtype}")
    logger.info(f"\t- Secret metadata size: {len(message_bits)} bits ({len(secret_message)} chars)")
    logger.info(f"\t- Original modality: {getattr(original_dicom, 'Modality', 'Unknown')}")

    # 2. Decompose Image
    logger.info("[2/5] Decomposing image adaptively...")
    global_planes, local_planes = decompose_image_adaptively(image_array, beta=beta)

    # 3. Create Capacity Map
    logger.info("[3/5] Creating embedding capacity map...")
    capacity_map, allowed_indices = create_embedding_capacity_map(image_array, block_size=block_size, threshold_factor=threshold_factor)
    capacity_map_path = os.path.join(output_dir, f"{base_filename}_capacity_map.png")
    Image.fromarray((capacity_map * 255).astype(np.uint8)).save(capacity_map_path)
    logger.info(f"\t✔ Capacity map saved to: {capacity_map_path}")

    if len(message_bits) > len(allowed_indices):
        logger.warning(f"Metadata ({len(message_bits)} bits) is too large for capacity ({len(allowed_indices)} bits). Aborting.")
        return None, None, None

    # 4. Embed Metadata
    logger.info("[4/5] Embedding DICOM metadata into local planes...")
    stego_planes, embedding_map, _, segments_lengths, segment_indices = embed_message_in_planes(
        local_planes, message_bits, allowed_indices, image_array.shape, 
        start_offset=start_offset, align_across_planes=align_across_planes
    )
    embedding_map_path = os.path.join(output_dir, f"{base_filename}_embedding_map.png")
    Image.fromarray(embedding_map * 127).save(embedding_map_path)
    logger.info(f"\t✔ Embedding map saved to: {embedding_map_path}")

    # 5. Create Stego Container with Clean Metadata
    logger.info("[5/5] Creating final steganography container with clean metadata...")
    stego_image_array = merge_bit_planes(global_planes, stego_planes)
    compressed_bytes = compress_image_data(stego_image_array, codec)
    bitmaps_blob = zlib.compress(embedding_map.tobytes(), level=9)
    
    header = create_steganography_header(
        codec=codec, s=len(local_planes), segments_lengths=segments_lengths,
        segments_indices=segment_indices, bitmaps_blob_size=len(bitmaps_blob),
        width=image_array.shape[1], height=image_array.shape[0], start_offset=start_offset,
        align_across_planes=align_across_planes, block_size=block_size, threshold_factor=threshold_factor
    )
    
    output_bin_file = os.path.join(output_dir, f"{base_filename}.bin")
    file_size = create_steganography_container(output_bin_file, header, bitmaps_blob, compressed_bytes)
    
    # Also save the stego DICOM with clean metadata for reference
    stego_dicom = create_clean_dicom_dataset(stego_image_array)
    stego_dicom_file = os.path.join(output_dir, f"{base_filename}_stego.dcm")
    save_dicom_file(stego_dicom, stego_dicom_file)
    
    logger.info(f"\t✔ Stego container created: {output_bin_file} ({file_size / 1024:.2f} KB)")
    logger.info(f"\t✔ Stego DICOM with clean metadata: {stego_dicom_file}")
    logger.info(f"\n{'='*100}\n\t\t    ENCODING COMPLETE\n")
    return file_size, output_bin_file, stego_dicom_file

def decode_steganography_container(filepath: str, output_prefix: str = "decoded"):
    """
    Decodes a .bin file, extracts the hidden metadata, and reconstructs the original DICOM.
    Returns: original DICOM with restored metadata, extracted metadata message, and restored image
    """
    logger.info(f"STARTING STEGANOGRAPHY DECODING\n{'='*100}")
    logger.info(f"File: {filepath}")

    # 1. Parse Container
    logger.info("[1/5] Parsing steganography container...")
    metadata, embedding_map_blob, stego_image_data = parse_steganography_file(filepath)
    logger.info(f"\t- Codec: {metadata['codec']}, Local Planes: {metadata['s']}")

    # 2. Decompress and Load Data
    logger.info("[2/5] Decompressing image and loading embedding map...")
    stego_array = decompress_image_data(stego_image_data, metadata['codec'])
    embedding_map_bytes = zlib.decompress(embedding_map_blob)
    embedding_map = np.frombuffer(embedding_map_bytes, dtype=np.uint8).reshape((metadata['height'], metadata['width']))

    # 3. Extract Hidden Metadata
    logger.info("[3/5] Extracting hidden DICOM metadata...")
    nbits = stego_array.dtype.itemsize * 8
    all_stego_planes = [(stego_array >> i) & 1 for i in range(nbits)]
    stego_local_planes = all_stego_planes[:metadata['s']]
    global_planes = all_stego_planes[metadata['s']:]
    
    extracted_metadata_json, restored_local_planes = extract_message_and_restore_planes(
        stego_local_planes, embedding_map, metadata
    )

    # Save extracted metadata
    metadata_file = f"{output_prefix}_extracted_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(extracted_metadata_json)
    logger.info(f"\t✔ Extracted metadata saved to: {metadata_file}")

    # 4. Reconstruct Original Image
    logger.info("[4/5] Reconstructing original image...")
    restored_image_array = merge_bit_planes(global_planes, restored_local_planes)
    
    # 5. Create DICOM with Restored Metadata
    logger.info("[5/5] Creating DICOM with restored original metadata...")
    
    # Create a clean DICOM dataset
    restored_dicom = create_clean_dicom_dataset(restored_image_array)
    
    # Restore the original metadata
    restored_dicom = restore_dicom_metadata(restored_dicom, extracted_metadata_json)
    
    # Save the final DICOM
    restored_dicom_file = f"{output_prefix}_restored.dcm"
    save_dicom_file(restored_dicom, restored_dicom_file)
    
    logger.info(f"\t✔ Original DICOM with restored metadata: {restored_dicom_file}")
    logger.info(f"\n{'='*100}\n\t\t    DECODING COMPLETE\n")
    
    return restored_dicom, extracted_metadata_json, restored_image_array

def main():
    """Main function for standalone script usage."""
    try:
        # --- Parameters ---
        input_dicom_file = "images/dx_8b/111.dcm"
        input_dicom_file = "images/mr_16b/666.dcm"
        output_dir = "output"
        beta = 0.4
        block_size = 4
        threshold_factor = 0.8
        codec = 'jxl'
        base_filename = f"meta_stego_beta{beta}_bs{block_size}_tf{threshold_factor}"
        # ---

        os.makedirs(output_dir, exist_ok=True)
        if not os.path.exists(input_dicom_file):
            logger.error(f"Input file not found: {input_dicom_file}")
            return

        # Encode: Hide metadata in image
        file_size, bin_path, stego_dcm_path = run_steganography(
            input_dicom_file, output_dir, base_filename,
            beta, block_size, threshold_factor, codec
        )

        if file_size:
            # Decode: Extract metadata and reconstruct original DICOM
            restored_dicom, extracted_metadata, restored_image = decode_steganography_container(
                bin_path, 
                output_prefix=os.path.join(output_dir, f"{base_filename}_decoded")
            )
            
            # Verify the restoration
            original_dicom = pydicom.dcmread(input_dicom_file)
            logger.info(f"VERIFICATION RESULTS\n{'='*100}")
            logger.info(f"Images match: {np.array_equal(original_dicom.pixel_array, restored_image)}")
            logger.info(f"Original PatientID: {getattr(original_dicom, 'PatientID', 'N/A')}")
            logger.info(f"Restored PatientID: {getattr(restored_dicom, 'PatientID', 'N/A')}")
            print('\n')

    except Exception as e:
        logger.error("An unexpected error occurred during execution.", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(relativeCreated)dms\t [ %(levelname)s ]   %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    main()