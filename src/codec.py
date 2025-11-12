import os
import struct
import zlib
import random
import subprocess
import tempfile
import logging
import json
from io import BytesIO
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, DeflatedExplicitVRLittleEndian
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Utilities: DICOM metadata extraction/restoration (kept behaviorally)
# ---------------------------------------------------------------------
def serialize_dicom_value(value):
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
        if len(value) > 1000:
            return f"<binary_data_{len(value)}_bytes>"
        else:
            return value.hex()
    elif hasattr(value, '__str__'):
        return str(value)
    else:
        return f"<unserializable_{type(value).__name__}>"

def extract_dicom_metadata(dicom_dataset: FileDataset) -> str:
    metadata_dict = {}
    for elem in dicom_dataset:
        if elem.tag == (0x7FE0, 0x0010):
            continue
        if elem.tag.group > 0x0008:
            continue
        if hasattr(elem, 'value') and elem.value is not None:
            try:
                serialized_value = serialize_dicom_value(elem.value)
                if serialized_value is not None:
                    metadata_dict[str(elem.tag)] = {
                        'value': serialized_value,
                        'VR': elem.VR if hasattr(elem, 'VR') else 'UN',
                        'name': elem.name if hasattr(elem, 'name') else 'Unknown'
                    }
            except Exception as e:
                logger.warning(f"Could not serialize tag {elem.tag}: {e}")
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
    try:
        metadata_dict = json.loads(metadata_json)
        if '_critical_tags' in metadata_dict:
            critical_tags = metadata_dict.pop('_critical_tags')
            for tag_name, value in critical_tags.items():
                if value is not None and hasattr(dicom_dataset, tag_name):
                    try:
                        current_value = getattr(dicom_dataset, tag_name)
                        if isinstance(current_value, pydicom.valuerep.PersonName) and isinstance(value, str):
                            setattr(dicom_dataset, tag_name, pydicom.valuerep.PersonName(value))
                        elif isinstance(current_value, pydicom.uid.UID) and isinstance(value, str):
                            setattr(dicom_dataset, tag_name, pydicom.uid.UID(value))
                        else:
                            setattr(dicom_dataset, tag_name, value)
                    except Exception as e:
                        logger.warning(f"Could not restore critical tag {tag_name}: {e}")
        # restore other tags if possible
        restored_count = 0
        for tag_str, tag_info in metadata_dict.items():
            try:
                tag = eval(tag_str)
                if tag in dicom_dataset:
                    if isinstance(tag_info, dict) and 'value' in tag_info:
                        dicom_dataset[tag].value = tag_info['value']
                        restored_count += 1
            except Exception:
                pass
        logger.info(f"\t✔ Restored {restored_count + len(metadata_dict.get('_critical_tags', {}))} DICOM metadata tags")
        return dicom_dataset
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata JSON: {e}")
        raise

def create_clean_dicom_dataset(image_array: np.ndarray) -> FileDataset:
    generated_uid = generate_uid()

    max_val = image_array.max() if image_array.size else 0
    bits_stored = int(np.ceil(np.log2(float(max_val) + 1.0))) if max_val > 0 else 1
    bits_stored = max(1, bits_stored)
    SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SOP_CLASS_UID
    file_meta.MediaStorageSOPInstanceUID = generated_uid
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "ANONYMIZED^PATIENT"
    ds.PatientID = "000000"
    ds.StudyInstanceUID = generated_uid
    ds.SeriesInstanceUID = generated_uid
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

def save_dicom_file(dicom_dataset: FileDataset, file_path: str):
    dicom_dataset.save_as(file_path, enforce_file_format=True)
    logger.info(f"\t✔ DICOM file saved to: {file_path}")

# ---------------------------------------------------------------------
# Image compression / decompression (kept from original)
# ---------------------------------------------------------------------
def compress_image_data(image_array: np.ndarray, codec: str) -> bytes:
    """Compresses image data using the specified codec (optimized, PNM via stdin)."""
    logger.info(f"\t- Compressing with {codec.upper()}...")
    if codec != 'jxl':
        raise ValueError(f"Unsupported codec: '{codec}' in this function.")

    try:
        # === Build PNM header ===
        height, width = image_array.shape[:2]
        # Clamp maxval to PNM allowed range
        if image_array.dtype == np.uint8:
            img_bytes = image_array.tobytes()
        else:
            tmp = image_array.astype(np.uint16)
            img_bytes = tmp.astype('>u2').tobytes()  # big-endian


        header = f"P5\n{width} {height}\n{image_array.max()}\n".encode('ascii')

        pnm_data = header + img_bytes

        # === cjxl command: read stdin '-' and write stdout '-' ===
        cmd = [
            'cjxl',
            '-',   # input from stdin (PNM)
            '-',   # output JXL to stdout
            '--distance=0',   # lossless
            '--effort=7'
        ]

        # Run and capture stdout/stderr; raise on non-zero exit
        result = subprocess.run(cmd, input=pnm_data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)

        if result.returncode != 0:
            stderr_text = result.stderr.decode(errors='ignore')
            logger.error("=== CJXL STDERR ===\n" + stderr_text + "\n===================")
            raise RuntimeError(f"cjxl failed with exit code {result.returncode}: {stderr_text[:400]}")

        # result.stdout contém os bytes .jxl
        return result.stdout

    except Exception as e:
        raise RuntimeError(f"cjxl failed: {e}")

def decompress_image_data(compressed_bytes, codec):
    if codec == 'jxl':
        from imageio.v3 import imread

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jxl') as temp_input:
            temp_input.write(compressed_bytes)
            temp_input_path = temp_input.name

        # Tenta decodificar como PNG (mantém profundidade e ordem de bytes)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_output:
            temp_output_path = temp_output.name

        # --bits_per_sample=16 garante que o djxl mantenha a profundidade se for 16 bits
        cmd = [
            'djxl',
            temp_input_path,
            temp_output_path,
            '--quiet',
            '--bits_per_sample=16'
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"djxl failed ({result.returncode}):\n{result.stderr.decode(errors='ignore')}"
            )

        # Carrega a imagem preservando o dtype original (8 ou 16 bits)
        image = imread(temp_output_path)
        image = np.array(image, copy=False)

        # Limpa temporários
        os.remove(temp_input_path)
        os.remove(temp_output_path)

        return image

    elif codec in ['j2k', 'jls', 'png']:
        dicom_dataset = pydicom.dcmread(BytesIO(compressed_bytes), force=True)
        return dicom_dataset.pixel_array
    else:
        raise ValueError(f"Unsupported codec: '{codec}'")

# ---------------------------------------------------------------------
# Bit/message helpers
# ---------------------------------------------------------------------
def convert_message_to_bits(message: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))

def convert_bits_to_message(bits: np.ndarray) -> str:
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - len(bits) % 8), mode='constant')
    bytes_array = np.packbits(bits).tobytes()
    return bytes_array.decode('utf-8', errors='replace').rstrip('\x00')

# ---------------------------------------------------------------------
# Image analysis functions (entropy, mutual information, decomposition)
# ---------------------------------------------------------------------
def calculate_entropy(data_array: np.ndarray) -> float:
    counts = np.bincount(data_array.ravel())
    probabilities = counts[counts > 0] / data_array.size
    return -np.sum(probabilities * np.log2(probabilities))

def calculate_mutual_information(bit_plane: np.ndarray, image_array: np.ndarray) -> float:
    if bit_plane.min() == bit_plane.max() or image_array.min() == image_array.max():
        return 0.0
    h_x = calculate_entropy(bit_plane)
    h_y = calculate_entropy(image_array)
    combined = bit_plane.astype(np.uint8) << 1 | (image_array > image_array.mean()).astype(np.uint8)
    h_xy = calculate_entropy(combined)
    mi = h_x + h_y - h_xy
    return max(0.0, mi)

def decompose_image_adaptively(image_array: np.ndarray, beta: float = 0.8, nbits: int = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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

# ---------------------------------------------------------------------
# NEW: Dynamic capacity map (stop when enough pixels selected)
# ---------------------------------------------------------------------
def create_capacity_map_dynamic(image_array: np.ndarray, required_bits: int, block_size: int = 8, threshold_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"\t- Creating dynamic capacity map (block {block_size}x{block_size}, threshold {threshold_factor})...")
    h, w = image_array.shape
    # pad to full blocks
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded = np.pad(image_array, ((0, h_pad), (0, w_pad)), 'edge')
    ph, pw = padded.shape
    bh = ph // block_size
    bw = pw // block_size

    # compute block stds
    blocks = padded.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)
    block_stds = np.std(blocks, axis=(2, 3))
    flat_stds = block_stds.ravel()
    order = np.argsort(flat_stds)[::-1]  # desc

    pixels_per_block = block_size * block_size
    selected = np.zeros(bh * bw, dtype=np.bool_)
    cum_pixels = 0
    for idx in order:
        # use threshold_factor by boosting blocks with std > mean*threshold_factor
        # but we still pick in order until we have enough
        selected[idx] = True
        cum_pixels += pixels_per_block
        if cum_pixels >= required_bits:
            break

    selected_map_blocks = selected.reshape(bh, bw)
    capacity_map_padded = np.kron(selected_map_blocks, np.ones((block_size, block_size), dtype=np.uint8))
    capacity_map = capacity_map_padded[:h, :w]
    allowed_indices = np.where(capacity_map.ravel() == 1)[0]

    logger.info(f"\t- Dynamic capacity: selected_blocks={selected.sum()}, allowed_pixels={len(allowed_indices)} (required={required_bits})")
    return capacity_map, allowed_indices

# ---------------------------------------------------------------------
# Message segmentation, embed/extract (kept similar to original)
# ---------------------------------------------------------------------
def create_message_segments(local_planes: List[np.ndarray], message_bits: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[int]]:
    num_local_planes = len(local_planes)
    total_bits = len(message_bits)
    weights = [(num_local_planes - i) ** 2 for i in range(num_local_planes)]
    total_weight = sum(weights) if weights else 0
    if total_weight == 0 and num_local_planes > 0:
        distributed_sizes = [total_bits // num_local_planes] * num_local_planes
    elif num_local_planes > 0:
        distributed_sizes = [max(1, int((w / total_weight) * total_bits)) for w in weights]
    else:
        return [], [], []
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
        all_segments[i] = message_bits[cursor:cursor + num]
        cursor += num
    segment_indices = list(range(num_local_planes))
    random.seed(42)
    random.shuffle(segment_indices)
    processing_segments = [all_segments[idx] for idx in segment_indices]
    final_lengths = [len(all_segments[i]) for i in range(num_local_planes)]
    return processing_segments, final_lengths, segment_indices

def embed_message_in_planes(local_planes: List[np.ndarray], message_bits: np.ndarray, allowed_indices: np.ndarray, image_shape: Tuple[int,int], start_offset: int = 0, align_across_planes: bool = False):
    h, w = image_shape
    num_message_bits = len(message_bits)
    if num_message_bits > len(allowed_indices):
        raise ValueError(f"Message ({num_message_bits} bits) is larger than image capacity ({len(allowed_indices)} bits).")
    indices_to_use_flat = allowed_indices[start_offset:start_offset + num_message_bits]
    segments, segments_lengths, segment_indices = create_message_segments(local_planes, message_bits)
    stego_planes = [p.copy() for p in local_planes]
    flip_bits = np.empty(num_message_bits, dtype=np.uint8)
    current_offset = 0
    for i, dest_idx in enumerate(segment_indices):
        segment = segments[i]
        stego_plane = stego_planes[dest_idx]
        num_bits_in_segment = len(segment)
        if num_bits_in_segment == 0:
            continue
        seg_indices_flat = indices_to_use_flat[current_offset: current_offset + num_bits_in_segment]
        y_coords, x_coords = divmod(seg_indices_flat, w)
        original_pixels = stego_plane[y_coords, x_coords]
        stego_pixels = (original_pixels & 0xFE) | segment
        stego_plane[y_coords, x_coords] = stego_pixels
        xor_values = (original_pixels ^ stego_pixels) & 1
        flip_bits[current_offset: current_offset + num_bits_in_segment] = xor_values
        current_offset += num_bits_in_segment
    return stego_planes, segments_lengths, segment_indices, indices_to_use_flat, flip_bits

def extract_message_and_restore_planes(stego_planes: List[np.ndarray], used_indices: np.ndarray, flip_bits: np.ndarray, metadata: dict) -> Tuple[str, List[np.ndarray]]:
    h, w = metadata['height'], metadata['width']
    segments_lengths = metadata['segments_lengths']
    segments_indices = metadata['segments_indices']
    total_bits = len(used_indices)
    all_bits_array = np.empty(total_bits, dtype=np.uint8)
    restored_planes = [p.copy() for p in stego_planes]
    current_offset_in_used = 0
    for i, dest_plane_idx in enumerate(segments_indices):
        num_bits = segments_lengths[dest_plane_idx]
        if num_bits == 0:
            continue
        indices_for_segment = used_indices[current_offset_in_used: current_offset_in_used + num_bits]
        flip_bits_for_segment = flip_bits[current_offset_in_used: current_offset_in_used + num_bits]
        y_coords, x_coords = np.unravel_index(indices_for_segment, (h, w))
        extracted_bits = stego_planes[dest_plane_idx][y_coords, x_coords] & 1
        segment_start = sum(segments_lengths[:dest_plane_idx])
        all_bits_array[segment_start: segment_start + num_bits] = extracted_bits
        original_lsb = extracted_bits ^ flip_bits_for_segment
        restored_planes[dest_plane_idx][y_coords, x_coords] = (restored_planes[dest_plane_idx][y_coords, x_coords] & 0xFE) | original_lsb
        current_offset_in_used += num_bits
    message = convert_bits_to_message(all_bits_array)
    return message, restored_planes

# ---------------------------------------------------------------------
# Merge bit planes into image
# ---------------------------------------------------------------------
def merge_bit_planes(global_planes: List[np.ndarray], local_planes: List[np.ndarray]) -> np.ndarray:
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

# ---------------------------------------------------------------------
# NEW: optimized bitmap_blob creation/parsing (diffs + packbits) - blob at EOF
# ---------------------------------------------------------------------
def create_optimized_bitmap_blob(used_indices: np.ndarray, flip_bits: np.ndarray) -> bytes:
    if used_indices.size == 0:
        raise ValueError("used_indices está vazio.")
    # Ensure indices are int64 and monotonic increasing
    used_indices = np.asarray(used_indices, dtype=np.int64)
    # store diffs from 0 as uint32 (differences expected small)
    diffs = np.diff(np.insert(used_indices.astype(np.uint32), 0, 0)).astype(np.uint32)
    diffs_bytes = diffs.tobytes()
    flips_packed = np.packbits(flip_bits, bitorder='little').tobytes()
    header = struct.pack("<I", len(diffs))
    raw = header + diffs_bytes + flips_packed
    return zlib.compress(raw, level=1)

def parse_bitmap_blob(blob: bytes, total_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    raw = zlib.decompress(blob)
    count = struct.unpack("<I", raw[:4])[0]
    diffs_end = 4 + count * 4
    diffs = np.frombuffer(raw[4:diffs_end], dtype=np.uint32)
    used_indices = np.cumsum(diffs).astype(np.int64)
    flips_packed = np.frombuffer(raw[diffs_end:], dtype=np.uint8)
    flip_bits = np.unpackbits(flips_packed, bitorder='little')[:total_bits].astype(np.uint8)
    return used_indices, flip_bits

# ---------------------------------------------------------------------
# Header & container functions (new format: header contains stego_image_size)
# ---------------------------------------------------------------------
# File layout:
# [4B 'STGC'][4B header_len][header_bytes][stego_image_bytes][bitmap_blob (EOF)]
def create_steganography_header_bytes(codec: str, s: int, segments_lengths: List[int], segments_indices: List[int],
                                     stego_image_size: int, width: int, height: int, start_offset: int,
                                     align_across_planes: bool, block_size: int, threshold_factor: float, version: int = 1) -> bytes:
    codec_map = {'png': 1, 'j2k': 2, 'jls': 3, 'jxl': 4}
    codec_id = codec_map.get(codec.lower(), 0)
    align_flag = 1 if align_across_planes else 0
    # base format: version(uint8), codec(uint8), s(uint8), align(uint8),
    # width(uint16), height(uint16), stego_image_size(uint32), start_offset(uint16),
    # block_size(uint16), threshold_factor(float), segments_count(uint16)
    base_format = "<BBBBHHI H H f H"
    packed = struct.pack(base_format,
                         version & 0xFF, codec_id & 0xFF, s & 0xFF, align_flag & 0xFF,
                         width & 0xFFFF, height & 0xFFFF,
                         stego_image_size & 0xFFFFFFFF,
                         start_offset & 0xFFFF,
                         block_size & 0xFFFF,
                         float(threshold_factor),
                         len(segments_lengths) & 0xFFFF)
    seg_lens_bytes = b''.join(struct.pack("<I", int(x)) for x in segments_lengths)
    seg_idx_bytes = bytes([int(x) & 0xFF for x in segments_indices])
    return packed + seg_lens_bytes + seg_idx_bytes

def parse_steganography_file(filepath: str):
    codec_map = {1: 'png', 2: 'j2k', 3: 'jls', 4: 'jxl'}
    with open(filepath, 'rb') as f:
        sig = f.read(4)
        if sig != b"STGC":
            raise ValueError("Invalid file: incorrect signature.")
        header_length = struct.unpack("<I", f.read(4))[0]
        header_data = f.read(header_length)
        # parse base header
        base_format = "<BBBBHHI H H f H"
        base_size = struct.calcsize(base_format)
        if len(header_data) < base_size:
            raise ValueError("Header too small/corrupted.")
        (version, codec_id, s, align_flag, width, height, stego_image_size,
start_offset, block_size, threshold_factor, segments_count) = struct.unpack(base_format, header_data[:base_size])
        cursor = base_size
        segments_lengths = []
        for _ in range(segments_count):
            segments_lengths.append(struct.unpack("<I", header_data[cursor:cursor+4])[0])
            cursor += 4
        segments_indices = list(header_data[cursor:cursor+segments_count])
        cursor += segments_count
        # Now read stego image bytes (stego_image_size)
        stego_image_bytes = f.read(stego_image_size)
        # The rest of file is the bitmap blob
        bitmaps_blob = f.read()
        metadata = {
            'version': version, 'codec': codec_map.get(codec_id, 'unknown'), 's': s,
            'align_flag': bool(align_flag), 'width': width, 'height': height,
            'start_offset': start_offset, 'segments_lengths': segments_lengths,
            'segments_indices': segments_indices, 'block_size': block_size,
            'threshold_factor': threshold_factor
        }
    return metadata, bitmaps_blob, stego_image_bytes

def create_steganography_container(filename: str, header_bytes: bytes, bitmap_bytes: bytes, stego_bytes: bytes) -> int:
    with open(filename, "wb") as f:
        f.write(b"STGC")
        f.write(struct.pack("<I", len(header_bytes)))
        f.write(header_bytes)
        f.write(stego_bytes)
        f.write(bitmap_bytes)
    return os.path.getsize(filename)

# ---------------------------------------------------------------------
# Main workflows with steganography (kept log step messages)
# ---------------------------------------------------------------------
def run_steganography(input_dicom_file, output_dir, base_filename, beta, block_size, threshold_factor, codec='jxl', align_across_planes=False, start_offset=0):
    print('\n')
    logger.info(f"STARTING STEGANOGRAPHY ENCODING\n{'='*100}")
    logger.info(f"Parameters: Beta={beta}, BlockSize={block_size}, Threshold={threshold_factor}, Codec={codec}")

    # 1. Read original DICOM and extract metadata
    logger.info("[1/5] Reading source DICOM and extracting metadata...")
    original_dicom = pydicom.dcmread(input_dicom_file)
    image_array = original_dicom.pixel_array

    secret_message = extract_dicom_metadata(original_dicom)
    message_bits = convert_message_to_bits(secret_message)

    logger.info(f"\t- Original image: {image_array.shape}, {image_array.dtype}")
    logger.info(f"\t- Secret metadata size: {len(message_bits)} bits ({len(secret_message)} chars)")
    logger.info(f"\t- Original modality: {getattr(original_dicom, 'Modality', 'Unknown')}")

    # 2. Decompose Image
    logger.info("[2/5] Decomposing image adaptively...")
    global_planes, local_planes = decompose_image_adaptively(image_array, beta=beta)

    # 3. Create dynamic Capacity Map
    logger.info("[3/5] Creating embedding capacity map (dynamic)...")
    capacity_map, allowed_indices = create_capacity_map_dynamic(image_array, required_bits=len(message_bits), block_size=block_size, threshold_factor=threshold_factor)

    if len(message_bits) > len(allowed_indices):
        logger.warning(f"Metadata ({len(message_bits)} bits) is too large for capacity ({len(allowed_indices)} bits). Aborting.")
        return None, None, None

    # 4. Embed Metadata
    logger.info("[4/5] Embedding DICOM metadata into local planes...")
    stego_planes, segments_lengths, segment_indices, used_indices, flip_bits = embed_message_in_planes(
        local_planes, message_bits, allowed_indices, image_array.shape,
        start_offset=start_offset, align_across_planes=align_across_planes
    )

    # create embedding_map_img for optional debugging (not saved in container)
    embedding_map_img = np.zeros(image_array.shape, dtype=np.uint8)
    y_coords, x_coords = np.unravel_index(used_indices, image_array.shape)
    embedding_map_img[y_coords, x_coords] = 1 + flip_bits  # 1 or 2

    # 5. Create Stego Container with Clean Metadata
    logger.info("[5/5] Creating final steganography container with clean metadata...")
    stego_image_array = merge_bit_planes(global_planes, stego_planes)
    compressed_bytes = compress_image_data(stego_image_array, codec)

    # Create optimized bitmap_blob (diffs + packbits) — blob will be appended at EOF
    bitmaps_blob = create_optimized_bitmap_blob(used_indices, flip_bits)

    # create header bytes containing stego image size (so parser knows where stego ends)
    header_bytes = create_steganography_header_bytes(
        codec=codec, s=len(local_planes), segments_lengths=segments_lengths,
        segments_indices=segment_indices, stego_image_size=len(compressed_bytes),
        width=image_array.shape[1], height=image_array.shape[0], start_offset=start_offset,
        align_across_planes=align_across_planes, block_size=block_size, threshold_factor=threshold_factor
    )

    output_bin_file = os.path.join(output_dir, f"{base_filename}.bin")
    os.makedirs(output_dir, exist_ok=True)
    file_size = create_steganography_container(output_bin_file, header_bytes, bitmaps_blob, compressed_bytes)

    # Also save the stego DICOM with clean metadata for reference
    stego_dicom = create_clean_dicom_dataset(stego_image_array)
    stego_dicom_file = os.path.join(output_dir, f"{base_filename}_stego.dcm")
    save_dicom_file(stego_dicom, stego_dicom_file)

    logger.info(f"\t✔ Stego container created: {output_bin_file} ({file_size / 1024:.2f} KB)")
    logger.info(f"\t✔ Stego DICOM with clean metadata: {stego_dicom_file}")
    logger.info(f"\n{'='*100}\n\t\t    ENCODING COMPLETE\n")
    return file_size, output_bin_file, stego_dicom_file

def decode_steganography_container(filepath: str, output_prefix: str = "decoded"):
    logger.info(f"STARTING STEGANOGRAPHY DECODING\n{'='*100}")
    logger.info(f"File: {filepath}")

    # 1. Parse Container (reads header, stego bytes and blob at EOF)
    logger.info("[1/5] Parsing steganography container...")
    metadata, bitmaps_blob, stego_image_bytes = parse_steganography_file(filepath)
    logger.info(f"\t- Codec: {metadata['codec']}, Local Planes: {metadata['s']}")

    # 2. Decompress Image Data
    logger.info("[2/5] Decompressing image data...")
    stego_array = decompress_image_data(stego_image_bytes, metadata['codec'])

    # 3. Extract Hidden Metadata
    logger.info("[3/5] Extracting hidden DICOM metadata...")
    nbits = stego_array.dtype.itemsize * 8
    all_stego_planes = [(stego_array >> i) & 1 for i in range(nbits)]
    stego_local_planes = all_stego_planes[:metadata['s']]
    global_planes = all_stego_planes[metadata['s']:]

    total_bits = sum(metadata['segments_lengths'])
    used_indices, flip_bits = parse_bitmap_blob(bitmaps_blob, total_bits)

    extracted_metadata_json, restored_local_planes = extract_message_and_restore_planes(
        stego_local_planes, used_indices, flip_bits, {
            'height': metadata['height'], 'width': metadata['width'],
            'segments_lengths': metadata['segments_lengths'], 'segments_indices': metadata['segments_indices']
        }
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
    restored_dicom = create_clean_dicom_dataset(restored_image_array)
    restored_dicom = restore_dicom_metadata(restored_dicom, extracted_metadata_json)
    restored_dicom_file = f"{output_prefix}_restored.dcm"
    save_dicom_file(restored_dicom, restored_dicom_file)

    logger.info(f"\t✔ Original DICOM with restored metadata: {restored_dicom_file}")
    logger.info(f"\n{'='*100}\n\t\t    DECODING COMPLETE\n")
    return restored_dicom, extracted_metadata_json, restored_image_array

# ---------------------------------------------------------------------
# main() preserved with original example usage (keeps both image examples)
# ---------------------------------------------------------------------
def main():
    try:
        # --- Parameters (kept original examples) ---
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
