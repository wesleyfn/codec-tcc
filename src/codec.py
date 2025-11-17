# codec.py (versão otimizada: c-contiguous bitpacking, streaming reconstruction,
# precomputed histograma na decomposição, vetorized capacity map)
import os
import struct
import zlib
import random
import logging
import json
import imagecodecs
from datetime import datetime
from typing import List, Tuple
from scipy.stats import entropy

from PIL import Image
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

logger = logging.getLogger(__name__)

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

def compress_image_data(image_array: np.ndarray, codec: str) -> bytes:
    logger.info(f"\t- Compressing image with codec: {codec.upper()}...")
    
    match codec:
        case 'jxl':
            bitspersample = 16 if image_array.dtype == np.uint16 else 8
            return imagecodecs.jpegxl_encode(
                image_array,
                lossless=True, effort=7, decodingspeed=0,
                photometric='GRAY', bitspersample=bitspersample,
                planar=False, numthreads=0
            )
        case 'j2k':
            bitspersample = 16 if image_array.dtype == np.uint16 else 8
            return imagecodecs.jpeg2k_encode(
                image_array,
                reversible=True, colorspace='GRAY', planar=False, 
                bitspersample=bitspersample, mct=False, numthreads=0
            )
        case 'jls':
            return imagecodecs.jpegls_encode(image_array)
        case 'png':
            return imagecodecs.png_encode(image_array)
        case _:
            raise ValueError(f"Unsupported codec: {codec}")

def decompress_image_data(compressed_bytes, codec):
    match codec:
        case 'jxl':
            return imagecodecs.jpegxl_decode(compressed_bytes, numthreads=0, keeporientation=True)
        case 'j2k':
            return imagecodecs.jpeg2k_decode(compressed_bytes, numthreads=0)
        case 'jls':
            return imagecodecs.jpegls_decode(compressed_bytes)
        case 'png':
            return imagecodecs.png_decode(compressed_bytes)
        case _:
            raise ValueError(f"Unsupported codec: {codec}")

def convert_message_to_bits(message: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(message.encode('utf-8'), dtype=np.uint8))

def convert_bits_to_message(bits: np.ndarray) -> str:
    if len(bits) % 8 != 0:
        bits = np.pad(bits, (0, 8 - len(bits) % 8), mode='constant')
    bytes_array = np.packbits(bits).tobytes()
    return bytes_array.decode('utf-8', errors='replace').rstrip('\x00')

# ------------------------------
# Bitplane functions (LSB -> MSB)
# ------------------------------

def extract_bit_planes(image: np.ndarray) -> List[np.ndarray]:
    """
    Extrai bit-planes em ordem LSB -> MSB.
    Retorna lista de planos uint8 (0/1), cada um com mesma shape da imagem.
    """
    img = np.ascontiguousarray(image)
    h, w = img.shape
    flat = img.reshape(-1)

    bits_per_pixel = img.dtype.itemsize * 8

    bit_planes = []
    for bit in range(bits_per_pixel):  # LSB → MSB
        plane_flat = ((flat >> bit) & 1).astype(np.uint8)
        plane = plane_flat.reshape(h, w)
        bit_planes.append(np.ascontiguousarray(plane))

    return bit_planes

def reconstruct_from_bit_planes(bit_planes: List[np.ndarray], original_dtype: type) -> np.ndarray:
    """
    Reconstrói imagem assumindo que bit_planes está em ordem LSB -> MSB.
    Implementação streaming para evitar alocação massiva (sem stacked gigante).
    """
    bits_per_pixel = np.dtype(original_dtype).itemsize * 8
    if len(bit_planes) != bits_per_pixel:
        if len(bit_planes) < bits_per_pixel:
            zero_plane = np.zeros_like(bit_planes[0], dtype=np.uint8)
            bit_planes = bit_planes + [zero_plane] * (bits_per_pixel - len(bit_planes))
        else:
            bit_planes = bit_planes[:bits_per_pixel]

    h, w = bit_planes[0].shape
    flat_out = np.zeros(h * w, dtype=original_dtype)

    # Aplicar OR por planos (somente 16 iterações) — operações vetorizadas em C
    for b, plane in enumerate(bit_planes):  # LSB -> MSB
        pflat = np.ascontiguousarray(plane.reshape(-1)).astype(original_dtype)
        if b == 0:
            flat_out |= (pflat << np.uint32(b))
        else:
            flat_out |= (pflat << np.uint32(b))

    return flat_out.reshape(h, w)

def calculate_mutual_information(bit_plane: np.ndarray, original_image: np.ndarray, hist_y: np.ndarray = None, bins_y: int = None) -> float:
    """
    MI rápida entre um bit-plane (0/1) e a imagem original.
    Aceita histograma pré-calculado (hist_y, bins_y) para evitar recalcular P(Y) repetidamente.
    Usa arrays C-contiguous e quantização para 16-bit (4096 bins).
    Retorna valor >= 0.
    """
    # Flatten contíguo
    X = np.ascontiguousarray(bit_plane.reshape(-1).astype(np.uint8))   # 0/1
    Y = np.ascontiguousarray(original_image.reshape(-1))

    total = Y.size
    bits = original_image.dtype.itemsize * 8

    # Quantização: 8-bit -> 256 bins; 16-bit -> 4096 bins (reduz bins sem perder lógica do paper)
    if bins_y is None:
        if bits == 8:
            bins = 256
            Yq = Y.astype(np.int64)
        else:
            bins = 4096
            Yq = (Y.astype(np.uint32) >> 4).astype(np.int64)
    else:
        bins = bins_y
        if bits == 8:
            Yq = Y.astype(np.int64)
        else:
            Yq = (Y.astype(np.uint32) >> 4).astype(np.int64)

    # P(X)
    ones = int(X.sum())
    p_x0 = 1.0 - ones / total
    p_x1 = ones / total

    # P(Y) usando histograma pré-computado se disponível
    if hist_y is not None and len(hist_y) == bins:
        hist_y_arr = hist_y.astype(np.float64)
    else:
        hist_y_arr = np.bincount(Yq, minlength=bins).astype(np.float64)

    p_y = hist_y_arr / total

    # histograma conjunto via duas bincounts (X==1 e X==0)
    mask1 = (X == 1)
    if mask1.any():
        hist_y1 = np.bincount(Yq[mask1], minlength=bins).astype(np.float64)
    else:
        hist_y1 = np.zeros(bins, dtype=np.float64)

    hist_y0 = hist_y_arr - hist_y1

    # Probabilidades conjuntas
    p_joint = np.vstack([hist_y0, hist_y1]) / total

    # Função de entropia robusta
    def entropy(p):
        p_nonzero = p[p > 0.0]
        return -np.sum(p_nonzero * np.log2(p_nonzero))

    H_x = entropy(np.array([p_x0, p_x1], dtype=np.float64))
    H_y = entropy(p_y)
    H_xy = entropy(p_joint.ravel())

    mi = H_x + H_y - H_xy
    return float(mi) if mi > 0.0 else 0.0

def adaptive_modalities_decomposition(image: np.ndarray, beta: float = 0.8):
    """
    Decomposição adaptativa (LSB -> MSB), otimizada para evitar recomputação do histograma da imagem.
    Retorna (global_planes, local_planes, bits_per_pixel).
    """
    # Se a imagem for muito grande, opcionalmente quantizar aqui para reduzir planos (mantido por padrão)
    all_planes = extract_bit_planes(image)  # LSB -> MSB
    bits_per_pixel = len(all_planes)

    # Entropia total via quantização (calcular histograma da imagem apenas 1 vez)
    if image.dtype.itemsize * 8 == 8:
        bins = 256
        img_q = image.reshape(-1).astype(np.int64)
    else:
        bins = 4096
        img_q = (image.reshape(-1).astype(np.uint32) >> 4).astype(np.int64)

    hist = np.bincount(img_q, minlength=bins).astype(np.float64)
    p = hist / hist.sum()
    p = p[p > 0]
    total_entropy = -np.sum(p * np.log2(p))

    # Calcular MI por plano, reutilizando histograma de Y
    mutual_info_values = [calculate_mutual_information(p_plane, image, hist_y=hist, bins_y=bins) for p_plane in all_planes]

    # Encontrar s* (somando MI em ordem LSB -> MSB)
    cumulative = 0.0
    target = beta * total_entropy
    s_star = bits_per_pixel

    for i, mi in enumerate(mutual_info_values):
        cumulative += mi
        if cumulative >= target:
            s_star = i + 1
            break

    s_star = max(1, min(s_star, bits_per_pixel - 1))

    # local = primeiros s* planos (LSB -> ...), global = restantes
    local_planes = all_planes[:s_star]
    global_planes = all_planes[s_star:]

    logger.info(f"\t- Total image entropy H(x): {total_entropy:.4f} bits")
    logger.info(f"\t- Target mutual information (β={beta}): {target:.4f} bits")
    logger.info(f"\t- Adaptive decomposition s*={s_star}, local={len(local_planes)}, global={len(global_planes)}")
    logger.info(f"\t- Achieved mutual info: {cumulative:.4f} / {target:.4f}")

    return global_planes, local_planes, bits_per_pixel

def merge_global_local_planes(global_planes, local_planes, original_dtype):
    """
    Mescla na ordem LSB -> MSB e reconstrói usando streaming (sem grande alocação temporária).
    """
    all_planes = local_planes + global_planes
    return reconstruct_from_bit_planes(all_planes, original_dtype)

# -----------------------------------------
# Capacity map otimizada (vetorizada, var)
# -----------------------------------------
def create_capacity_map_dynamic(image_array: np.ndarray, required_bits: int, block_size: int = 8, threshold_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    logger.info(f"\t- Creating dynamic capacity map (block {block_size}x{block_size}, threshold {threshold_factor})...")
    h, w = image_array.shape
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded = np.pad(image_array, ((0, h_pad), (0, w_pad)), 'edge')
    ph, pw = padded.shape
    bh = ph // block_size
    bw = pw // block_size

    # Vetorizar criação de blocos: shape -> (bh, bw, block_size, block_size)
    blocks = padded.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)
    # usar var ao invés de std para evitar custo sqrt e ainda ordenar textura
    block_vars = np.var(blocks, axis=(2, 3))
    flat_vars = block_vars.ravel()
    order = np.argsort(flat_vars)[::-1]

    pixels_per_block = block_size * block_size
    selected = np.zeros(bh * bw, dtype=np.bool_)
    cum_pixels = 0
    for idx in order:
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

# -------------------------------
# Steganography embedding tooling
# -------------------------------
def create_message_segments(local_planes: List[np.ndarray], message_bits: np.ndarray) -> Tuple[List[np.ndarray], List[int], List[int]]:
    num_local_planes = len(local_planes)
    total_bits = len(message_bits)
    
    # Distribui bits igualmente entre os planos locais
    base_bits_per_plane = total_bits // num_local_planes
    extra_bits = total_bits % num_local_planes
    
    distributed_sizes = [base_bits_per_plane] * num_local_planes
    for i in range(extra_bits):
        distributed_sizes[i] += 1
    
    all_segments = {}
    cursor = 0
    for i in range(num_local_planes):
        num = distributed_sizes[i]
        all_segments[i] = message_bits[cursor:cursor + num]
        cursor += num
    
    # Mantém ordem original para preservar estrutura
    segment_indices = list(range(num_local_planes))
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
        
        # Extrai os bits originais
        original_pixels = stego_plane[y_coords, x_coords]
        
        # Substitui apenas o LSB (bit 0) - método mais seguro
        stego_pixels = (original_pixels & 0xFE) | segment
        
        stego_plane[y_coords, x_coords] = stego_pixels
        
        # Calcula quais bits foram alterados
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
        
        # Extrai bits da imagem stego
        extracted_bits = stego_planes[dest_plane_idx][y_coords, x_coords] & 1
        
        # Calcula posição correta no array final
        segment_start = sum(segments_lengths[:dest_plane_idx])
        all_bits_array[segment_start: segment_start + num_bits] = extracted_bits
        
        # Restaura bits originais
        original_lsb = extracted_bits ^ flip_bits_for_segment
        restored_planes[dest_plane_idx][y_coords, x_coords] = (restored_planes[dest_plane_idx][y_coords, x_coords] & 0xFE) | original_lsb
        
        current_offset_in_used += num_bits
    
    message = convert_bits_to_message(all_bits_array)
    return message, restored_planes

def create_optimized_bitmap_blob(used_indices: np.ndarray, flip_bits: np.ndarray) -> bytes:
    if used_indices.size == 0:
        raise ValueError("used_indices está vazio.")
    
    used_indices = np.asarray(used_indices, dtype=np.int64)
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

def create_steganography_header_bytes(codec: str, s: int, segments_lengths: List[int], segments_indices: List[int],
                                     stego_image_size: int, width: int, height: int, start_offset: int,
                                     align_across_planes: bool, block_size: int, threshold_factor: float, bits_per_pixel: int, version: int = 1) -> bytes:
    codec_map = {'png': 1, 'j2k': 2, 'jls': 3, 'jxl': 4}
    codec_id = codec_map.get(codec.lower(), 0)
    align_flag = 1 if align_across_planes else 0
    
    base_format = "<BBBBHH I H H f H B"
    packed = struct.pack(base_format,
                         version, codec_id, s, align_flag,
                         width, height,
                         stego_image_size,
                         start_offset,
                         block_size,
                         threshold_factor,
                         len(segments_lengths),
                         bits_per_pixel)
    
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
        
        base_format = "<BBBBHHI H H f H B"
        base_size = struct.calcsize(base_format)
        
        if len(header_data) < base_size:
            raise ValueError("Header too small/corrupted.")
        
        (version, codec_id, s, align_flag, width, height, stego_image_size,
         start_offset, block_size, threshold_factor, segments_count, bits_per_pixel) = struct.unpack(base_format, header_data[:base_size])
        
        cursor = base_size
        segments_lengths = []
        for _ in range(segments_count):
            segments_lengths.append(struct.unpack("<I", header_data[cursor:cursor+4])[0])
            cursor += 4
        
        segments_indices = list(header_data[cursor:cursor+segments_count])
        cursor += segments_count
        
        stego_image_bytes = f.read(stego_image_size)
        bitmaps_blob = f.read()
        
        metadata = {
            'version': version, 'codec': codec_map.get(codec_id, 'unknown'), 's': s,
            'align_flag': bool(align_flag), 'width': width, 'height': height,
            'start_offset': start_offset, 'segments_lengths': segments_lengths,
            'segments_indices': segments_indices, 'block_size': block_size,
            'threshold_factor': threshold_factor, 'bits_per_pixel': bits_per_pixel
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

def run_steganography(input_dicom_file, output_dir, base_filename, beta, block_size, threshold_factor, codec='jxl', align_across_planes=False, start_offset=0):
    print('\n')
    logger.info(f"STARTING STEGANOGRAPHY ENCODING\n{'='*100}")
    logger.info(f"Parameters: Beta={beta}, BlockSize={block_size}, Threshold={threshold_factor}, Codec={codec}")

    logger.info("[1/5] Reading source DICOM and extracting metadata...")
    original_dicom = pydicom.dcmread(input_dicom_file)
    image_array = original_dicom.pixel_array

    logger.info(f"\t- Original image: {image_array.shape}, {image_array.dtype}")
    
    # Preserva o tipo original da imagem (8-bit ou 16-bit)
    original_dtype = image_array.dtype

    secret_message = extract_dicom_metadata(original_dicom)
    message_bits = convert_message_to_bits(secret_message)

    logger.info(f"\t- Secret metadata size: {len(message_bits)} bits ({len(secret_message)} chars)")

    logger.info("[2/5] Decomposing image adaptively...")
    global_planes, local_planes, bits_per_pixel = adaptive_modalities_decomposition(image_array, beta=beta)

    logger.info("[3/5] Creating embedding capacity map (dynamic)...")
    capacity_map, allowed_indices = create_capacity_map_dynamic(image_array, required_bits=len(message_bits), block_size=block_size, threshold_factor=threshold_factor)

    logger.info("[4/5] Embedding DICOM metadata into local planes...")
    stego_planes, segments_lengths, segment_indices, used_indices, flip_bits = embed_message_in_planes(
        local_planes, message_bits, allowed_indices, image_array.shape,
        start_offset=start_offset, align_across_planes=align_across_planes
    )

    logger.info("[5/5] Creating final steganography container with clean metadata...")
    stego_image_array = merge_global_local_planes(global_planes, stego_planes, original_dtype)

    # Verificação da reconstrução
    diff = np.abs(image_array.astype(float) - stego_image_array.astype(float))
    max_diff = diff.max()
    mean_diff = diff.mean()
    different_pixels = np.count_nonzero(diff)
    
    logger.info(f"\t- Image fidelity check:")
    logger.info(f"\t  Max diff: {max_diff}, Mean diff: {mean_diff:.6f}")
    logger.info(f"\t  Different pixels: {different_pixels}/{image_array.size} ({different_pixels/image_array.size*100:.2f}%)")

    compressed_bytes = compress_image_data(stego_image_array, codec)
    bitmaps_blob = create_optimized_bitmap_blob(used_indices, flip_bits)

    header_bytes = create_steganography_header_bytes(
        codec=codec, s=len(local_planes), segments_lengths=segments_lengths,
        segments_indices=segment_indices, stego_image_size=len(compressed_bytes),
        width=image_array.shape[1], height=image_array.shape[0], start_offset=start_offset,
        align_across_planes=align_across_planes, block_size=block_size, threshold_factor=threshold_factor,
        bits_per_pixel=bits_per_pixel
    )

    output_bin_file = os.path.join(output_dir, f"{base_filename}.bin")
    os.makedirs(output_dir, exist_ok=True)
    file_size = create_steganography_container(output_bin_file, header_bytes, bitmaps_blob, compressed_bytes)

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

    logger.info("[1/5] Parsing steganography container...")
    metadata, bitmaps_blob, stego_image_bytes = parse_steganography_file(filepath)
    logger.info(f"\t- Codec: {metadata['codec']}, Local Planes: {metadata['s']}, Bits per pixel: {metadata['bits_per_pixel']}")

    logger.info("[2/5] Decompressing image data...")
    stego_array = decompress_image_data(stego_image_bytes, metadata['codec'])
    
    # Determina o dtype correto baseado no bits_per_pixel
    if metadata['bits_per_pixel'] == 8:
        target_dtype = np.uint8
    else:
        target_dtype = np.uint16
        
    if stego_array.dtype != target_dtype:
        stego_array = stego_array.astype(target_dtype)

    logger.info("[3/5] Extracting hidden DICOM metadata...")
    all_stego_planes = extract_bit_planes(stego_array)
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

    metadata_file = f"{output_prefix}_extracted_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        f.write(extracted_metadata_json)
    logger.info(f"\t✔ Extracted metadata saved to: {metadata_file}")

    logger.info("[4/5] Reconstructing original image...")
    restored_image_array = merge_global_local_planes(global_planes, restored_local_planes, target_dtype)

    logger.info("[5/5] Creating DICOM with restored original metadata...")
    restored_dicom = create_clean_dicom_dataset(restored_image_array)
    restored_dicom = restore_dicom_metadata(restored_dicom, extracted_metadata_json)
    restored_dicom_file = f"{output_prefix}_restored.dcm"
    save_dicom_file(restored_dicom, restored_dicom_file)

    logger.info(f"\t✔ Original DICOM with restored metadata: {restored_dicom_file}")
    logger.info(f"\n{'='*100}\n\t\t    DECODING COMPLETE\n")
    
    return restored_dicom, extracted_metadata_json, restored_image_array

def main():
    try:
        input_dicom_file = "../images/mg_16b/666.dcm"
        output_dir = "output"
        beta = 0.4
        block_size = 4
        threshold_factor = 1
        codec = 'jxl'
        base_filename = f"meta_stego_beta{beta}_bs{block_size}_tf{threshold_factor}"

        os.makedirs(output_dir, exist_ok=True)
        file_size, bin_path, stego_dcm_path = run_steganography(
            input_dicom_file, output_dir, base_filename,
            beta, block_size, threshold_factor, codec
        )

        if file_size:
            restored_dicom, extracted_metadata, restored_image = decode_steganography_container(
                bin_path,
                output_prefix=os.path.join(output_dir, f"{base_filename}_decoded")
            )

            original_dicom = pydicom.dcmread(input_dicom_file)
            original_image = original_dicom.pixel_array
            
            if original_image.dtype != restored_image.dtype:
                original_image = original_image.astype(restored_image.dtype)
            
            images_match = np.array_equal(original_image, restored_image)
            
            logger.info(f"VERIFICATION RESULTS\n{'='*100}")
            logger.info(f"Images match perfectly: {images_match}")
            
            if not images_match:
                diff = np.abs(original_image.astype(float) - restored_image.astype(float))
                logger.info(f"Max pixel difference: {diff.max()}")
                logger.info(f"Mean pixel difference: {diff.mean():.6f}")
                logger.info(f"Number of different pixels: {np.count_nonzero(diff)}")
            
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
