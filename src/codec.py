import os
import struct
import zlib
import time
import logging
import json
import imagecodecs
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse

# =============================================================================
# CONFIGURAÇÃO DE LOG & PARÂMETROS GLOBAIS
# =============================================================================
class CleanFormatter(logging.Formatter):
    def format(self, record):
        return f"{record.msg}"

handler = logging.StreamHandler()
handler.setFormatter(CleanFormatter())
logging.root.handlers = []
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
logger = logging.getLogger(__name__)

# --- PARÂMETROS GLOBAIS DE EXPERIMENTO ---
DATASET_DIR = 'images/'
OUTPUT_DIR = 'tcc_results/'
CODECS_TO_TEST = ['jxl', 'j2k', 'jls']
BETAS_TO_TEST = [0.2, 0.8] 
BLOCK_SIZE = 4
TARGET_PERCENTILE = 90 
TARGET_BIT_DEPTH = 16

# --- CONSTANTES DE CÁLCULO ---
MS_PER_S = 1000
BYTES_PER_MB = 1024 * 1024 
# ----------------------------------------

# =============================================================================
# VISUAL DEBUGGING HELPERS
# =============================================================================

def save_visual_debug_image(data: np.ndarray, path: str, title: str = None, normalize: bool = True, cmap: str = 'gray', show_colorbar: bool = False, hline_pos: int = None, debug_mode: bool = True, side_by_side_img: np.ndarray = None, side_by_side_title: str = "Original"):
    if not debug_mode: return
    try:
        num_plots = 2 if side_by_side_img is not None else 1
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
        if num_plots == 1: axes = [axes]

        # Plot da imagem principal (data)
        ax1 = axes[0]
        
        if normalize and data.max() > data.min():
            d_min, d_max = data.min(), data.max()
            display_data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            display_data = data
        
        img_plot = ax1.imshow(display_data, cmap=cmap, interpolation="nearest")
        
        if show_colorbar:
            fig.colorbar(img_plot, ax=ax1, fraction=0.046, pad=0.04)
            
        if hline_pos is not None and hline_pos < data.shape[0]:
            ax1.axhline(y=hline_pos, color='red', linestyle='--', linewidth=1.5, label='Scan Limit')
            ax1.legend(loc='upper right', fontsize=8, framealpha=0.8)

        if title: ax1.set_title(title, fontsize=10, fontweight='bold')
        ax1.tick_params(labelsize=8)

        # Plot da imagem lado a lado (se existir)
        if num_plots == 2:
            ax2 = axes[1]
            ax2.imshow(side_by_side_img, cmap='gray', interpolation='nearest')
            ax2.set_title(side_by_side_title, fontsize=10, fontweight='bold')
            ax2.tick_params(labelsize=8)
            
        plt.axis("on")
        plt.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"    [SAVED] Image: {os.path.basename(path)}")
        return True
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar imagem '{path}': {e}")
        return False

def save_bitplanes_debug(planes: List[np.ndarray], path: str, split_point: int = -1, debug_mode: bool = True):
    if not debug_mode: return
    try:
        count = len(planes)
        cols = 4
        rows = (count + cols - 1) // cols
        
        plt.figure(figsize=(4 * cols, 4 * rows))
        
        for i in range(count):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(planes[i], cmap='gray', interpolation='nearest')
            
            if split_point >= 0:
                type_label = "[LOCAL]" if i < split_point else "[GLOBAL]"
                color = 'darkred' if i < split_point else 'navy'
            else:
                type_label = ""
                color = 'black'

            title = f"Bit-Plane #{i}\n{type_label}"
            plt.title(title, fontsize=10, fontweight='bold', color=color)
            plt.axis('on')
            plt.xticks([])
            plt.yticks([])
            for spine in plt.gca().spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        logger.info(f"    [SAVED] Image: {os.path.basename(path)}")
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar bitplanes: {e}")

def save_zoomed_comparison(original_img: np.ndarray, stego_img: np.ndarray, roi_coords: tuple, output_path: str, zoom_factor: int = 4, debug_mode: bool = True):
    """
    Cria uma figura com zoom comparando a imagem original, a imagem esteganográfica e o resíduo.
    
    Args:
        original_img (np.ndarray): A imagem original.
        stego_img (np.ndarray): A imagem com a mensagem embutida.
        roi_coords (tuple): Uma tupla (x, y, width, height) definindo a região de interesse.
        output_path (str): Caminho para salvar a figura.
        zoom_factor (int): Fator de ampliação para o título (ex: 4 para 400%).
        debug_mode (bool): Se a função deve ser executada.
    """
    if not debug_mode: return
    try:
        x, y, w, h = roi_coords
        
        # 1. Cortar as imagens na região de interesse
        original_crop = original_img[y:y+h, x:x+w]
        stego_crop = stego_img[y:y+h, x:x+w]
        
        # 2. Calcular o resíduo
        residual = stego_img.astype(np.int32) - original_img.astype(np.int32)
        residual_crop = np.abs(residual[y:y+h, x:x+w])
        
        # 3. Criar a figura com 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Painel A: Original
        axes[0].imshow(original_crop, cmap='gray', interpolation='nearest')
        axes[0].set_title(f"(A) Original ({zoom_factor}00% Zoom)", fontweight='bold')
        
        # Painel B: Stego
        axes[1].imshow(stego_crop, cmap='gray', interpolation='nearest')
        axes[1].set_title(f"(B) Stego ({zoom_factor}00% Zoom)", fontweight='bold')
        
        # Painel C: Resíduo
        im = axes[2].imshow(residual_crop, cmap='inferno', interpolation='nearest')
        axes[2].set_title(f"(C) Diferença (Resíduo)", fontweight='bold')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"    [SAVED] Zoom Comparison: {os.path.basename(output_path)}")
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar comparação com zoom: {e}")

def save_histogram_debug(values: np.ndarray, threshold: float, path: str, title: str, xlabel: str, debug_mode: bool = True):
    if not debug_mode: return
    try:
        plt.figure(figsize=(8, 4))
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        plt.hist(values.ravel(), bins=bins, color='#e74c3c', edgecolor='black', linewidth=0.8, alpha=0.9, log=True)
        
        if threshold is not None:
            plt.axvline(x=threshold, color='navy', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')
            plt.legend()
            
        plt.title(title, fontsize=11, fontweight='bold')
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel("Pixel Count", fontsize=10)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        if values.max() - values.min() < 10:
            plt.xticks(np.arange(values.min(), values.max() + 1))
            
        plt.savefig(path)
        plt.close()
        logger.info(f"    [SAVED] Image: {os.path.basename(path)}")
        return True
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar histograma: {e}")
        return False

# =============================================================================
# DICOM UTILS & METRICS
# =============================================================================

def serialize_dicom_value(value):
    if value is None: return None
    if isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, pydicom.uid.UID): return str(value)
    if isinstance(value, pydicom.multival.MultiValue): return [serialize_dicom_value(x) for x in value]
    if isinstance(value, pydicom.valuerep.DSfloat): return float(value)
    if isinstance(value, pydicom.valuerep.IS): return int(value)
    if isinstance(value, pydicom.valuerep.PersonName): return str(value)
    if isinstance(value, bytes): return f"<binary_{len(value)}b>" if len(value) > 1000 else value.hex()
    return str(value)

def extract_dicom_metadata(dicom_dataset: FileDataset) -> str:
    metadata_dict = {}
    for elem in dicom_dataset:
        if elem.tag == (0x7FE0, 0x0010): continue
        if hasattr(elem, 'value') and elem.value is not None:
            try:
                val = serialize_dicom_value(elem.value)
                if val is not None:
                    metadata_dict[str(elem.tag)] = {'value': val, 'VR': elem.VR}
            except: pass
    
    critical_tags = {}
    fields = ['PatientName', 'PatientID', 'StudyDate', 'Modality', 'PixelSpacing', 'Manufacturer']
    for field in fields:
        if hasattr(dicom_dataset, field):
            critical_tags[field] = serialize_dicom_value(getattr(dicom_dataset, field))
    
    metadata_dict['_critical_tags'] = critical_tags
    json_str = json.dumps(metadata_dict)
    
    logger.info(f"  > Payload Extraction: {len(metadata_dict)} tags found ({len(json_str)} bytes)")
    return json_str

def restore_dicom_metadata(dicom_dataset: FileDataset, metadata_json: str) -> FileDataset:
    try:
        data = json.loads(metadata_json)
        crit = data.pop('_critical_tags', {})
        for k, v in crit.items():
            if v and hasattr(dicom_dataset, k):
                setattr(dicom_dataset, k, v)
        count = 0
        for tag_s, info in data.items():
            try:
                tag = eval(tag_s)
                if tag in dicom_dataset:
                    dicom_dataset[tag].value = info['value']
                    count += 1
            except: pass
        logger.info(f"  > Metadata Restore: {len(data)} tags restored.")
        return dicom_dataset
    except Exception as e:
        logger.error(f"  [!] Metadata restore failed: {e}")
        raise

def create_clean_dicom_dataset(image_array: np.ndarray) -> FileDataset:
    sop_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    series_instance_uid = generate_uid()
    sop_class_uid = "1.2.840.10008.5.1.4.1.1.7"

    max_val = image_array.max() if image_array.size > 0 else 0
    if max_val > 0:
        bits_stored = int(np.ceil(np.log2(float(max_val) + 1.0)))
    else:
        bits_stored = 8
    
    if image_array.dtype == np.uint16:
        bits_allocated = 16
    else:
        bits_allocated = 8
        bits_stored = 8
        
    bits_stored = min(bits_stored, bits_allocated)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class_uid
    file_meta.MediaStorageSOPInstanceUID = sop_instance_uid
    file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)

    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.StudyInstanceUID = study_instance_uid
    ds.SeriesInstanceUID = series_instance_uid
    
    ds.PatientName = "ANONYMIZED"
    ds.PatientID = "000000"
    ds.Modality = "OT"
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    
    dt = datetime.now()
    ds.StudyDate = dt.strftime('%Y%m%d')
    ds.SeriesDate = dt.strftime('%Y%m%d')
    ds.ContentDate = dt.strftime('%Y%m%d')
    ds.StudyTime = dt.strftime('%H%M%S')
    ds.SeriesTime = dt.strftime('%H%M%S')
    ds.ContentTime = dt.strftime('%H%M%S')

    ds.Rows, ds.Columns = image_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    
    ds.PixelData = image_array.tobytes()
    
    logger.info("    - Clean DICOM structure created (Compliant).")
    return ds

def save_dicom_file(ds, path, debug_mode: bool = True):
    ds.save_as(path, enforce_file_format=True)
    if debug_mode:
        logger.info(f"    - Saved DICOM: {os.path.basename(path)}")

def get_image_info_for_exp(img_path):
    """Extrai informações básicas do arquivo DICOM para o loop de experimento."""
    original_dicom = pydicom.dcmread(img_path, stop_before_pixels=True)
    original_size = os.path.getsize(img_path)
    
    try:
        modality = os.path.basename(os.path.dirname(img_path))
    except Exception:
        modality = getattr(original_dicom, 'Modality', 'Unknown')
        
    bits_stored = getattr(original_dicom, 'BitsStored', 16)
    bits_allocated = getattr(original_dicom, 'BitsAllocated', 16)
    shape = (int(getattr(original_dicom, 'Rows', 0)), int(getattr(original_dicom, 'Columns', 0)))
    
    return original_size, modality, bits_stored, bits_allocated, shape


# =============================================================================
# COMPRESSION & CORE ALGORITHMS
# =============================================================================

def compress_image_data(img: np.ndarray, codec: str) -> bytes:
    logger.info(f"  > Compressing ({codec.upper()})...")
    t0 = time.time()
    
    if codec == 'jxl': data = imagecodecs.jpegxl_encode(img, numthreads=0, lossless=True, effort=5)
    elif codec == 'jls': data = imagecodecs.jpegls_encode(img)
    elif codec == 'j2k': data = imagecodecs.jpeg2k_encode(img, reversible=True, numthreads=0)
    #elif codec == 'png': data = imagecodecs.png_encode(img)
    else: raise ValueError(f"Unknown codec: {codec}")
    
    logger.info(f"    - Size: {len(data)} bytes (Time: {time.time()-t0:.4f}s)")
    return data

def decompress_image_data(data, codec):
    if codec == 'jxl': return imagecodecs.jpegxl_decode(data, numthreads=0)
    elif codec == 'jls': return imagecodecs.jpegls_decode(data)
    elif codec == 'j2k': return imagecodecs.jpeg2k_decode(data, numthreads=0)
    #elif codec == 'png': return imagecodecs.png_decode(data)
    else: raise ValueError(f"Unknown codec: {codec}")

def convert_message_to_bits(msg: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(msg.encode('utf-8'), dtype=np.uint8))

def convert_bits_to_message(bits: np.ndarray) -> str:
    if len(bits) % 8 != 0: 
        bits = np.pad(bits, (0, 8 - len(bits) % 8), mode='constant')
    return np.packbits(bits).tobytes().decode('utf-8', 'replace').rstrip('\x00')

def extract_bit_planes(image: np.ndarray) -> List[np.ndarray]:
    img = np.ascontiguousarray(image)
    planes = []
    depth = img.dtype.itemsize * 8
    flat = img.reshape(-1)
    for b in range(depth):
        planes.append(((flat >> b) & 1).astype(np.uint8).reshape(img.shape))
    return planes

def reconstruct_from_bit_planes(planes: List[np.ndarray], dtype) -> np.ndarray:
    h, w = planes[0].shape
    flat_out = np.zeros(h * w, dtype=dtype)
    for b, plane in enumerate(planes):
        flat_out |= (plane.reshape(-1).astype(dtype) << b)
    return flat_out.reshape(h, w)

def calculate_mutual_information(plane: np.ndarray, image: np.ndarray, hist_y: np.ndarray, bins: int, bits_stored: int) -> float:
    """Calcula a informação mútua entre um plano de bits e a imagem, adaptando-se ao bits_stored."""
    X = plane.reshape(-1)
    Y = image.reshape(-1)
    total = Y.size

    # Quantiza a imagem para 12 bits se a profundidade for maior que 8.
    if bits_stored > 8:
        shift = max(0, bits_stored - 8)
        Y = (Y.astype(np.uint32) >> shift).astype(np.int64)
    else:
        Y = Y.astype(np.int64)

    ones = int(X.sum())
    p_x = np.array([total - ones, ones], dtype=np.float64) / total
    mask1 = (X == 1)
    h1 = np.bincount(Y[mask1], minlength=bins)
    h0 = hist_y - h1
    p_joint = np.vstack([h0, h1]) / total

    def ent(p): 
        pnz = p[p > 0]
        return -np.sum(pnz * np.log2(pnz))
    
    return float(ent(p_x) + ent(hist_y/total) - ent(p_joint.ravel()))

def adaptive_modality_decomposition(image: np.ndarray, beta: float, output_dir: str, base_name: str, bits_stored: int, debug_mode: bool = True):
    all_planes = extract_bit_planes(image)
    depth = len(all_planes)
    
    # O cálculo do histograma deve ser consistente com a quantização em calculate_mutual_information
    bins = 4096 if depth > 8 else 256
    shift = max(0, bits_stored - 8) if bits_stored > 8 else 0
    img_quantized = (image.astype(np.uint32) >> shift)
    hist = np.bincount(img_quantized.ravel(), minlength=bins).astype(np.float64)
    p = hist / hist.sum()
    total_H = -np.sum(p[p > 0] * np.log2(p[p > 0]))
    target_H = beta * total_H

    if debug_mode:
        logger.info(f"  > Entropy Analysis:")
        logger.info(f"    - H(x) Image: {total_H:.4f} bits (Max Depth: {bits_stored})")
        logger.info(f"    - Target Cumulative MI (Beta={beta}): {target_H:.4f}")

    cum_mi = 0.0
    s = depth
    found_s = False
    
    for i in range(bits_stored):
        plane = all_planes[i]
        
        mi = calculate_mutual_information(plane, image, hist, bins, bits_stored)
        cum_mi += mi
        status = ""
        if not found_s and cum_mi >= target_H:
            s = i + 1
            status = "  <-- CUTOFF POINT (s)"
            found_s = True
            if debug_mode:
                logger.info(f"    - Plane {i}: MI={mi:.5f} | Cumulative={cum_mi:.5f}{status}")
            break
    
    s = min(s, bits_stored)
    s = max(1, min(s, depth - 1))
    
    local = all_planes[:s]
    global_p = all_planes[s:]
    
    save_bitplanes_debug(all_planes, os.path.join(output_dir, f"{base_name}_debug_all_planes.png"), split_point=s, debug_mode=debug_mode)

    return global_p, local, depth

def create_capacity_map_lge(image: np.ndarray, block_size: int, target_percentile: float, output_dir: str, base_name: str, required_bits: int = None, debug_mode: bool = True):
    img = image.astype(np.int32)
    h, w = img.shape
    # LGE significa Local Gradient Energy

    # 1. CÁLCULO VETORIZADO DO LGE
    dh = np.abs(img[:, 1:] - img[:, :-1])
    dh = np.pad(dh, ((0,0),(0,1)))

    dv = np.abs(img[1:, :] - img[:-1, :])
    dv = np.pad(dv, ((0,1),(0,0)))
    
    lge = dh + dv
    
    # 2. MÉDIA POR BLOCOS
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    lge_padded = np.pad(lge, ((0,h_pad), (0,w_pad)), mode='edge')
    
    bh, bw = lge_padded.shape[0] // block_size, lge_padded.shape[1] // block_size
    blocks = lge_padded.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)
    block_energy = blocks.mean(axis=(2,3))
    
    # 3. THRESHOLDING
    flat = block_energy.ravel()
    calculated_threshold = np.percentile(flat, target_percentile)
    
    if debug_mode:
        p50, p75 = np.percentile(flat, [50, 75])
        logger.info(f"  > LGE Stats: P50={p50:.1f} | P75={p75:.1f} | Cutoff={calculated_threshold:.1f}")
        save_histogram_debug(flat, calculated_threshold, os.path.join(output_dir, f"{base_name}_debug_energy_hist.png"), 
                             f"LGE Distribution", "Energy", debug_mode=debug_mode)

    # 4. MÁSCARA E LISTA DE PIXELS
    mask = block_energy >= calculated_threshold
    
    # Expande a máscara de blocos para máscara de pixels
    capacity_map_full = np.kron(mask, np.ones((block_size, block_size), dtype=np.uint8))[:h, :w]
    
    # Lista completa de todos os pixels onde poderíamos esconder dados
    allowed_full = np.where(capacity_map_full.ravel() == 1)[0].astype(np.int64)
    
    total_capacity = len(allowed_full)

    if required_bits is not None:
        allowed = allowed_full[:required_bits]
        if debug_mode:
            logger.info(f"  > Early Exit Applied: Truncated {total_capacity} -> {required_bits} pixels.")
            
            last_idx = allowed[-1]
            last_row, _ = np.unravel_index(last_idx, (h, w))
            scan_limit_y = (last_row // block_size + 1) * block_size

            save_visual_debug_image(capacity_map_full*255, os.path.join(output_dir, f"{base_name}_debug_capacity.png"), title=f"Capacity Map (Scanned until Y={scan_limit_y})", 
                                    cmap='gray', hline_pos=scan_limit_y, debug_mode=debug_mode, side_by_side_img=image, side_by_side_title="Original Image")
            save_visual_debug_image(block_energy, os.path.join(output_dir, f"{base_name}_debug_energy_map.png"), 
                                    title="LGE Energy Map", cmap='viridis', show_colorbar=True, debug_mode=debug_mode)   
    else:
        allowed = allowed_full
        
    return capacity_map_full, allowed

def embed_message_in_planes(planes, msg_bits, allowed):
    if len(msg_bits) > len(allowed):
        logger.error(f"  [!] OVERFLOW: Msg={len(msg_bits)} bits > Capacity={len(allowed)} bits.")
        raise ValueError("Capacity Overflow")
    
    n_planes = len(planes)
    base = len(msg_bits) // n_planes
    rem = len(msg_bits) % n_planes
    seg_lens = [base + 1 if i < rem else base for i in range(n_planes)]
    
    logger.info(f"  > Embedding Distribution:")
    for i, l in enumerate(seg_lens):
        if l > 0: logger.info(f"    - Plane {i}: {l} bits")

    stego_planes = [p.copy() for p in planes]
    flips = np.zeros(len(msg_bits), dtype=np.uint8)
    idx_used = allowed[:len(msg_bits)]
    
    cursor = 0
    cursor_idx = 0
    h, w = planes[0].shape
    
    for i, count in enumerate(seg_lens):
        if count == 0: continue
        bits = msg_bits[cursor : cursor + count]
        idxs = idx_used[cursor_idx : cursor_idx + count]
        
        y, x = np.unravel_index(idxs, (h, w))
        orig = stego_planes[i][y, x]
        new = (orig & 0xFE) | bits
        stego_planes[i][y, x] = new
        flips[cursor : cursor+count] = (orig & 1) ^ (new & 1)
        
        cursor += count
        cursor_idx += count
        
    return stego_planes, seg_lens, idx_used, flips

def pack_bitmap(used: np.ndarray, flips: np.ndarray):
    diffs = np.diff(np.insert(used.astype(np.uint32), 0, 0)).astype(np.uint32)
    fp = np.packbits(flips, bitorder='little')
    raw = struct.pack("<I", len(diffs)) + diffs.tobytes() + fp.tobytes() # type: ignore
    packed = zlib.compress(raw, level=1)
    return packed

def save_container(path, codec, s, w, h, bs, depth, lens, img_bytes, bmp_bytes):
    codec_id = {'j2k':1, 'jls':2, 'jxl':3}.get(codec, 0)

    hdr = struct.pack("<BBBB HH IHHB", 1, codec_id, s, 0, w, h, len(img_bytes), bs, len(lens), depth)
    lens_b = b''.join(struct.pack("<I", x) for x in lens)
    
    with open(path, "wb") as f:
        f.write(b"STGC")
        f.write(struct.pack("<I", len(hdr)+len(lens_b)))
        f.write(hdr)
        f.write(lens_b)
        f.write(img_bytes)
        f.write(bmp_bytes)
    return os.path.getsize(path)

# =============================================================================
# PIPELINE CONTROL (MODO EXPERIMENTO E SINGLE TEST)
# =============================================================================


def run_encoder(dicom_path, out_dir, beta, b_size, initial_percentile, codec, message_override=None, debug_mode: bool = False):
    t_start = time.time()
    
    if debug_mode:
        print("\n" + "="*60)
        logger.info("STARTING ENCODER (ADAPTIVE MODE)")
        logger.info(f"File: {os.path.basename(dicom_path)}")
    
    # [1] LOAD
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    bits_stored = getattr(ds, 'BitsStored', img.dtype.itemsize * 8)
    
    # Prepara nome base e metadados
    try:
        modality = os.path.basename(os.path.dirname(dicom_path))
    except:
        modality = 'unknown'
    
    fname = os.path.splitext(os.path.basename(dicom_path))[0]
    
    if message_override:
        meta_str = message_override
    else:
        meta_str = extract_dicom_metadata(ds)
    bits = convert_message_to_bits(meta_str)

    # [2] DECOMPOSITION (Feito uma vez, pois não depende do percentil)
    temp_base_name = f"{fname}_{modality}_b{beta}_{codec}"
    glo, loc, depth = adaptive_modality_decomposition(img, beta, out_dir, temp_base_name, bits_stored=bits_stored, debug_mode=debug_mode)

    # [3] LOOP ADAPTATIVO DE CAPACIDADE
    current_percentile = initial_percentile
    min_percentile = 50 # Abaixo disso a qualidade visual degrada muito
    success = False
    
    # Variáveis para o loop
    c_map, allowed, stego_loc, lens, used, flips = None, None, None, None, None, None
    
    while current_percentile >= min_percentile:
        if debug_mode: logger.info(f"[...] Trying Percentile P{current_percentile}...")
        
        c_map, allowed = create_capacity_map_lge(
            img, b_size, current_percentile, out_dir, temp_base_name, 
            required_bits=len(bits), debug_mode=debug_mode
        )
        
        # Verifica se cabe
        if len(bits) <= len(allowed):
            success = True
            if debug_mode: logger.info(f"    -> Success! Found capacity at P{current_percentile}")
            break
        else:
            if debug_mode: logger.info(f"    -> Failed (Overflow). Reducing percentile...")
            current_percentile -= 10 # Tenta ser menos seletivo
            
    if not success:
        logger.error(f"Encoding FAILED: Message too big even at P{min_percentile}.")
        return None, None, 0.0, None

    # [4] EMBED (Agora com o percentil garantido)
    base_name_full = f"{fname}_{modality}_b{beta}_p{current_percentile}_{codec}" # Nome final com o P correto
    
    stego_loc, lens, used, flips = embed_message_in_planes(loc, bits, allowed)
    stego_img = reconstruct_from_bit_planes(stego_loc + glo, img.dtype)
    
    # (Opcional) Recriar os debugs visuais finais com o nome correto e o percentil vencedor
    if debug_mode:
        diff = stego_img.astype(np.int32) - img.astype(np.int32)
        save_visual_debug_image(np.abs(diff), os.path.join(out_dir, f"{base_name_full}_debug_residuals.png"), title=f"Residuals (Final P{current_percentile})", cmap='inferno', show_colorbar=True, debug_mode=True)
        
        # Adiciona a chamada para a nova função de zoom
        # ROI (x, y, width, height) - Escolhi uma área que geralmente tem bordas em CTs de tórax.
        roi_of_interest = (200, 250, 64, 64) 
        save_zoomed_comparison(img, stego_img, roi_of_interest, os.path.join(out_dir, f"{base_name_full}_debug_zoom_comparison.png"), zoom_factor=4, debug_mode=True)

    # [5] PACKAGE
    if stego_img.dtype.itemsize * 8 > 8:
        stego_img_final = stego_img.astype(np.uint16)
        if stego_img_final.dtype.byteorder == '>': stego_img_final = stego_img_final.byteswap().newbyteorder() 
    else:
        stego_img_final = stego_img.astype(np.uint8)
        
    stego_img_reshaped = np.squeeze(stego_img_final)
    stego_img_contiguous = np.ascontiguousarray(stego_img_reshaped)
    
    img_bin = compress_image_data(stego_img_contiguous, codec) 
    bmp_bin = pack_bitmap(used, flips)
    
    bin_path = os.path.join(out_dir, f"{base_name_full}.bin")
    # Salva usando o percentil vencedor no metadata (embora não vá no header binário, é bom saber)
    sz = save_container(bin_path, codec, len(loc), img.shape[1], img.shape[0], b_size, depth, lens, img_bin, bmp_bin)
    
    if debug_mode:
        mock_ds = create_clean_dicom_dataset(stego_img)
        save_dicom_file(mock_ds, os.path.join(out_dir, f"{base_name_full}_container.dcm"), debug_mode=True)

    total_t = time.time() - t_start
    if not debug_mode:
        logger.info(f"Encode Success | P{current_percentile} | {total_t:.4f}s | {os.path.basename(bin_path)}")
    
    return bin_path, meta_str, total_t, stego_img

def run_decoder(bin_path, out_dir, debug_mode: bool = False):
    t_start = time.time()
    
    if debug_mode:
        print("\n" + "="*60)
        logger.info("STARTING DECODER PROCESS (DEBUG MODE)")
        logger.info(f"Input File: {os.path.basename(bin_path)}")
        print("-" * 60)
    else:
        logger.info(f"Decoding {os.path.basename(bin_path)}...")

    c_map = {1:'j2k', 2:'jls', 3:'jxl'}
    
    if debug_mode: logger.info("[+] Step 1: Unpacking")
    with open(bin_path, "rb") as f:
        if f.read(4) != b"STGC": raise ValueError("Invalid Signature")
        h_len = struct.unpack("<I", f.read(4))[0]
        h_data = f.read(h_len)
        
        base_sz = struct.calcsize("<BBBBHH IH H B")
        
        ver, cid, s, _, w, h, isz, bs, sc, bpp = struct.unpack("<BBBBHH IH H B", h_data[:base_sz])
        
        lens = []
        curr = base_sz
        for _ in range(sc):
            lens.append(struct.unpack("<I", h_data[curr:curr+4])[0])
            curr += 4
            
        img_data = f.read(isz)
        bmp_data = f.read()
        
    if debug_mode: logger.info(f"  > Header (v{ver}): Codec={c_map.get(cid)}, S={s}, Blocks={bs}, Segments={sc}")

    if debug_mode: logger.info("[+] Step 2: Decompressing & Extracting")
    stego_img = decompress_image_data(img_data, c_map.get(cid))
    dtype = np.uint16 if bpp > 8 else np.uint8
    stego_img = stego_img.astype(dtype)
    
    planes = extract_bit_planes(stego_img)
    loc, glo = planes[:s], planes[s:]
    
    raw_bmp = zlib.decompress(bmp_data)
    cnt = struct.unpack("<I", raw_bmp[:4])[0]
    diffs = np.frombuffer(raw_bmp[4:4+cnt*4], dtype=np.uint32)
    used_idx = np.cumsum(diffs).astype(np.int64)
    flips = np.unpackbits(np.frombuffer(raw_bmp[4+cnt*4:], dtype=np.uint8), bitorder='little')[:sum(lens)]
    
    bits_out = np.zeros(sum(lens), dtype=np.uint8)
    restored_loc = [p.copy() for p in loc]
    
    cursor = 0
    for i, count in enumerate(lens):
        if count == 0: continue
        idx = used_idx[cursor : cursor+count]
        f_bits = flips[cursor : cursor+count]
        y, x = np.unravel_index(idx, (h, w))
        vals = loc[i][y, x]
        bits_out[cursor : cursor+count] = vals & 1
        restored_loc[i][y, x] = (vals & 0xFE) | ((vals & 1) ^ f_bits)
        cursor += count
        
    msg_str = convert_bits_to_message(bits_out)
    
    if debug_mode: logger.info("[+] Step 3: Restoring DICOM")
    full_img = reconstruct_from_bit_planes(restored_loc + glo, dtype)
    
    if debug_mode:
        base = os.path.splitext(os.path.basename(bin_path))[0]
        
        # Salva a mensagem recuperada como JSON
        try:
            msg_json = json.loads(msg_str)
            json_path = os.path.join(out_dir, f"{base}_recovered_message.json")
            with open(json_path, 'w') as f:
                json.dump(msg_json, f, indent=4)
            logger.info(f"    [SAVED] Recovered message: {os.path.basename(json_path)}")
        except json.JSONDecodeError:
            logger.warning("    [!] Failed to parse recovered message as JSON.")

        ds = create_clean_dicom_dataset(full_img)
        ds = restore_dicom_metadata(ds, msg_str)
        save_dicom_file(ds, os.path.join(out_dir, f"{base}_restored.dcm"), debug_mode=debug_mode)
    
    total_t = time.time() - t_start
    if not debug_mode:
        logger.info(f"Decode Success | Time: {total_t:.4f}s")
    
    return full_img, msg_str, total_t

def process_single_image(img_path, output_dir, codes, betas, block_size, percentile, target_bit_depth, debug_mode):
    """Executa o processamento com NORMALIZAÇÃO [0-1] para gerar MSE compatível com a literatura."""
    results = []
    
    # 1. Carregar imagem original
    try:
        logger.info(f"--- Processando: {os.path.basename(img_path)} ---")
        
        original_dicom_full = pydicom.dcmread(img_path)
        original_array = original_dicom_full.pixel_array
        original_size, modality, bits_stored, bits_allocated, shape = get_image_info_for_exp(img_path)

        if bits_allocated != target_bit_depth:
            return []
            
        secret_message_json = extract_dicom_metadata(original_dicom_full)
        message_bits_count = len(convert_message_to_bits(secret_message_json))
        metadata_size_bytes = len(secret_message_json.encode('utf-8'))
        
    except Exception as e:
        logger.error(f"Erro ao carregar {img_path}: {e}")
        return []

    # 2. Loop de Testes
    for beta in betas:
        for codec_name in codes:
            try:
                # A. ENCODE
                bin_file_result, original_msg_check, total_encoding_time, stego_array = run_encoder(
                    img_path, output_dir, beta, block_size, percentile, codec_name, debug_mode=debug_mode
                )
                
                if bin_file_result is None: continue
            
                # B. DECODE
                restored_image, decoded_msg_check, decoding_time = run_decoder(
                    bin_file_result, output_dir, debug_mode=debug_mode 
                )
                
                # === NORMALIZAÇÃO (O SEGREDO PARA MSE BAIXO) ===
                # Converte para float e normaliza entre 0.0 e 1.0, igual ao metrics.py
                
                # 1. Prepara Original
                orig_float = original_array.astype(np.float64)
                orig_norm = orig_float - orig_float.min()
                if orig_norm.max() > 0:
                    orig_norm /= orig_norm.max()
                
                # 2. Prepara Stego (com correção de view se necessário)
                if original_array.dtype == np.int16 and stego_array.dtype == np.uint16:
                    stego_float = stego_array.view(np.int16).astype(np.float64)
                else:
                    stego_float = stego_array.astype(np.float64)
                
                stego_norm = stego_float - stego_float.min()
                if stego_norm.max() > 0:
                    stego_norm /= stego_norm.max()

                # 3. Prepara Restaurada
                if original_array.dtype == np.int16 and restored_image.dtype == np.uint16:
                    rest_float = restored_image.view(np.int16).astype(np.float64)
                else:
                    rest_float = restored_image.astype(np.float64)
                
                rest_norm = rest_float - rest_float.min()
                if rest_norm.max() > 0:
                    rest_norm /= rest_norm.max()

                # -----------------------------------------------

                # C. Métricas (Agora calculadas sobre os valores 0.0-1.0)
                stego_mse = mse(orig_norm, stego_norm)
                stego_psnr = psnr(orig_norm, stego_norm, data_range=1.0)
                stego_ssim = ssim(orig_norm, stego_norm, data_range=1.0, channel_axis=None)
                
                restored_mse = mse(orig_norm, rest_norm) # Deve ser 0.0
                
                # Performance
                final_bin_size = os.path.getsize(bin_file_result)
                shape_size = shape[0] * shape[1]
                bpp = (final_bin_size * 8) / shape_size
                compression_ratio = original_size / final_bin_size if final_bin_size > 0 else float('inf')
                
                original_size_mb = original_size / BYTES_PER_MB
                encoding_speed_ms_mb = (total_encoding_time * MS_PER_S) / original_size_mb if original_size_mb > 0 else 0
                decoding_speed_ms_mb = (decoding_time * MS_PER_S) / original_size_mb if original_size_mb > 0 else 0
                
                # Reversibilidade Binária (nos dados brutos originais)
                # Aqui usamos os dados crus para garantir bit-perfect
                arr_rest_view = restored_image
                if original_array.dtype == np.int16 and restored_image.dtype == np.uint16:
                    arr_rest_view = restored_image.view(np.int16)
                
                reversibility_check = np.array_equal(original_array, arr_rest_view) 
                message_check = original_msg_check == decoded_msg_check
                
                results.append({
                    'Image_File': os.path.basename(img_path),
                    'Modality': modality,
                    'Bits_Stored': bits_stored,
                    'Original_Size_Bytes': original_size,
                    'Metadata_Size_Bytes': metadata_size_bytes,
                    'Message_Size_Bits': message_bits_count,
                    'Beta': beta,
                    'Codec': codec_name,
                    'Percentile': percentile,
                    'PSNR_dB': stego_psnr,
                    'SSIM': stego_ssim,
                    'MSE': stego_mse,          # Agora estará na escala 0.00xxx
                    'Restored_MSE': restored_mse,
                    'Final_Bin_Size_Bytes': final_bin_size,
                    'Bpp': bpp,
                    'CR': compression_ratio,
                    'Encoding_Speed_ms_MB': encoding_speed_ms_mb,
                    'Decoding_Speed_ms_MB': decoding_speed_ms_mb,
                    'Total_Encoding_Time_s': total_encoding_time,
                    'Decoding_Time_s': decoding_time,
                    'Reversibility_Check': reversibility_check,
                    'Reversibility_Message': message_check
                })
                
                logger.info(f"  ✓ {codec_name.upper()} | B={beta} | MSE={stego_mse:.6f} | PSNR={stego_psnr:.2f}dB")

            except Exception as e:
                logger.error(f"  [ERRO] {os.path.basename(img_path)} ({codec_name}): {e}")
                continue
    
    return results

def run_full_experiment_mode(output_dir, dataset_dir, codes, betas, block_size, percentile, target_bit_depth, debug_mode):
    """Executa todos os experimentos em modo sequencial e salva os resultados em CSV."""
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    image_paths = glob.glob(os.path.join(dataset_dir, '**/*.dcm'), recursive=True)
    if not image_paths:
        logger.error(f"Nenhum arquivo .dcm encontrado em '{dataset_dir}'. Verifique o caminho.")
        return

    all_results = []
    start_time = time.time()
    total_images = len(image_paths)
    
    logger.info(f"--- INICIANDO EXPERIMENTOS ({total_images} imagens) ---")
    logger.info(f"Filtro: BitsAllocated = {target_bit_depth}. DEBUG={debug_mode}")

    for i, img_path in enumerate(image_paths):
        current_index = i + 1
        img_name = os.path.basename(img_path)
        
        logger.info(f"\n[ PROCESSO {current_index}/{total_images} ] Imagem: {img_name}")

        results = process_single_image(img_path, output_dir, codes, betas, block_size, percentile, target_bit_depth, debug_mode)
        all_results.extend(results)

    total_time = time.time() - start_time
    
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(output_dir, 'results_sequential.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"EXPERIMENTOS CONCLUÍDOS. Tempo total: {total_time:.2f}s")
        logger.info(f"Resultados salvos em: {csv_path}")
        logger.info(f"{'='*50}")
    else:
        logger.error("Nenhum resultado foi gerado!")

### Bloco Principal de Controle de Execução
if __name__ == "__main__":
    
    MODE = 'EXPERIMENT' # Opções: 'EXPERIMENT' ou 'SINGLE_TEST'
    
    SINGLE_TEST_FILE = "images/MG/000.dcm"
    SINGLE_TEST_CODEC = 'jxl'
    
    if MODE == 'EXPERIMENT':
        run_full_experiment_mode(OUTPUT_DIR, DATASET_DIR, CODECS_TO_TEST, BETAS_TO_TEST, BLOCK_SIZE, TARGET_PERCENTILE, TARGET_BIT_DEPTH, debug_mode=False)
        
    elif MODE == 'SINGLE_TEST':
        if os.path.exists(SINGLE_TEST_FILE):
            # Executa o encoder e depois o decoder para um teste de ponta a ponta
            bin_file_path, _, _, _ = run_encoder(
                SINGLE_TEST_FILE, OUTPUT_DIR, BETAS_TO_TEST[0], BLOCK_SIZE, TARGET_PERCENTILE, SINGLE_TEST_CODEC, debug_mode=True
            )
            if bin_file_path:
                run_decoder(bin_file_path, OUTPUT_DIR, debug_mode=True)

        else:
            logger.error(f"Arquivo de teste único não encontrado: {SINGLE_TEST_FILE}")