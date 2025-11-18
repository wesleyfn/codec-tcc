import os
import struct
import zlib
import time
import logging
import json
import imagecodecs
from datetime import datetime
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# =============================================================================
# CONFIGURAÇÃO DE LOG (ESTILO LIMPO)
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

# =============================================================================
# VISUAL DEBUGGING HELPERS (ESTILO CIENTÍFICO)
# =============================================================================

def save_visual_debug_image(data: np.ndarray, path: str, title: str = None, normalize: bool = True, cmap: str = 'gray', show_colorbar: bool = False, hline_pos: int = None):
    """Salva imagem com eixos (axes), visual clean e opcionalmente uma linha horizontal."""
    try:
        plt.figure(figsize=(6, 6))
        
        if normalize and data.max() > data.min():
            d_min, d_max = data.min(), data.max()
            display_data = ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            display_data = data

        img_plot = plt.imshow(display_data, cmap=cmap, interpolation="nearest", aspect='auto')
        
        if show_colorbar:
            plt.colorbar(img_plot, fraction=0.046, pad=0.04)
            
        if hline_pos is not None and hline_pos < data.shape[0]:
            plt.axhline(y=hline_pos, color='red', linestyle='--', linewidth=1.5, label='Scan Limit')
            # Coloca a legenda fora ou num canto para não atrapalhar
            plt.legend(loc='upper right', fontsize=8, framealpha=0.8)

        if title: 
            plt.title(title, fontsize=10, fontweight='bold')
            
        plt.axis("on")
        plt.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return True
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar imagem '{path}': {e}")
        return False

def save_bitplanes_debug(planes: List[np.ndarray], path: str, split_point: int = -1):
    """Salva TODOS os bitplanes em um único grid."""
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
                
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar bitplanes: {e}")

def save_histogram_debug(values: np.ndarray, threshold: float, path: str, title: str, xlabel: str):
    """Histograma focado na distribuição de erro (-1, 0, +1)."""
    try:
        plt.figure(figsize=(8, 4))
        bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        plt.hist(values.ravel(), bins=bins, color='#e74c3c', edgecolor='black', linewidth=0.8, alpha=0.9)
        
        if threshold is not None:
            plt.axvline(x=threshold, color='navy', linestyle='--', linewidth=1.5, label=f'Threshold ({threshold})')
            plt.legend()
            
        plt.title(title, fontsize=11, fontweight='bold')
        plt.xlabel(xlabel, fontsize=10)
        plt.ylabel("Pixel Count", fontsize=10)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        if values.max() - values.min() < 10:
            plt.xticks(np.arange(values.min(), values.max() + 1))
            
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return True
    except Exception as e:
        logger.warning(f"    [!] Erro ao salvar histograma: {e}")
        return False

# =============================================================================
# DICOM UTILS
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
        if elem.tag == (0x7FE0, 0x0010) or elem.tag.group > 0x0008: continue
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
        logger.info(f"  > Metadata Restore: {count + len(crit)} tags restored.")
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

def save_dicom_file(ds, path):
    ds.save_as(path, enforce_file_format=True)
    logger.info(f"    - Saved DICOM: {os.path.basename(path)}")

# =============================================================================
# COMPRESSION
# =============================================================================

def compress_image_data(img: np.ndarray, codec: str) -> bytes:
    logger.info(f"  > Compressing ({codec.upper()})...")
    t0 = time.time()
    if codec == 'jxl': data = imagecodecs.jpegxl_encode(img, lossless=True, effort=6, photometric='GRAY')
    elif codec == 'jls': data = imagecodecs.jpegls_encode(img)
    elif codec == 'j2k': data = imagecodecs.jpeg2k_encode(img, reversible=True)
    elif codec == 'png': data = imagecodecs.png_encode(img)
    else: raise ValueError(f"Unknown codec: {codec}")
    logger.info(f"    - Size: {len(data)} bytes (Time: {time.time()-t0:.4f}s)")
    return data

def decompress_image_data(data, codec):
    if codec == 'jxl': return imagecodecs.jpegxl_decode(data)
    elif codec == 'jls': return imagecodecs.jpegls_decode(data)
    elif codec == 'j2k': return imagecodecs.jpeg2k_decode(data)
    elif codec == 'png': return imagecodecs.png_decode(data)
    raise ValueError(f"Unknown codec: {codec}")

def convert_message_to_bits(msg: str) -> np.ndarray:
    return np.unpackbits(np.frombuffer(msg.encode('utf-8'), dtype=np.uint8))

def convert_bits_to_message(bits: np.ndarray) -> str:
    if len(bits) % 8 != 0: bits = np.pad(bits, (0, 8 - len(bits)%8), mode='constant')
    return np.packbits(bits).tobytes().decode('utf-8', 'replace').rstrip('\x00')

# =============================================================================
# CORE ALGORITHMS
# =============================================================================

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

def calculate_mutual_information(plane, image, hist_y, bins) -> float:
    X = plane.reshape(-1)
    Y = image.reshape(-1)
    total = Y.size
    if image.dtype.itemsize * 8 > 8: Y = (Y.astype(np.uint32) >> 4).astype(np.int64)
    else: Y = Y.astype(np.int64)
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

def adaptive_modalities_decomposition(image: np.ndarray, beta: float, output_dir: str, base_name: str):
    planes = extract_bit_planes(image)
    depth = len(planes)
    bins = 4096 if depth > 8 else 256
    img_q = (image >> 4) if depth > 8 else image
    hist = np.bincount(img_q.reshape(-1), minlength=bins).astype(np.float64)
    p = hist / hist.sum()
    total_H = -np.sum(p[p>0] * np.log2(p[p>0]))
    target_H = beta * total_H
    logger.info(f"  > Entropy Analysis:")
    logger.info(f"    - H(x) Image: {total_H:.4f} bits")
    logger.info(f"    - Target Cumulative MI (Beta={beta}): {target_H:.4f}")
    cum_mi = 0.0
    s = depth
    found_s = False
    for i, plane in enumerate(planes):
        mi = calculate_mutual_information(plane, image, hist, bins)
        cum_mi += mi
        status = ""
        if not found_s and cum_mi >= target_H:
            s = i + 1
            status = "  <-- CUTOFF POINT (s)"
            found_s = True
        logger.info(f"    - Plane {i}: MI={mi:.5f} | Cumulative={cum_mi:.5f}{status}")
    s = max(1, min(s, depth - 1))
    local = planes[:s]
    global_p = planes[s:]
    save_bitplanes_debug(planes, os.path.join(output_dir, f"{base_name}_debug_all_planes.png"), split_point=s)
    return global_p, local, depth

def create_capacity_map_lge(image: np.ndarray, block_size: int, target_percentile: float, output_dir: str, base_name: str, required_bits: int = None):
    """
    Gera mapa de capacidade definindo o threshold automaticamente via percentil.
    target_percentile: 0 a 100 (ex: 75 seleciona os 25% blocos mais complexos).
    """
    img = image.astype(np.int32)
    h, w = img.shape
    
    # 1. LGE Calc
    dh = np.abs(img[:, 1:] - img[:, :-1]); dh = np.pad(dh, ((0,0),(0,1)))
    dv = np.abs(img[1:, :] - img[:-1, :]); dv = np.pad(dv, ((0,1),(0,0)))
    lge = dh + dv
    
    # 2. Block Processing
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    lge_padded = np.pad(lge, ((0,h_pad), (0,w_pad)), mode='edge')
    
    bh, bw = lge_padded.shape[0]//block_size, lge_padded.shape[1]//block_size
    blocks = lge_padded.reshape(bh, block_size, bw, block_size).transpose(0, 2, 1, 3)
    block_energy = blocks.mean(axis=(2,3))
    
    # 3. Cálculo Automático do Threshold
    flat = block_energy.ravel()
    
    # Se o usuário passar P75, ele quer cortar os 75% mais lisos e manter o topo.
    calculated_threshold = np.percentile(flat, target_percentile)
    
    # Debug Stats
    p25, p50, p75 = np.percentile(flat, [25, 50, 75])
    logger.info(f"  > LGE Stats (Block={block_size}):")
    logger.info(f"    - Distribution: P25={p25:.1f} | P50={p50:.1f} | P75={p75:.1f}")
    logger.info(f"    - Auto-Threshold: Target P{target_percentile} => Value {calculated_threshold:.1f}")
    
    save_histogram_debug(flat, calculated_threshold, os.path.join(output_dir, f"{base_name}_debug_energy_hist.png"), 
                         f"LGE Distribution (Cutoff at P{target_percentile})", "Energy Value")

    # 4. Thresholding
    mask = block_energy >= calculated_threshold
    
    # Lógica de "Early Exit"
    scan_limit_y = img.shape[0]
    
    if required_bits is not None:
        pixels_per_block = block_size * block_size
        row_counts = np.sum(mask, axis=1)
        current_blocks = 0
        stop_row_idx = -1
        
        for r, count in enumerate(row_counts):
            current_blocks += count
            current_capacity_bits = current_blocks * pixels_per_block
            
            if current_capacity_bits >= required_bits:
                stop_row_idx = r
                break
        
        if stop_row_idx != -1:
            mask[stop_row_idx+1:, :] = False
            scan_limit_y = (stop_row_idx + 1) * block_size
            logger.info(f"  > Optimized Scan: Stopped at row {stop_row_idx} (Pixel Y={scan_limit_y})")
        else:
            logger.info("  > Full Scan Required: Message needs entire capacity or more.")

    capacity_map = np.kron(mask, np.ones((block_size, block_size), dtype=np.uint8))[:h, :w]
    allowed = np.where(capacity_map.ravel() == 1)[0].astype(np.int64)
    
    # Mapas Visuais
    save_visual_debug_image(block_energy, os.path.join(output_dir, f"{base_name}_debug_energy_map.png"), 
                            title="LGE Energy Map", cmap='viridis', show_colorbar=True)
                            
    save_visual_debug_image(capacity_map*255, os.path.join(output_dir, f"{base_name}_debug_capacity.png"), 
                            title=f"Capacity Mask (P{target_percentile} > {calculated_threshold:.1f})", cmap='gray', hline_pos=scan_limit_y)
    
    return capacity_map, allowed

# =============================================================================
# EMBEDDING & IO
# =============================================================================

def embed_message_in_planes(planes, msg_bits, allowed, shape):
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
    h, w = shape
    
    for i, count in enumerate(seg_lens):
        if count == 0: continue
        bits = msg_bits[cursor : cursor+count]
        idxs = idx_used[cursor_idx : cursor_idx+count]
        
        y, x = np.unravel_index(idxs, (h, w))
        orig = stego_planes[i][y, x]
        new = (orig & 0xFE) | bits
        stego_planes[i][y, x] = new
        flips[cursor : cursor+count] = (orig ^ new) & 1
        
        cursor += count
        cursor_idx += count
        
    return stego_planes, seg_lens, idx_used, flips

def pack_bitmap(used, flips):
    diffs = np.diff(np.insert(used.astype(np.uint32), 0, 0)).astype(np.uint32)
    fp = np.packbits(flips, bitorder='little')
    raw = struct.pack("<I", len(diffs)) + diffs.tobytes() + fp.tobytes()
    packed = zlib.compress(raw, level=1)
    return packed

def save_container(path, codec, s, w, h, bs, depth, lens, img_bytes, bmp_bytes):
    cid = {'png':1, 'j2k':2, 'jls':3, 'jxl':4}.get(codec, 0)
    # Format: < B(Ver) B(Codec) B(s) B(Pad) H(W) H(H) I(ImgSz) H(BS) H(Segs) B(Depth)
    hdr = struct.pack("<BBBBHH IH H B", 3, cid, s, 0, w, h, len(img_bytes), bs, len(lens), depth)
    
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
# PIPELINE CONTROL
# =============================================================================

def run_encoder(dicom_path, out_dir, beta, b_size, percentile, codec, message_override=None):
    print("\n" + "="*60)
    logger.info("STARTING ENCODER PROCESS")
    logger.info(f"Input File: {os.path.basename(dicom_path)}")
    logger.info(f"Settings: Beta={beta}, BlockSize={b_size}, TargetPercentile=P{percentile}, Codec={codec}")
    print("-" * 60)
    t_start = time.time()

    # [1] LOAD
    logger.info("[+] Phase 1: Loading & Payload Prep")
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    
    # --- ALTERAÇÃO AQUI: Prioriza a mensagem de override ---
    if message_override:
        meta_str = message_override
        logger.info(f"  > Using OVERRIDE message for capacity test.")
    else:
        meta_str = extract_dicom_metadata(ds)
    
    bits = convert_message_to_bits(meta_str)

    # [2] DECOMPOSITION
    logger.info("[+] Phase 2: Adaptive Decomposition")
    base_name = f"stego_b{beta}_p{percentile}"
    glo, loc, depth = adaptive_modalities_decomposition(img, beta, out_dir, base_name)
    logger.info(f"  > Result: {len(loc)} Local planes (Noise) | {len(glo)} Global planes (Structure)")

    # [3] CAPACITY
    logger.info("[+] Phase 3: Capacity Analysis (Auto-LGE)")
    c_map, allowed = create_capacity_map_lge(img, b_size, percentile, out_dir, base_name, required_bits=len(bits))
    
    usage_pct = (len(bits) / len(allowed)) * 100 if len(allowed) > 0 else float('inf')
    logger.info(f"  > Capacity: {len(allowed)} bits | Required: {len(bits)} bits")
    logger.info(f"  > Utilization: {usage_pct:.2f}%")
    
    if len(bits) > len(allowed):
        logger.error("  [!] ERROR: Insufficient Capacity. LOWER the percentile to include more blocks.")
        return None, None

    # [4] EMBED
    logger.info("[+] Phase 4: Embedding")
    stego_loc, lens, used, flips = embed_message_in_planes(loc, bits, allowed, img.shape)
    stego_img = reconstruct_from_bit_planes(stego_loc + glo, img.dtype)
    diff = stego_img.astype(np.int32) - img.astype(np.int32)
    save_visual_debug_image(np.abs(diff), os.path.join(out_dir, f"{base_name}_debug_residuals.png"), 
                            title="Difference Map (Residuals)", cmap='inferno', show_colorbar=True)
    save_histogram_debug(diff.ravel(), None, os.path.join(out_dir, f"{base_name}_debug_residuals_hist.png"), 
                         "Pixel Difference Distribution", "Difference Value")
    logger.info(f"  > Max Pixel Change: {np.abs(diff).max()}")

    # [5] PACKAGE
    logger.info("[+] Phase 5: Packaging")
    img_bin = compress_image_data(stego_img, codec)
    bmp_bin = pack_bitmap(used, flips)
    
    bin_path = os.path.join(out_dir, f"{base_name}.stgc")
    sz = save_container(bin_path, codec, len(loc), img.shape[1], img.shape[0], b_size, depth, lens, img_bin, bmp_bin)
    
    mock_ds = create_clean_dicom_dataset(stego_img)
    save_dicom_file(mock_ds, os.path.join(out_dir, f"{base_name}_container.dcm"))

    total_t = time.time() - t_start
    print("-" * 60)
    logger.info(f"ENCODING COMPLETE in {total_t:.2f}s")
    logger.info(f"Output Container: {bin_path} ({sz/1024:.1f} KB)")
    print("="*60 + "\n")
    return bin_path, meta_str

def run_decoder(bin_path, out_dir):
    print("\n" + "="*60)
    logger.info("STARTING DECODER PROCESS")
    logger.info(f"Input File: {os.path.basename(bin_path)}")
    print("-" * 60)
    
    c_map = {1:'png', 2:'j2k', 3:'jls', 4:'jxl'}
    
    logger.info("[+] Step 1: Unpacking")
    with open(bin_path, "rb") as f:
        if f.read(4) != b"STGC": raise ValueError("Invalid Signature")
        h_len = struct.unpack("<I", f.read(4))[0]
        h_data = f.read(h_len)
        
        base_sz = struct.calcsize("<BBBBHH IH H B")
        
        # Removida a variável 'th' do unpack
        ver, cid, s, _, w, h, isz, bs, sc, bpp = struct.unpack("<BBBBHH IH H B", h_data[:base_sz])
        
        lens = []
        curr = base_sz
        for _ in range(sc):
            lens.append(struct.unpack("<I", h_data[curr:curr+4])[0])
            curr += 4
            
        img_data = f.read(isz)
        bmp_data = f.read()
        
    # Log atualizado (sem threshold)
    logger.info(f"  > Header (v{ver}): Codec={c_map.get(cid)}, S={s}, Blocks={bs}, Segments={sc}")

    logger.info("[+] Step 2: Decompressing & Extracting")
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
    
    logger.info("[+] Step 3: Restoring DICOM")
    full_img = reconstruct_from_bit_planes(restored_loc + glo, dtype)
    
    base = os.path.splitext(os.path.basename(bin_path))[0]
    ds = create_clean_dicom_dataset(full_img)
    ds = restore_dicom_metadata(ds, msg_str)
    save_dicom_file(ds, os.path.join(out_dir, f"{base}_restored.dcm"))
    
    print("-" * 60)
    logger.info("DECODING COMPLETE")
    print("="*60 + "\n")
    return full_img, msg_str

if __name__ == "__main__":
    IN_FILE = "images/dx_8b/888.dcm" # Ajuste seu arquivo
    OUT_DIR = "output"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    if os.path.exists(IN_FILE):
        try:
            # percentile=75 significa "Use os 25% mais complexos da imagem"
            # Se precisar de mais espaço, diminua para 50 ou 40.
            bin_file, original_msg = run_encoder(IN_FILE, OUT_DIR, beta=0.2, b_size=4, percentile=75, codec='jls')
            
            if bin_file:
                orig = pydicom.dcmread(IN_FILE).pixel_array
                rec, decoded_msg = run_decoder(bin_file, OUT_DIR)
                
                print("\n" + "="*60)
                logger.info("FINAL VERIFICATION REPORT")
                print("-" * 60)
                
                if np.array_equal(orig, rec):
                    logger.info("  [OK] PIXEL DATA: Images match exactly.")
                else:
                    diff = np.abs(orig.astype(float) - rec.astype(float)).sum()
                    logger.error(f"  [FAIL] PIXEL DATA: Mismatch found (Sum Diff: {diff})")
                
                if original_msg == decoded_msg:
                    logger.info("  [OK] METADATA: Message extracted perfectly.")
                else:
                    logger.error("  [FAIL] METADATA: Message corruption detected.")
                    logger.info(f"    > Orig len: {len(original_msg)}")
                    logger.info(f"    > Rec  len: {len(decoded_msg)}")
                print("="*60 + "\n")
        except Exception as e:
            logger.error(f"CRITICAL ERROR: {e}", exc_info=True)
    else:
        logger.error(f"Input file not found: {IN_FILE}")