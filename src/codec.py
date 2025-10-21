from fileinput import filename
import numpy as np
import pydicom
import os, io
import random
import struct
from datetime import datetime
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, JPEGLSLossless, JPEG2000Lossless, DeflatedExplicitVRLittleEndian
from pydicom.encaps import encapsulate
import zlib
import pydicom.config
from pydicom.pixel_data_handlers import pylibjpeg_handler
pydicom.config.image_handlers = [pylibjpeg_handler]
import subprocess

def save_dicom(ds: FileDataset, file_path: str):
    ds.save_as(file_path, write_like_original=False)
    print(f"Arquivo DICOM salvo em: {file_path}")

def create_dicom(image_array: np.ndarray) -> FileDataset:
    """
    Cria um Dataset DICOM simples com dados de imagem NÃO COMPRIMIDOS.
    """
    max_val = image_array.max()
    
    # Calcula bits necessários para representar o valor máximo
    log_val = np.log2(float(max_val) + 1.0)
    bits_stored = int(np.ceil(log_val))
    bits_stored = max(1, bits_stored)  # Garante pelo menos 1 bit

    if image_array.ndim != 2: raise ValueError("A imagem deve ser 2D (grayscale).")

    if image_array.dtype not in [np.uint8, np.uint16]:
        raise ValueError("A imagem deve ser uint8 ou uint16.")

    SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage

    ds = FileDataset(None, {}, file_meta=FileMetaDataset(), preamble=b"\x00" * 128)

    ds.file_meta.MediaStorageSOPClassUID = SOP_CLASS_UID
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.PatientName = "STEGO^"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
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
    
    window_center = int((image_array.max() + image_array.min()) / 2)
    window_width = int(image_array.max() - image_array.min())
    ds.WindowCenter = str(window_center)
    ds.WindowWidth = str(window_width)

    ds.PixelData = image_array.tobytes()
    
    return ds

def compress_image(image_array: np.ndarray, codec: str) -> bytes:
    print(f"   - Comprimindo com {codec.upper()}...")
    
    if codec == 'jxl':      
        temp_input_png = 'temp_for_jxl.png'
        temp_output_jxl = 'temp_compressed.jxl'
        try:
            pil_img = Image.fromarray(image_array)
            pil_img.save(temp_input_png)
            cmd = ['cjxl.exe', temp_input_png, temp_output_jxl, '-d', '0', '-e', '9']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(temp_output_jxl, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(temp_input_png): os.remove(temp_input_png)
            if os.path.exists(temp_output_jxl): os.remove(temp_output_jxl)

    elif codec in ['j2k', 'jls']:
        temp_uncompressed = 'temp_uncompressed.dcm'
        temp_compressed = 'temp_compressed.dcm'
        try:
            ds_uncompressed = create_dicom(image_array)
            ds_uncompressed.save_as(temp_uncompressed)
            if codec == 'j2k':
                cmd = ['gdcmconv', '--j2k', temp_uncompressed, temp_compressed]
            else:
                cmd = ['gdcmconv', '--jpegls', temp_uncompressed, temp_compressed]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with open(temp_compressed, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(temp_uncompressed): os.remove(temp_uncompressed)
            if os.path.exists(temp_compressed): os.remove(temp_compressed)
            
    elif codec == 'png':
        ds = create_dicom(image_array)
        ds.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
        buffer = io.BytesIO()
        ds.save_as(buffer)
        return buffer.getvalue()

    else:
        raise ValueError(f"Codec '{codec}' não suportado.")

def decompress_image(compressed_bytes: bytes, codec: str) -> np.ndarray:
    """Descomprime bytes de imagem com base no codec especificado ('jxl', 'j2k', 'jls')."""
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

    elif codec in ['j2k', 'jls']:
        ds = pydicom.dcmread(io.BytesIO(compressed_bytes), force=True)
        return ds.pixel_array
            
    elif codec == 'png':
        buffer = io.BytesIO(compressed_bytes)
        ds = pydicom.dcmread(buffer, force=True)
        return ds.pixel_array
        
    else:
        raise ValueError(f"Codec '{codec}' não suportado.")

def load_dicom_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    return dicom_data

def merge_modalities(global_planes: np.ndarray, local_planes: np.ndarray) -> np.ndarray:
    if not local_planes:
        raise ValueError("A lista de planos locais não pode estar vazia.")
    sample_plane = local_planes[0]
    
    total_bits = len(global_planes) + len(local_planes)
    dtype = np.uint16 if total_bits > 8 else np.uint8

    global_image = np.zeros(sample_plane.shape, dtype=dtype)
    local_image = np.zeros(sample_plane.shape, dtype=dtype)

    for i, plane in enumerate(global_planes):
        shift = i + len(local_planes)
        global_image |= (plane.astype(dtype) << shift)

    for i, plane in enumerate(local_planes):
        local_image |= (plane.astype(dtype) << i)

    return global_image | local_image

def message_to_bits(message: str) -> str:
    encoded_bytes = message.encode('utf-8')
    return ''.join(f"{byte:08b}" for byte in encoded_bytes)

def distribute_message_segments(local_planes, message_bits):
    """
    Distribui a mensagem em segmentos, um para cada plano, e retorna a ordem de inserção.
    """
    s = len(local_planes)
    total_bits = len(message_bits)
    weights = [(s - i) ** 2 for i in range(s)]
    total_weight = sum(weights)
    
    if total_weight == 0:
        distributed_sizes = [total_bits // s] * s
    else:
        distributed_sizes = [max(1, int((w / total_weight) * total_bits)) for w in weights]

    excess = sum(distributed_sizes) - total_bits
    if excess != 0:
        for i in range(abs(excess)):
            distributed_sizes[-(i + 1)] -= np.sign(excess)
    
    final_adjustment = total_bits - sum(distributed_sizes)
    distributed_sizes[0] += final_adjustment
    
    all_segments = {}
    bit_idx = 0
    for i in range(s):
        size = distributed_sizes[i]
        all_segments[i] = message_bits[bit_idx:bit_idx+size]
        bit_idx += size
        
    segment_indices = list(range(s))
    random.seed(42)
    random.shuffle(segment_indices)
    
    processing_segments = [all_segments[idx] for idx in segment_indices]
    final_lengths = [len(all_segments[i]) for i in range(s)]

    return processing_segments, final_lengths, segment_indices

def lsb_embed_sequential(local_planes, message_bits, allowed_indices, image_shape, start_offset=0, align_across_planes: bool = False):
    """
    MODIFICADO: Insere a mensagem de forma sequencial usando apenas os pixels permitidos pelo mapa de capacidade.
    Cria e retorna um único bitmap combinado (com OR) e o total de bits usados.
    """
    s = len(local_planes)
    h, w = image_shape

    # Verificar se a capacidade é suficiente
    total_capacity = len(allowed_indices)
    if len(message_bits) > total_capacity:
        raise ValueError(f"Mensagem ({len(message_bits)} bits) é maior que a capacidade da imagem ({total_capacity} bits) com o limiar atual.")

    segments, segments_lengths, segment_indices = distribute_message_segments(local_planes, message_bits)

    stego_planes = [p.copy() for p in local_planes]
    combined_bitmap = np.zeros((h, w), dtype=np.uint8)
    total_used = 0
    current_start_offset_in_allowed = start_offset # Este offset agora é dentro da lista de índices permitidos

    for i, dest_idx in enumerate(segment_indices):
        segment = segments[i] 
        stego_plane = stego_planes[dest_idx]
        num_bits = segments_lengths[dest_idx]

        if num_bits == 0:
            continue

        # Seleciona os índices lineares da lista de pixels permitidos
        indices_to_use = allowed_indices[current_start_offset_in_allowed : current_start_offset_in_allowed + num_bits]
        
        y_coords, x_coords = np.unravel_index(indices_to_use, (h, w))

        original_pixels = stego_plane[y_coords, x_coords]
        msg_bits_arr = np.fromiter(segment, dtype=np.uint8, count=num_bits)

        stego_pixels = (original_pixels & 0xFE) | msg_bits_arr
        stego_plane[y_coords, x_coords] = stego_pixels

        xor_values = original_pixels ^ stego_pixels
        combined_bitmap[y_coords, x_coords] |= xor_values
        total_used += num_bits

        if not align_across_planes:
            current_start_offset_in_allowed = (current_start_offset_in_allowed + num_bits)

    return stego_planes, combined_bitmap, total_used, segments_lengths, segment_indices

def calculate_entropy(data_array):
    counts = np.bincount(data_array.ravel())
    probabilities = counts[counts > 0] / data_array.size
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_mutual_information(bit_plane, image_array):
    if not hasattr(calculate_mutual_information, '_cache'):
        calculate_mutual_information._cache = {}
    cache_key = (hash(bit_plane.tobytes()), hash(image_array.tobytes()))
    if cache_key in calculate_mutual_information._cache:
        return calculate_mutual_information._cache[cache_key]
    
    if bit_plane.min() == bit_plane.max() or image_array.min() == image_array.max():
        result = 0.0
        calculate_mutual_information._cache[cache_key] = result
        return result

    bit_plane_flat = bit_plane.ravel()
    image_array_flat = image_array.ravel()
    
    counts_x = np.bincount(bit_plane_flat, minlength=2)
    probs_x = counts_x[counts_x > 0] / bit_plane.size
    h_x = -np.sum(probs_x * np.log2(probs_x))

    max_val = int(image_array.max())
    counts_y = np.bincount(image_array_flat, minlength=max_val + 1)
    probs_y = counts_y[counts_y > 0] / image_array.size
    h_y = -np.sum(probs_y * np.log2(probs_y))

    combined_indices = bit_plane_flat.astype(np.int32) * (max_val + 1) + image_array_flat.astype(np.int32)
    joint_counts = np.bincount(combined_indices)
    joint_probs = joint_counts[joint_counts > 0] / image_array.size
    h_xy = -np.sum(joint_probs * np.log2(joint_probs))
    
    mi = h_x + h_y - h_xy
    result = max(0.0, mi)
    calculate_mutual_information._cache[cache_key] = result
    return result

def adaptive_modalities_decomposition(image_array, beta=0.8, nbits=None):
    nbits = image_array.dtype.itemsize * 8 if nbits is None else nbits
    print(f"   - Profundidade de bits efetiva: {nbits}")
    bit_planes = [(image_array >> i) & 1 for i in range(nbits)]
    total_info = calculate_entropy(image_array)
    target_info = beta * total_info
    print(f"   - Informação total da imagem: {total_info:.4f}")
    print(f"   - Meta de retenção ({beta*100}%): {target_info:.4f}")
    
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

# MODIFICADO: para incluir block_size e threshold_factor
def create_header(codec: str, s: int, segments_lengths: list, segments_indices: list, bitmaps_blob_size: int, 
                  width: int, height: int, start_offset: int, align_across_planes: bool,
                  block_size: int, threshold_factor: float) -> bytes:
    codec_map = {'png': 1, 'j2k': 2, 'jls': 3, 'jxl': 4}
    codec_id = codec_map.get(codec.lower(), 0)
    align_flag = 1 if align_across_planes else 0

    # Formato: Versão, ID Codec, S, Align Flag, Width, Height, Start Offset, Block Size, Threshold Factor
    header_format = '>BBBBHHH B f' # Adicionado B para block_size (uint8) e f para threshold_factor (float 4 bytes)
    header_parts = [1, codec_id, s, align_flag, width, height, start_offset, block_size, threshold_factor]

    header_format += f'{s}H'
    header_parts.extend(segments_lengths)

    header_format += f'{s}B'
    header_parts.extend(segments_indices)

    header_format += 'I'
    header_parts.append(bitmaps_blob_size)
    
    packed_header = struct.pack(header_format, *header_parts)
    
    final_header = struct.pack('>I', len(packed_header)) + packed_header
    
    print(" HEADER CRIADO:")
    print(f"   - ... (outros campos)")
    print(f"   - Block Size: {block_size}")
    print(f"   - Threshold Factor: {threshold_factor}")
    return final_header

# MODIFICADO: para ler block_size e threshold_factor
def parse_bin_file(filepath: str):
    codec_map = {1: 'png', 2: 'j2k', 3: 'jls', 4: 'jxl'}
    with open(filepath, 'rb') as f:
        if f.read(4) != b'STGC':
            raise ValueError("Arquivo inválido ou com assinatura incorreta.")
        
        header_length = struct.unpack('>I', f.read(4))[0]
        header_data = f.read(header_length)
        
        base_format = '>BBBBHHH B f' # Adicionado B e f
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
            'block_size': block_size, 'threshold_factor': threshold_factor # Adicionado ao metadata
        }
    return metadata, bitmaps_data, stego_image_data

def create_binary_file(filename, header_bytes, bitmap_bytes, stego_compressed):
    with open(filename, "wb") as f:
        f.write(b"STGC")
        f.write(header_bytes)
        f.write(bitmap_bytes)
        f.write(stego_compressed)
    return os.path.getsize(filename)

# MODIFICADO: para usar a lista de allowed_indices
def extract_message_and_restore_planes(stego_planes, combined_bitmap, metadata, allowed_indices):
    """
    MODIFICADO: Extrai a mensagem e restaura os planos usando a lista de pixels permitidos.
    """
    s = metadata['s']
    h = metadata['height']
    w = metadata['width']
    align_across_planes = metadata.get('align_flag', 0)
    current_start_offset_in_allowed = metadata.get('start_offset', 0)
    
    message_segments = [''] * s
    restored_planes = [p.copy() for p in stego_planes]
    
    segment_indices = metadata['segments_indices']
    segments_lengths = metadata['segments_lengths']

    for i, dest_plane_idx in enumerate(segment_indices):
        stego_plane = stego_planes[dest_plane_idx]
        plane_to_restore = restored_planes[dest_plane_idx]
        num_bits = segments_lengths[dest_plane_idx]
        
        if num_bits == 0:
            continue

        # CORREÇÃO PRINCIPAL: Usa a lista de allowed_indices em vez de gerar índices sequenciais
        indices_to_use = allowed_indices[current_start_offset_in_allowed : current_start_offset_in_allowed + num_bits]
        y_coords, x_coords = np.unravel_index(indices_to_use, (h, w))
        
        # 1. Extração da Mensagem
        extracted_bits = stego_plane[y_coords, x_coords] & 1
        message_segments[dest_plane_idx] = ''.join(map(str, extracted_bits))
        
        # 2. Restauração do Plano
        xor_diff = combined_bitmap[y_coords, x_coords]
        original_lsb = extracted_bits ^ xor_diff
        plane_to_restore[y_coords, x_coords] = (plane_to_restore[y_coords, x_coords] & 0xFE) | original_lsb
        
        if not align_across_planes:
            current_start_offset_in_allowed += num_bits

    all_bits = ''.join(message_segments)
    message_bytes = bytearray()
    for i in range(0, len(all_bits), 8):
        byte_bits = all_bits[i:i+8]
        if len(byte_bits) == 8:
            byte_val = int(byte_bits, 2)
            message_bytes.append(byte_val)
    
    message = message_bytes.decode('utf-8', errors='replace')
    
    return message, restored_planes

def extract_local_planes(stego_array, s):
    return [(stego_array >> i) & 1 for i in range(s)]

# MODIFICADO: para regenerar o mapa de capacidade
def decode_bin(filepath: str, output_prefix: str = "decoded"):
    """
    Decodifica um arquivo .bin, extraindo a mensagem e recuperando a imagem original.
    """
    print(f"🔄 Decodificando arquivo: {filepath}")
    
    metadata, bitmaps_blob, stego_image_data = parse_bin_file(filepath)
    s = metadata['s']
    codec = metadata['codec']
    print(f"   - Codec detectado: {codec}")
    print(f"   - Planos locais (s): {s}")

    stego_array = decompress_image(stego_image_data, codec)

    # --- NOVA ETAPA NA DECODIFICAÇÃO ---
    print("🔄 Regenerando mapa de capacidade a partir da imagem esteganografada...")
    block_size = metadata['block_size']
    threshold_factor = metadata['threshold_factor']
    print(f"   - Usando block_size={block_size} e threshold_factor={threshold_factor:.2f}")
    _ , allowed_indices = create_capacity_map_and_indices(stego_array, block_size=block_size, threshold_factor=threshold_factor)
    # ------------------------------------

    w = metadata['width']
    h = metadata['height']
    bitmap_bytes = zlib.decompress(bitmaps_blob)
    combined_bitmap = np.frombuffer(bitmap_bytes, dtype=np.uint8).reshape((h, w))

    nbits = stego_array.dtype.itemsize * 8
    all_stego_planes = [(stego_array >> i) & 1 for i in range(nbits)]
    stego_local_planes = all_stego_planes[:s]
    global_planes = all_stego_planes[s:]

    print("🔄 Extraindo mensagem e restaurando planos...")
    # Passa os allowed_indices regenerados para a função de extração
    message, restored_local_planes = extract_message_and_restore_planes(
        stego_local_planes, combined_bitmap, metadata, allowed_indices
    )

    message_file = f"{output_prefix}_mensagem.txt"
    with open(message_file, 'w', encoding='utf-8') as f:
        f.write(message)
    print(f"✅ Mensagem salva em: {message_file}")
    
    print("🔄 Reconstruindo imagem original...")
    restored_image_array = merge_modalities(global_planes, restored_local_planes)

    print("🔄 Criando arquivo DICOM da imagem original...")
    ds = create_dicom(restored_image_array)
    dicom_file = f"{output_prefix}_imagem.dcm"
    save_dicom(ds, dicom_file)
    
    return message, ds

def create_capacity_map_and_indices(image_array: np.ndarray, block_size: int = 8, threshold_factor: float = 1.0):
    """
    Analisa a imagem para encontrar regiões complexas adequadas para embedding.

    Args:
        image_array: A imagem de entrada.
        block_size: O tamanho dos blocos para análise (ex: 4, 8).
        threshold_factor: Fator multiplicativo para o limiar. < 1.0 para mais capacidade, > 1.0 para mais segurança.

    Returns:
        Uma tupla contendo:
        - capacity_map (np.ndarray): Um mapa 2D onde 1 indica um pixel utilizável.
        - allowed_indices (np.ndarray): Um array 1D com os índices lineares dos pixels utilizáveis.
    """
    print(f"   - Criando mapa de capacidade com blocos {block_size}x{block_size} (otimizado)...")
    h, w = image_array.shape

    # 1. Pad the image to be divisible by block_size
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    padded_image = np.pad(image_array, ((0, h_pad), (0, w_pad)), 'edge')
    padded_h, padded_w = padded_image.shape
    
    # 2. Create a view of the image as blocks (no loops)
    # Shape: (num_blocks_h, num_blocks_w, block_size, block_size)
    blocks = padded_image.reshape(padded_h // block_size, block_size, padded_w // block_size, block_size).transpose(0, 2, 1, 3)

    # 3. Calculate standard deviation for all blocks at once
    block_stds = np.std(blocks, axis=(2, 3))

    # 4. Calculate adaptive threshold
    # Ignore blocks with zero variance (completely flat) for the mean calculation
    non_zero_stds = block_stds[block_stds > 0]
    
    if non_zero_stds.size == 0:
        raise ValueError("A imagem é completamente plana, não é possível realizar esteganografia adaptativa.")

    adaptive_threshold = np.mean(non_zero_stds) * threshold_factor
    print(f"   - Limiar de desvio padrão adaptativo calculado: {adaptive_threshold:.4f}")

    # 5. Identify complex blocks
    complex_blocks_map = block_stds > adaptive_threshold

    # 6. Build the full capacity map by scaling up the block map
    # np.kron expands each element of complex_blocks_map into a block_size x block_size block
    capacity_map_padded = np.kron(complex_blocks_map, np.ones((block_size, block_size), dtype=np.uint8))

    # 7. Un-pad the capacity map to match original image dimensions
    capacity_map = capacity_map_padded[:h, :w]

    # 8. Get the linear indices of the allowed pixels
    allowed_indices = np.where(capacity_map.ravel() == 1)[0]
    
    total_capacity = len(allowed_indices)
    print(f"   - Capacidade total encontrada: {total_capacity} bits ({total_capacity / (h*w) * 100:.2f}% da imagem)")
    
    return capacity_map, allowed_indices

# Em codec.py, substitua a função main() por estas duas:

def run_steganography_process(input_dicom_file, message, output_dir, base_filename, beta, block_size, threshold_factor, codec='jxl', align_across_planes=False, start_offset=0):
    """
    Encapsula toda a lógica de esteganografia em uma função parametrizável.
    Retorna o tamanho do arquivo binário gerado e o caminho do DICOM recuperado.
    """
    print(f"Executando com Beta={beta}, BlockSize={block_size}, Threshold={threshold_factor}")
    
    # Carrega a imagem
    image_array = load_dicom_image(input_dicom_file).pixel_array
    message_bits = message_to_bits(message)

    # Decomposição
    global_planes, local_planes = adaptive_modalities_decomposition(image_array, beta=beta)
    s = len(local_planes)

    # Análise adaptativa
    _, allowed_indices = create_capacity_map_and_indices(image_array, block_size=block_size, threshold_factor=threshold_factor)

    if len(message_bits) > len(allowed_indices):
        print(f"⚠️  AVISO: Mensagem ({len(message_bits)} bits) muito grande para a capacidade ({len(allowed_indices)} bits). Pulando esta combinação.")
        return None, None

    # Embedding
    stego_planes, combined_bitmap, _, segments_lengths, segment_indices = lsb_embed_sequential(
        local_planes, message_bits, allowed_indices, image_array.shape, 
        start_offset=start_offset, align_across_planes=align_across_planes
    )
    
    # Reconstrução e Compressão
    stego_image = merge_modalities(global_planes, stego_planes)
    compressed_bytes = compress_image(stego_image, codec)
    bitmaps_blob = zlib.compress(combined_bitmap.tobytes())
    bitmaps_blob_size = len(bitmaps_blob)
    
    # Criação do arquivo binário
    height, width = stego_image.shape
    header = create_header(
        codec=codec, s=s, segments_lengths=segments_lengths,
        segments_indices=segment_indices, bitmaps_blob_size=bitmaps_blob_size,
        width=width, height=height, start_offset=start_offset,
        align_across_planes=align_across_planes,
        block_size=block_size, threshold_factor=threshold_factor
    )
    
    output_bin_file = os.path.join(output_dir, f"{base_filename}.bin")
    file_size = create_binary_file(output_bin_file, header, bitmaps_blob, compressed_bytes)
    
    # Decodifica para obter a imagem stego para análise MSE
    output_prefix = os.path.join(output_dir, f"{base_filename}_decoded")
    _, recovered_dicom_ds = decode_bin(output_bin_file, output_prefix=output_prefix)
    recovered_dicom_path = f"{output_prefix}_imagem.dcm"
    
    print(f"✅ Geração concluída: {output_bin_file} ({file_size} bytes)")
    return file_size, recovered_dicom_path


def main():
    """
    Função principal original para uso autônomo do script.
    """
    input_dicom_file = "images/torax.dcm"
    if not os.path.exists(input_dicom_file):
        print(f"❌ Arquivo {input_dicom_file} não encontrado.")
        return

    try:
        message = "Mensagem de teste para esteganografia! v3 para teste final."
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Parâmetros padrão para execução autônoma
        beta = 0.4
        block_size = 8
        threshold_factor = 0.8
        base_filename = f"saida_beta{beta}_bs{block_size}_tf{threshold_factor}"

        run_steganography_process(
            input_dicom_file, message, output_dir, base_filename,
            beta, block_size, threshold_factor
        )

    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        raise

# Não se esqueça de manter o if __name__ == "__main__": main() no final do codec.py

if __name__ == "__main__":
    main()

