from fileinput import filename
import numpy as np
import pandas as pd
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

    # --- Informações Essenciais ---
    # Este é o UID padrão para imagens que não vêm diretamente de um scanner
    # (ex: geradas por software, como as suas).
    SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.7"  # Secondary Capture Image Storage

    # --- Preparar o Dataset ---
    # Usar FileDataset para criar a estrutura de ficheiro correta com preâmbulo
    ds = FileDataset(filename, {}, file_meta=FileMetaDataset(), preamble=b"\x00" * 128)

    # --- File Meta (Grupo 0002) - Informações de Transporte ---
    ds.file_meta.MediaStorageSOPClassUID = SOP_CLASS_UID
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid() # UID único para este ficheiro
    ds.file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    # --- Dataset Principal (Grupos > 0002) - Informações do "Mundo Real" ---
    
    # Adicionar a sigla "DICM" após o preâmbulo
    ds.preamble = b"\x00" * 128
    # ds.is_little_endian e ds.is_implicit_VR são definidos pelo TransferSyntaxUID

    # Metadados de Paciente/Estudo (mínimos para ser válido)
    ds.PatientName = "STEGO^"
    ds.PatientID = "123456"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = SOP_CLASS_UID

    # Adicionar datas e horas formatadas corretamente
    now = datetime.now()
    ds.StudyDate = now.strftime("%Y%m%d")
    ds.StudyTime = now.strftime("%H%M%S")
    ds.SeriesDate = now.strftime("%Y%m%d")
    ds.ContentDate = now.strftime("%Y%m%d")
    ds.ContentTime = now.strftime("%H%M%S")

    ds.Modality = "OT"  # Other
    ds.InstanceNumber = "1"
    ds.SeriesNumber = "1"

    # Informações da Imagem (baseadas no array de entrada)
    ds.Rows, ds.Columns = image_array.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0  # unsigned
    
    bits_allocated = image_array.dtype.itemsize * 8
    ds.BitsAllocated = bits_allocated
    ds.BitsStored = min(bits_stored, bits_allocated)
    ds.HighBit = ds.BitsStored - 1
    
    # Adicionar Window/Level para um bom display inicial
    # Centraliza a janela no meio da gama de intensidade da imagem
    window_center = int((image_array.max() + image_array.min()) / 2)
    window_width = image_array.max() - image_array.min()
    ds.WindowCenter = str(window_center)
    ds.WindowWidth = str(window_width)

    # Dados dos Pixels
    # Garante que o array está no formato correto para o DICOM
    if image_array.dtype == np.uint16:
        arr = image_array.astype(np.uint16)
    else:
        arr = image_array.astype(np.uint8)
    ds.PixelData = arr.tobytes()
    
    return ds

def compress_image(image_array: np.ndarray, codec: str) -> bytes:
    print(f"   - Comprimindo com {codec.upper()}...")
    
    if codec == 'jxl':      
        # Para comprimir para JXL, usamos a ferramenta 'cjxl'
        temp_input_png = 'temp_for_jxl.png'
        temp_output_jxl = 'temp_compressed.jxl'
        try:
            # 1. Salva o array de 16-bit como um PNG temporário, que o cjxl consegue ler
            pil_img = Image.fromarray(image_array.astype(np.uint16))
            pil_img.save(temp_input_png)

            # 2. Constrói e executa o comando 'cjxl' para compressão lossless
            cmd = ['cjxl.exe', temp_input_png, temp_output_jxl, '-d', '0', '-e', '3']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 3. Lê os bytes comprimidos do ficheiro JXL resultante
            with open(temp_output_jxl, 'rb') as f:
                return f.read()
        finally:
            if os.path.exists(temp_input_png): os.remove(temp_input_png)
            if os.path.exists(temp_output_jxl): os.remove(temp_output_jxl)

        
    elif codec in ['j2k', 'jls']:
        # Abordagem com GDCM para JPEG2000 e JPEG-LS (robusta e já funcional)
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
        # Para PNG, usamos o handler interno do pydicom (Deflate/zlib)
        # 1. Cria um dataset DICOM com dados brutos
        ds = create_dicom(image_array)
        
        # 2. Define a sintaxe de transferência para Deflate (compressão tipo PNG/ZIP)
        ds.file_meta.TransferSyntaxUID = DeflatedExplicitVRLittleEndian
        
        # 3. Salva no buffer. O pydicom irá comprimir os dados automaticamente.
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
        temp_file = 'temp_decompress.dcm'
        
        try:
            ds = create_dicom(np.zeros((100, 100), dtype=np.uint8))  # Dummy array
            if codec == 'j2k':
                ds.file_meta.TransferSyntaxUID = JPEG2000Lossless
            else:
                ds.file_meta.TransferSyntaxUID = JPEGLSLossless
            
            ds.PixelData = compressed_bytes
            ds.save_as(temp_file)
            
            ds_compressed = pydicom.dcmread(temp_file)
            return ds_compressed.pixel_array
            
        finally:
            if os.path.exists(temp_file): os.remove(temp_file)
            
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
    # Determina o tipo de dado baseado nos planos disponíveis
    sample_plane = global_planes[0] if global_planes else local_planes[0]
    total_bits = len(global_planes) + len(local_planes)
    
    # Escolhe uint16 se precisar de mais de 8 bits
    dtype = np.uint16 if total_bits > 8 else np.uint8

    # Cria arrays zerados do tipo correto
    global_image = np.zeros(sample_plane.shape, dtype=dtype)
    local_image = np.zeros(sample_plane.shape, dtype=dtype)

    # Combina planos globais (mais significativos)
    for i, plane in enumerate(global_planes):
        shift = i + len(local_planes)
        global_image |= (plane.astype(dtype) << shift)

    # Combina planos locais (menos significativos)
    for i, plane in enumerate(local_planes):
        local_image |= (plane.astype(dtype) << i)

    # Combina as duas partes
    return global_image | local_image

def message_to_bits(message: str) -> list:
    return ''.join(f"{ord(c):08b}" for c in message)

def distribute_message_segments(local_planes, message_bits):
    """Distribui message_bits em segmentos compatíveis com lsb_embed_multi_plane.

    Retorna: (segments, distributed_sizes, segment_indices)
    """
    s = len(local_planes)
    total_bits = len(message_bits)

    # Pesos quadráticos decrescentes (planos menos significativos têm menor índice)
    weights = [(s - i) ** 2 for i in range(s)]
    total_weight = sum(weights)
    distributed_sizes = [max(1, int((w / total_weight) * total_bits)) for w in weights]

    # Ajuste fino para garantir soma == total_bits
    excess = sum(distributed_sizes) - total_bits
    if excess != 0:
        max_idx = distributed_sizes.index(max(distributed_sizes))
        distributed_sizes[max_idx] -= excess

    # Ordem determinística embaralhada (mesma semente usada na versão original)
    segment_indices = list(range(s))
    random.seed(42)
    random.shuffle(segment_indices)

    # Criar segmentos seguindo os tamanhos distribuídos (ordem natural dos destinos)
    segments = []
    bit_idx = 0
    for dest_plane_idx in segment_indices:
        size = distributed_sizes[dest_plane_idx]
        segments.append(message_bits[bit_idx:bit_idx+size])
        bit_idx += size

    return segments, distributed_sizes, segment_indices

def lsb_embed_multi_plane(local_planes, message_bits):
        s = len(local_planes)
        total_bits = len(message_bits)

        # Distribui mensagem em segmentos (padronizado entre métodos)
        segments, distributed_sizes, segment_indices = distribute_message_segments(local_planes, message_bits)

        stego_planes = [None] * s
        bitmaps = [None] * s
        segments_lengths = [0] * s
        total_used = 0

        for orig_segment_idx, dest_plane_idx in enumerate(segment_indices):
            segment = segments[orig_segment_idx]
            plane = local_planes[dest_plane_idx]

            h, w = plane.shape
            stego_plane = plane.copy()
            num_bits = min(len(segment), h * w)

            # Cria bitmap do mesmo tamanho da imagem
            bitmap = np.zeros((h, w), dtype=np.uint8)

            linear_indices = np.arange(num_bits)
            y_coords = linear_indices // w
            x_coords = linear_indices % w
            original_pixels = stego_plane[y_coords, x_coords]
            msg_bits = np.array(list(segment[:num_bits]), dtype=np.uint8)

            # Cria os pixels stego
            stego_pixels = (original_pixels & 0xFE) | msg_bits
            stego_plane[y_coords, x_coords] = stego_pixels

            # BITMAP XOR: Armazena o XOR entre pixel original e pixel stego
            xor_values = original_pixels ^ stego_pixels
            bitmap[y_coords, x_coords] = xor_values

            stego_planes[dest_plane_idx] = stego_plane
            bitmaps[dest_plane_idx] = bitmap
            segments_lengths[dest_plane_idx] = num_bits  # Tamanho real do segmento embutido
            total_used += num_bits

        return stego_planes, bitmaps, total_used, segments_lengths, segment_indices

def lsb_embed_block_adaptive(local_planes, message_bits, block_size=8):
    """
    Esteganografia adaptativa por blocos.

    Estratégia:
    - Divide cada plano local em blocos de size (block_size x block_size).
    - Calcula um score por bloco (variança por default) que indica a "segurança" do bloco
      para modificação (blocos mais ruidosos recebem mais bits).
    - Ordena blocos por score e preenche bits em raster dentro dos blocos até consumir a mensagem.

    Retorna mesma assinatura que lsb_embed_multi_plane: (stego_planes, bitmaps, total_used, segments_lengths, segment_indices)
    """
    s = len(local_planes)
    total_bits = len(message_bits)

    stego_planes = [None] * s
    bitmaps = [None] * s
    segments_lengths = [0] * s
    total_used = 0

    # Obter segmentos padronizados (mesma distribuição do multi_plane)
    segments, distributed_sizes, segment_indices = distribute_message_segments(local_planes, message_bits)

    # Para cada plano local, processa em blocos e embute apenas o segmento correspondente
    for orig_segment_idx, plane_idx in enumerate(segment_indices):
        plane = local_planes[plane_idx]
        h, w = plane.shape
        stego_plane = plane.copy()

        # Prepara blocos
        bh = block_size
        bw = block_size
        blocks = []  # cada entrada: (score, y0, x0, block_h, block_w)
        for y in range(0, h, bh):
            for x in range(0, w, bw):
                y1 = min(y + bh, h)
                x1 = min(x + bw, w)
                block = plane[y:y1, x:x1]
                score = float(np.var(block)) 
                blocks.append((score, y, x, y1 - y, x1 - x))

        # Ordena blocos decrescentemente (mais ruído primeiro)
        blocks.sort(key=lambda x: x[0], reverse=True)

        # Cria bitmap do mesmo tamanho da imagem
        bitmap = np.zeros((h, w), dtype=np.uint8)

        # Obtém segmento dedicado a este plano
        segment = segments[orig_segment_idx]
        num_segment_bits = min(len(segment), h * w)
        seg_bit_idx = 0

        # Preenche blocos em ordem consumindo apenas os bits do segmento
        # Vectorizar: converte segmento para numpy e escreve em fatias planas por bloco
        seg_bits = np.fromiter((int(b) for b in segment[:num_segment_bits]), dtype=np.uint8, count=num_segment_bits)

        for score, y0, x0, bh_real, bw_real in blocks:
            if seg_bit_idx >= num_segment_bits:
                break

            # Obter view do bloco e sua versão plana (1D view)
            block_view = stego_plane[y0:y0+bh_real, x0:x0+bw_real]
            bitmap_view = bitmap[y0:y0+bh_real, x0:x0+bw_real]
            flat_block = block_view.ravel()
            flat_bitmap = bitmap_view.ravel()

            remaining = num_segment_bits - seg_bit_idx
            block_capacity = flat_block.size
            k = remaining if remaining < block_capacity else block_capacity
            if k <= 0:
                continue

            orig_vals = flat_block[:k].astype(np.uint8)
            bits_chunk = seg_bits[seg_bit_idx:seg_bit_idx+k]
            new_vals = (orig_vals & 0xFE) | bits_chunk

            # Escreve de volta usando operações vetorizadas
            flat_block[:k] = new_vals
            flat_bitmap[:k] = orig_vals ^ new_vals

            # Atualiza ponteiro do segmento
            seg_bit_idx += k

        used_bits = seg_bit_idx
        stego_planes[plane_idx] = stego_plane
        bitmaps[plane_idx] = bitmap
        segments_lengths[plane_idx] = used_bits
        total_used += used_bits

    # Como a distribuição por planos não foi embaralhada aqui, segment_indices é identidade
    return stego_planes, bitmaps, total_used, segments_lengths, segment_indices

def lsb_embed_block_then_multiplane(local_planes, message_bits, search_block_size=8, align_across_planes: bool = False):
    """
    Híbrido: encontra o melhor bloco (por variância) de tamanho `search_block_size` e
    inicia a inserção no offset raster desse bloco, mas usa a distribuição/embedding
    do `lsb_embed_multi_plane` (ou seja, escreve bits LSB raster-wise por plano),
    começando do pixel inicial do bloco e envolvendo (wrap) até consumir os segmentos.

    Retorna a mesma assinatura: (stego_planes, bitmaps, total_used, segments_lengths, segment_indices)
    """
    s = len(local_planes)
    total_bits = len(message_bits)

    # Distribuição de segmentos compatível com multi_plane
    segments, segments_lengths, segment_indices = distribute_message_segments(local_planes, message_bits)

    stego_planes = [None] * s
    bitmaps = [None] * s
    total_used = 0

    # Para escolher o melhor ponto inicial, somamos variância local numa janela search_block_size
    # Usamos o primeiro plano como referência de textura (poderia usar média/score entre planos)
    ref_plane = local_planes[0]
    h, w = ref_plane.shape
    sb = search_block_size

    best_score = -1.0
    best_y = 0
    best_x = 0

    for y in range(0, h, sb):
        for x in range(0, w, sb):
            y1 = min(y + sb, h)
            x1 = min(x + sb, w)
            block = ref_plane[y:y1, x:x1]
            score = float(np.var(block))
            if score > best_score:
                best_score = score
                best_y = y
                best_x = x

    # Compute starting linear offset (raster order)
    start_offset = best_y * w + best_x

    # Now embed per-plane like multi_plane but starting at start_offset (wrap around)
    for orig_segment_idx, dest_plane_idx in enumerate(segment_indices):
        segment = segments[orig_segment_idx]
        plane = local_planes[dest_plane_idx]
        h, w = plane.shape
        stego_plane = plane.copy()

        bitmap = np.zeros((h, w), dtype=np.uint8)

        num_bits = min(len(segment), h * w)
        linear_indices = (np.arange(start_offset, start_offset + num_bits) % (h * w))
        y_coords = linear_indices // w
        x_coords = linear_indices % w

        original_pixels = stego_plane[y_coords, x_coords]
        msg_bits = np.fromiter((int(b) for b in segment[:num_bits]), dtype=np.uint8, count=num_bits)

        stego_pixels = (original_pixels & 0xFE) | msg_bits
        stego_plane[y_coords, x_coords] = stego_pixels

        xor_values = original_pixels ^ stego_pixels
        bitmap[y_coords, x_coords] = xor_values

        stego_planes[dest_plane_idx] = stego_plane
        bitmaps[dest_plane_idx] = bitmap
        total_used += num_bits

        # advance start_offset by num_bits so next plane begins after previous embedding
        # if align_across_planes is True we keep the same start_offset for all planes
        if not align_across_planes:
            start_offset = (start_offset + num_bits) % (h * w)

    return stego_planes, bitmaps, total_used, segments_lengths, segment_indices

def calculate_entropy(data_array):
    """
    Calcula a entropia de um array de dados de forma eficiente.
    H(X) = -sum(p(x) * log2(p(x)))
    """
    # Achatamos o array para 1D e contamos a ocorrência de cada valor
    counts = np.bincount(data_array.ravel())
    
    # Calculamos a probabilidade de cada valor
    probabilities = counts[counts > 0] / data_array.size
    
    # Aplicamos a fórmula da entropia
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_mutual_information(bit_plane, image_array):
    """
    Calcula a informação mútua I(X;Y) entre um plano de bit (X) e a imagem (Y).
    Versão otimizada para performance com cache e cálculos eficientes.
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    # Cache para evitar recálculos desnecessários
    if not hasattr(calculate_mutual_information, '_cache'):
        calculate_mutual_information._cache = {}
    
    # Chave do cache baseada no hash dos arrays
    cache_key = (hash(bit_plane.tobytes()), hash(image_array.tobytes()))
    if cache_key in calculate_mutual_information._cache:
        return calculate_mutual_information._cache[cache_key]
    
    # Verificação rápida para variáveis constantes
    if bit_plane.min() == bit_plane.max() or image_array.min() == image_array.max():
        result = 0.0
        calculate_mutual_information._cache[cache_key] = result
        return result

    # Flatten uma vez só para ambos os arrays
    bit_plane_flat = bit_plane.ravel()
    image_array_flat = image_array.ravel()
    
    # H(X) - Entropia do plano de bit (sempre 0 ou 1)
    counts_x = np.bincount(bit_plane_flat, minlength=2)
    probs_x = counts_x[counts_x > 0] / bit_plane.size
    h_x = -np.sum(probs_x * np.log2(probs_x))

    # H(Y) - Entropia da imagem (usar bincount direto é mais rápido)
    if image_array.dtype == np.uint8:
        max_val = 255
    elif image_array.dtype == np.uint16:
        max_val = 65535
    else:
        max_val = int(image_array.max())
    
    counts_y = np.bincount(image_array_flat, minlength=max_val + 1)
    probs_y = counts_y[counts_y > 0] / image_array.size
    h_y = -np.sum(probs_y * np.log2(probs_y))

    bit_plane_int32 = bit_plane_flat.astype(np.int32)
    image_array_int32 = image_array_flat.astype(np.int32)
    combined_indices = bit_plane_int32 * (max_val + 1) + image_array_int32
    joint_counts = np.bincount(combined_indices, minlength=2 * (max_val + 1))
    joint_probs = joint_counts[joint_counts > 0] / image_array.size
    h_xy = -np.sum(joint_probs * np.log2(joint_probs))
    
    # Informação Mútua
    mi = h_x + h_y - h_xy
    
    # Garante que não seja negativo devido a pequenos erros de precisão do float
    result = max(0.0, mi)
    calculate_mutual_information._cache[cache_key] = result
    return result

def adaptive_modalities_decomposition(image_array, beta=0.8, nbits=None):
    """
    Algoritmo 2 (Adaptado): Encontra o ponto de corte 's' usando o cálculo de MI específico.
    Versão otimizada com cache e cálculos eficientes.
    """    
    # Determina o número de bits a partir dos metadados ou do próprio array
    nbits = image_array.dtype.itemsize * 8 if nbits is None else nbits
    print(f"   - Profundidade de bits efetiva: {nbits}")

    # Pré-calcula todos os bit planes uma vez só
    bit_planes = [(image_array >> i) & 1 for i in range(nbits)]
    
    # Calcula a entropia total da imagem apenas uma vez
    total_info = calculate_entropy(image_array)
    target_info = beta * total_info
    
    print(f"   - Informação total da imagem: {total_info:.4f}")
    print(f"   - Meta de retenção ({beta*100}%): {target_info:.4f}")
    
    cumulative_info = 0.0
    s = 1  # Ponto de corte padrão
    
    # Processa os bit planes em ordem (LSB para MSB)
    for i in range(nbits):
        current_plane = bit_planes[i]
        
        # Calcula MI apenas para o plano atual
        mi = calculate_mutual_information(current_plane, image_array)
        cumulative_info += mi
                
        if cumulative_info >= target_info:
            s = i + 1
            break
    
    # Separa os planos (sem recalcular)
    local_planes = bit_planes[:s]   # 's' planos menos significativos
    global_planes = bit_planes[s:]  # O restante dos planos mais significativos
    
    return global_planes, local_planes

def create_header(
    codec: str, 
    s: int, 
    segments_lengths: list,
    segments_indices: list,
    bitmaps_blob_size: int,
    width: int,
    height: int,
    start_offset: int,
    align_across_planes: bool
) -> bytes:
    """
    Cria o cabeçalho binário completo e robusto para o codec.
    """
    # Mapeia o nome do codec para um código de 1 byte
    codec_map = {'png': 1, 'j2k': 2, 'jls': 3, 'jxl': 4}
    codec_id = codec_map.get(codec.lower(), 0)
    
    # Converte o booleano para um inteiro (0 ou 1)
    align_flag = 1 if align_across_planes else 0

    header_format = '>BBBBHHH'
    header_parts = [
        1,  
        codec_id,
        s,
        align_flag,
        width,
        height,
        start_offset
    ]

    # Parte variável: as listas de tamanhos
    header_format += f'{s}H'  # 's' inteiros para os comprimentos dos segmentos
    header_parts.extend(segments_lengths)

    header_format += f'{s}B'  # 's' inteiros para os índices dos segmentos
    header_parts.extend(segments_indices)

    # Adiciona o tamanho do blob de bitmaps (4 bytes)
    header_format += 'I'
    header_parts.append(bitmaps_blob_size)
    
    # Empacota tudo numa string de bytes
    packed_header = struct.pack(header_format, *header_parts)
    
    print(f" HEADER:")
    print(f"   - Versão: {header_parts[0]}")
    print(f"   - Codec ID: {header_parts[1]} ({codec})")
    print(f"   - Número de planos locais (s): {header_parts[2]}")
    print(f"   - Alinhamento entre planos: {'Sim' if align_flag else 'Não'}")
    print(f"   - Dimensões da imagem: {header_parts[4]}x{header_parts[5]}")
    print(f"   - Offset inicial de embedding: {header_parts[6]}")
    print(f"   - Tamanhos dos segmentos: {segments_lengths}")
    print(f"   - Índices dos segmentos: {segments_indices}")
    return packed_header

def create_binary_file(filename, header_bytes, stego_compressed, bitmaps_bytes):
    """
    Salva o arquivo binário simplificado.
    Estrutura: STGC + header + stego_compressed
    """
    with open(filename, "wb") as f:
        f.write(b"STGC")           # Assinatura
        f.write(struct.pack('>I', len(header_bytes)))
        f.write(header_bytes)      # Header principal
        f.write(bitmaps_bytes)     # Bitmaps comprimidos
        f.write(stego_compressed)  # Imagem comprimida

    return os.path.getsize(filename)


"""
-------------------------------------------------
BINARY FILE STRUCTURE (Simplified)
-------------------------------------------------
STGC (4 bytes) - Signature
Header (variable size) - Metadata
  - header_length (2 bytes)
  - version (1 byte)
  - index_s (1 byte)
  - codec_id (1 byte)
  - segments_lengths (index_s * 2 bytes)
  - segment_indices (index_s * 2 bytes)
Stego Image (variable size) - Compressed stego image
-------------------------------------------------
"""

def parse_bin_file(filepath: str):
    """
    Lê um ficheiro .bin, analisa o cabeçalho e separa todos os blocos de dados.
    """
    codec_map = {1: 'png', 2: 'j2k', 3: 'jls', 4: 'jxl'}
    
    with open(filepath, 'rb') as f:
        signature = f.read(4)
        if signature != b'STGC':
            raise ValueError("Arquivo inválido ou com assinatura incorreta.")
        
        header_length_bytes = f.read(4)
        header_length = struct.unpack('>I', header_length_bytes)[0]
        header_data = f.read(header_length)
        
        # Análise do cabeçalho
        base_format = '>BBBBHHH'
        base_size = struct.calcsize(base_format)
        base_parts = struct.unpack(base_format, header_data[:base_size])
        
        version, codec_id, s, align_flag, width, height, start_offset = base_parts
        codec = codec_map.get(codec_id, 'unknown')
        
        # Tamanhos dos segmentos
        seg_lengths_format = f'>{s}H'
        seg_lengths_size = struct.calcsize(seg_lengths_format)
        seg_lengths_start = base_size
        seg_lengths_end = seg_lengths_start + seg_lengths_size
        segments_lengths = list(struct.unpack(seg_lengths_format, header_data[seg_lengths_start:seg_lengths_end]))
        
        # Índices dos segmentos
        seg_indices_format = f'>{s}B'
        seg_indices_size = struct.calcsize(seg_indices_format)
        seg_indices_start = seg_lengths_end
        seg_indices_end = seg_indices_start + seg_indices_size
        segments_indices = list(struct.unpack(seg_indices_format, header_data[seg_indices_start:seg_indices_end]))
        
        # Tamanho do blob de bitmaps
        bitmap_size_format = '>I'
        bitmap_size_start = seg_indices_end
        bitmap_size_end = bitmap_size_start + struct.calcsize(bitmap_size_format)
        bitmaps_blob_size = struct.unpack(bitmap_size_format, header_data[bitmap_size_start:bitmap_size_end])[0]
        
        # Lê o blob de bitmaps
        bitmaps_data = f.read(bitmaps_blob_size)
        
        # Lê o restante como dados da imagem stego comprimida
        stego_image_data = f.read()

        metadata = {
            'version': version,
            'codec': codec,
            's': s,
            'align_flag': align_flag,
            'width': width,
            'height': height,
            'start_offset': start_offset,
            'segments_lengths': segments_lengths,
            'segments_indices': segments_indices
        }

    return metadata, bitmaps_data, stego_image_data

def decode_message(stego_planes, bitmaps, metadata):
    """
    Extrai a mensagem escondida dos planos stego usando os bitmaps.
    """
    s = metadata['s']
    segments = [''] * s
    total_bits = 0
    
    # Percorre os planos na ordem correta
    for i, plane_idx in enumerate(metadata['segments_indices']):
        stego_plane = stego_planes[plane_idx]
        bitmap = bitmaps[plane_idx]
        num_bits = metadata['segments_lengths'][plane_idx]
        
        # Extrai apenas onde o bitmap indica mudanças (1s)
        changes = np.nonzero(bitmap.ravel())[0][:num_bits]
        linear_stego = stego_plane.ravel()
        
        # Extrai os bits LSB onde houve mudanças
        segment_bits = linear_stego[changes] & 1
        segments[plane_idx] = ''.join(str(bit) for bit in segment_bits)
        total_bits += num_bits
    
    # Junta todos os segmentos na ordem correta
    all_bits = ''.join(segments)
    
    # Converte bits em bytes e depois em string
    message_bytes = []
    for i in range(0, len(all_bits), 8):
        byte_bits = all_bits[i:i+8]
        if len(byte_bits) == 8:  # ignora bits incompletos
            byte_val = int(byte_bits, 2)
            message_bytes.append(byte_val)
    
    message = bytes(message_bytes).decode('utf-8', errors='replace')
    return message

def extract_local_planes(stego_array, s):
    """
    Extrai os s planos menos significativos de um array.
    """
    return [(stego_array >> i) & 1 for i in range(s)]

def decode_bin(filepath: str, output_prefix: str = "decoded"):
    """
    Decodifica um arquivo .bin, extraindo a mensagem e recuperando a imagem.

    Args:
        filepath: Caminho do arquivo .bin
        output_prefix: Prefixo para os arquivos de saída
        
    Returns:
        tuple: (mensagem extraída, imagem recuperada)
    """
    print(f"🔄 Decodificando arquivo: {filepath}")
    
    # 1. Ler e analisar o arquivo binário
    metadata, bitmaps, stego_array = parse_bin_file(filepath)
    s = metadata['s']
    codec = metadata['codec']
    print(f"   - Codec detectado: {codec}")
    print(f"   - Planos locais (s): {s}")

    # 2. Descomprimir os dados
    stego_decompressed = decompress_image(stego_array, codec)
    stego_array = stego_decompressed

    # 3. Descomprimir os bitmaps usando zlib e transformar em array
    bitmaps_array = np.frombuffer(zlib.decompress(bitmaps), dtype=np.uint8)
    bitmaps = np.split(bitmaps_array, s)

    # 4. Extrair os planos locais da imagem stego
    local_planes = extract_local_planes(stego_array, s)

    # 5. Extrair a mensagem usando os bitmaps
    print("🔄 Extraindo mensagem...")
    message = decode_message(local_planes, bitmaps, metadata)

    # 6. Salvar os resultados
    message_file = f"{output_prefix}_mensagem.txt"
    with open(message_file, 'w', encoding='utf-8') as f:
        f.write(message)
    print(f"✅ Mensagem salva em: {message_file}")

    # 7. Criar e salvar o DICOM
    print("🔄 Criando arquivo DICOM...")
    ds = create_dicom(stego_array)
    dicom_file = f"{output_prefix}_imagem.dcm"
    save_dicom(ds, dicom_file)
    
    return message, stego_array




def main():
    # Exemplo simples de uso do código
    # 1. Carregar uma imagem DICOM
    input_dicom_file = "images/pe.dcm"  # Substitua pelo caminho do seu arquivo DICOM
    if not os.path.exists(input_dicom_file):
        print(f"❌ Arquivo {input_dicom_file} não encontrado.")
        print("Por favor, forneça o caminho correto para um arquivo DICOM de entrada.")
        return

    try:
        # Carrega a imagem DICOM
        print("🔄 Carregando imagem DICOM...")
        dicom_data = load_dicom_image(input_dicom_file)
        image_array = dicom_data.pixel_array

        # 2. Preparar uma mensagem de exemplo
        message = "Mensagem de teste para esteganografia!"
        message_bits = message_to_bits(message)

        # 3. Decompor a imagem em planos
        print("🔄 Decompondo imagem em planos...")
        global_planes, local_planes = adaptive_modalities_decomposition(image_array, beta=0.4)
        s = len(local_planes)
        print(f"   - Número de planos locais (s): {s}")

        # 4. Embutir a mensagem usando o método híbrido
        print("🔄 Embutindo mensagem usando método híbrido...")
        stego_planes, bitmaps, total_used, segments_lengths, segment_indices = lsb_embed_block_then_multiplane(
            local_planes, message_bits, search_block_size=16
        )

        # 5. Combinar os planos novamente
        print("🔄 Reconstruindo imagem...")
        stego_image = merge_modalities(global_planes, stego_planes)

        # 6. Comprimir a imagem usando um codec
        codec = 'jxl'  # pode ser 'png', 'j2k', 'jls' ou 'jxl'
        compressed_bytes = compress_image(stego_image, codec)

        # 7. Comprimindo os bitmaps
        print("🔄 Comprimindo bitmaps...")
        all_bitmaps_array = np.stack(bitmaps, axis=0)
        bitmaps_blob = zlib.compress(all_bitmaps_array.tobytes())
        bitmaps_blob_size = len(bitmaps_blob)

        # 7. Criar cabeçalho e arquivo binário
        print("🔄 Criando arquivo binário...")
        height, width = stego_image.shape
        header = create_header(
            codec=codec,
            s=s,
            segments_lengths=segments_lengths,
            segments_indices=segment_indices,
            bitmaps_blob_size=bitmaps_blob_size,
            width=width,
            height=height,
            start_offset=0,
            align_across_planes=False
        )

        output_file = "output/saida_exemplo.bin"
        file_size = create_binary_file(output_file, header, compressed_bytes, bitmaps_blob)
        print(f"✅ Arquivo gerado com sucesso: {output_file} ({file_size} bytes)")

    except Exception as e:
        print(f"❌ Erro durante a execução: {str(e)}")
        raise

    try:
        # 8. Decodificar o arquivo gerado
        decoded_message, recovered_image = decode_bin(output_file, output_prefix="output/decoded")
        print(f"✅ Mensagem decodificada: {decoded_message}")

    except Exception as e:
        print(f"❌ Erro durante a decodificação: {str(e)}")
        raise

# Exemplo de uso do decode
if __name__ == "__main__":
    main()

