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
import pillow_jxl

def save_dicom(ds: FileDataset, file_path: str):
    ds.save_as(file_path, write_like_original=False)
    print(f"Arquivo DICOM salvo em: {file_path}")

def create_dicom(image_array: np.ndarray) -> FileDataset:
    """
    Cria um Dataset DICOM simples com dados de imagem NÃO COMPRIMIDOS.
    """
    max_val = image_array.max()
    min_val = image_array.min()
    
    print(f"   - Debug: max_val={max_val}, min_val={min_val}, dtype={image_array.dtype}")

    # Calcula bits necessários para representar o valor máximo
    log_val = np.log2(float(max_val) + 1.0)
    bits_stored = int(np.ceil(log_val))
    bits_stored = max(1, bits_stored)  # Garante pelo menos 1 bit

    print(f"   - Bits Stored calculado: {bits_stored} (max pixel value: {max_val})")

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
    """Comprime um array de imagem usando o codec especificado ('png', 'jpeg2000', 'jpegls')."""
    print(f"   - Comprimindo com {codec.upper()}...")
    
    if codec in ['j2k', 'jls']:
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

    elif codec == 'jxl':      
        temp_input_png = 'temp_for_jxl.png'
        temp_output_jxl = 'temp_compressed.jxl'
        try:
            if image_array.dtype == np.uint16:
                pil_img = Image.fromarray(image_array.astype(np.uint16))
            else:
                pil_img = Image.fromarray(image_array)
            pil_img.save(temp_input_png)

            cmd = ['cjxl.exe', temp_input_png, temp_output_jxl, '-d', '0', '-e', '9']
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            with open(temp_output_jxl, 'rb') as f:
                return f.read()
        finally:
            # 4. Limpa os ficheiros temporários
            if os.path.exists(temp_input_png): os.remove(temp_input_png)
            if os.path.exists(temp_output_jxl): os.remove(temp_output_jxl)
        
    else:
        raise ValueError(f"Codec '{codec}' não suportado.")

def decompress_image(compressed_bytes: bytes, codec: str) -> np.ndarray:
    """Descomprime bytes de imagem com base no codec especificado ('jxl', 'j2k', 'jls')."""
    if codec == 'jxl':
        temp_input_jxl = 'temp_compressed.jxl'
        temp_output_png = 'temp_decompressed.png'
        
        try:
            with open(temp_input_jxl, 'wb') as f:
                f.write(compressed_bytes)
                
            cmd = ['djxl.exe', temp_input_jxl, temp_output_png]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            img = Image.open(temp_output_png)
            return np.array(img)
            
        finally:
            # Limpa arquivos temporários
            if os.path.exists(temp_input_jxl): os.remove(temp_input_jxl)
            if os.path.exists(temp_output_png): os.remove(temp_output_png)
            
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
        img = Image.open(buffer)
        return np.array(img)
        
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

def lsb_embed_multi_plane(local_planes, message_bits):
        s = len(local_planes)
        total_bits = len(message_bits)

        # Distribui os bits usando pesos (planos mais significativos recebem mais bits)
        weights = [(s - i) ** 2 for i in range(s)]
        total_weight = sum(weights)
        distributed_sizes = [max(1, int((w / total_weight) * total_bits)) for w in weights]
        
        # Ajusta para garantir que a soma seja exatamente total_bits
        excess = sum(distributed_sizes) - total_bits
        if excess > 0:
            max_idx = distributed_sizes.index(max(distributed_sizes))
            distributed_sizes[max_idx] -= excess
        elif excess < 0:
            max_idx = distributed_sizes.index(max(distributed_sizes))
            distributed_sizes[max_idx] -= excess  # excess é negativo, então subtrai um negativo = soma

        # Cria segmentos com os tamanhos redistribuídos
        segment_indices = list(range(s))
        random.seed(42)
        random.shuffle(segment_indices)
        
        bit_idx = 0
        segments = []
        
        # Cria os segmentos na ordem correta dos tamanhos distribuídos
        for dest_plane_idx in segment_indices:
            size = distributed_sizes[dest_plane_idx]
            segments.append(message_bits[bit_idx:bit_idx+size])
            bit_idx += size

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

            # Calcula quantas linhas são realmente necessárias
            lines_needed = (num_bits + w - 1) // w  # Ceiling division
            
            # Cria bitmap compacto apenas com as linhas necessárias
            bitmap_compact = np.zeros((lines_needed, w), dtype=np.uint8)

            linear_indices = np.arange(num_bits)
            y_coords = linear_indices // w
            x_coords = linear_indices % w
            original_pixels = stego_plane[y_coords, x_coords]
            msg_bits = np.array(list(segment[:num_bits]), dtype=np.uint8)
            
            # Cria os pixels stego
            stego_pixels = (original_pixels & 0xFE) | msg_bits
            stego_plane[y_coords, x_coords] = stego_pixels

            # BITMAP XOR: Armazena o XOR entre pixel original e pixel stego
            # Para recuperação perfeita: original_pixel = stego_pixel XOR bitmap_value
            xor_values = original_pixels ^ stego_pixels
            bitmap_compact[y_coords, x_coords] = xor_values
            
            stego_planes[dest_plane_idx] = stego_plane
            bitmaps[dest_plane_idx] = bitmap_compact
            segments_lengths[dest_plane_idx] = len(segment)  # Tamanho real do segmento embarcado
            total_used += num_bits

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
    
    return s, global_planes, local_planes

def create_header(segments_lengths, segment_indices,):
    index_s = len(segments_lengths)
    # Formato: header_length, version, index_s, [segments_lengths], [segment_indices]
    header_format = f">BBB{index_s}H{index_s}H"
    header_length = struct.calcsize(header_format)

    header_parts = [
        header_length,
        1,      # Version
        index_s,
    ]
    header_parts.extend(segments_lengths)
    header_parts.extend(segment_indices)

    print(f"Header: {header_parts}")

    header_bytes = struct.pack(header_format, *header_parts)
    return header_bytes

DEBUG_BITMAPS = []

def create_binary_file(filename, header_bytes, stego_compressed, bitmaps_list):
    """
    Salva o arquivo binário simplificado.
    Estrutura: STGC + header + [bitmaps_compressed] + stego_compressed
    """
    # Comprime cada bitmap individualmente e concatena
    bitmaps_compressed = b''
    for i, bitmap in enumerate(bitmaps_list):
        bitmap_bytes = bitmap.tobytes()
        compressed = zlib.compress(bitmap_bytes)
        bitmaps_compressed += compressed
        print(f"   - Bitmap {i}: original {len(bitmap_bytes)} bytes, comprimido {len(compressed)} bytes")

    DEBUG_BITMAPS.append(bitmaps_list)

    with open(filename, "wb") as f:
        f.write(b"STGC")                          # Assinatura
        f.write(header_bytes)                     # Header principal
        f.write(bitmaps_compressed)               # Bitmaps comprimidos
        f.write(stego_compressed)                 # Imagem comprimida

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
  - segments_lengths (index_s * 2 bytes)
  - segment_indices (index_s * 2 bytes)
Bitmaps (variable size) - Compressed bitmaps
Stego Image (variable size) - Compressed stego image
-------------------------------------------------
"""


def extract_binary(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"O arquivo {filepath} não foi encontrado.")
        
    with open(filepath, 'rb') as f:
        # 1. Ler e Verificar o Número Mágico
        if f.read(4) != b'STGC':
            raise ValueError("Formato de arquivo inválido. Assinatura 'STGC' não encontrada.")
        
        # 2. Ler o Tamanho do Cabeçalho
        header_size_bytes = f.read(1)
        print(header_size_bytes)
        header_size = struct.unpack('>B', header_size_bytes)[0]
        
        # 3. Ler o Bloco do Cabeçalho
        header_bytes = f.read(header_size - 1)

        # 4. Analisar (Parse) o Bloco do Cabeçalho
        offset = 0
        
        # Parte fixa inicial: versão e index_s
        version, index_s = struct.unpack('>BB', header_bytes[offset:offset+2])
        offset += 2
        
        # Usa 'index_s' para ler as listas de 2 bytes (short, 'H')
        list_format = f'>{index_s}H'
        list_size = struct.calcsize(list_format)
        
        segments_lengths = list(struct.unpack(list_format, header_bytes[offset:offset+list_size]))
        offset += list_size
        
        segment_indices = list(struct.unpack(list_format, header_bytes[offset:offset+list_size]))
        offset += list_size
        
        # Guarda os metadados num dicionário
        metadata = {
            'version': version,
            's': index_s,
            'segments_lengths': segments_lengths,
            'segment_indices': segment_indices,
        }
        
        print(f"Metadata extraída: {metadata}")
        
        # 5. Ler os Bitmaps Comprimidos

        bitmaps_data = []
        for i in range(index_s):
            bitmap_length = segments_lengths[i]
            bitmap_compressed = f.read(bitmap_length)
            bitmap_decompressed = zlib.decompress(bitmap_compressed)
            bitmaps_data.append(bitmap_decompressed)
        


        # DEBUG: verificar se todos os bitmaps foram lidos corretamente
        for i, bitmap in enumerate(bitmaps_data):
            print(f"   > Bitmap {i}: {bitmap[:50]}\n   > {len(bitmap)} bytes")
            print(f"   > Bitmap original {i}: {DEBUG_BITMAPS[i][:50]}\n   > {len(DEBUG_BITMAPS[i])} bytes")
        




        print(f"   > Total de bitmaps lidos: {len(bitmaps_data)}")
        print("✅ Arquivo STGC lido e analisado com sucesso.")

        return {
            'metadata': metadata,
            #'bitmaps_data': bitmaps_data,
            #'stego_image_data': stego_image_data
        }
    

def main():
    name = "pe"
    dicom_data = load_dicom_image(f"images/{name}.dcm")
    image_array = dicom_data.pixel_array
    bits_stored = dicom_data.BitsStored
    message = (
        "Mensagem longa para testar o algoritmo de decomposição adaptativa em DICOM. \n"
        "Esta mensagem deve ser suficientemente grande para preencher vários planos de bits e \n"
        "permitir a avaliação do algoritmo de embaralhamento e distribuição de segmentos entre os \n"
        "planos locais. Logo será possível verificar se a mensagem é corretamente embutida e \n"
        "recuperada. O esperado é que a mensagem seja embutida e recuperada corretamente. Sempre \n"
        "que a mensagem seja embutida e recuperada corretamente.\n"
        "\n"
        "Mensagem longa para testar o algoritmo de decomposição adaptativa em DICOM. \n"
        "Esta mensagem deve ser suficientemente grande para preencher vários planos de bits e \n"
        "permitir a avaliação do algoritmo de embaralhamento e distribuição de segmentos entre os \n"
        "planos locais. Logo será possível verificar se a mensagem é corretamente embutida e \n"
        "recuperada. O esperado é que a mensagem seja embutida e recuperada corretamente. Sempre \n"
        "que a mensagem seja embutida e recuperada corretamente."
    )
    message_bits = message_to_bits(message)

    index_s, global_modality, local_modality = adaptive_modalities_decomposition(image_array, beta=0.5, nbits=bits_stored)
    local_stego, bitmaps, total_used, segments_lengths, segment_indices = lsb_embed_multi_plane(local_modality, message_bits)
    image_stego = merge_modalities(global_modality, local_stego)

    
    stego_compressed = compress_image(image_stego, codec='jxl')

    # Cria header sem alturas dos bitmaps
    header_bytes = create_header(segments_lengths, segment_indices)
    binary_size = create_binary_file(f"output/{name}.stgc", header_bytes, stego_compressed, bitmaps)

    print(f"   > Tamanho do arquivo binário: {binary_size / (1024 * 1024):<.2f} MB")

    extract_binary(f"output/{name}.stgc")

# Exemplo de uso do decode
if __name__ == "__main__":
    main()

