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
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

def create_dicom(image_array: np.ndarray, filename="synthetic.dcm", bits_stored=16) -> FileDataset:
    """
    Cria um DICOM sintético VÁLIDO e que pode ser visualizado a partir de um numpy array.
    """
    if image_array.ndim != 2:
        raise ValueError("A imagem deve ser 2D (grayscale).")
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

    # --- Salvar no disco ---
    # A flag `write_like_original=False` não é necessária com FileDataset
    ds.save_as(filename, write_like_original=False)
    return ds

def compress_png(image_array: np.ndarray) -> bytes:
    mode = "I;16" if image_array.dtype == np.uint16 else "L"
    pil_img = Image.fromarray(image_array, mode=mode)
    
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG", optimize=True, compress_level=9)

    return buffer.getvalue()

def decompress_png(png_bytes: bytes) -> np.ndarray:
    buffer = io.BytesIO(png_bytes)
    pil_img = Image.open(buffer)
    arr = np.array(pil_img)

    return arr.astype(np.uint16) if arr.dtype == np.uint16 else arr.astype(np.uint8)

def load_dicom_image(file_path):
    dicom_data = pydicom.dcmread(file_path)
    return dicom_data

def save_dicom_image(dicom_data, file_path):
    dicom_data.save_as(file_path)

def save_image_binary(local_stego: np.ndarray, global_compressed: np.ndarray, file_path: str):
    bitstream = build_bitstream(local_stego, global_compressed)
    with open(file_path, 'wb') as file:
        file.write(bitstream)

def merge_modalities(global_planes: np.ndarray, local_planes: np.ndarray) -> np.ndarray:
    dtype = global_planes[0].dtype

    global_image = np.zeros_like(global_planes[0], dtype)
    for i, plane in enumerate(global_planes):
        global_image |= (plane << (i + len(local_planes)))

    local_image = np.zeros_like(local_planes[0], dtype)
    for i, plane in enumerate(local_planes):
        local_image |= (plane << i)

    merged_image = global_image | local_image
    return merged_image

def message_to_bits(message: str) -> list:
    return ''.join(f"{ord(c):08b}" for c in message)

def lsb_embed_multi_plane(local_planes, message_bits):
        s = len(local_planes)
        total_bits = len(message_bits)

        segment_size, remainder = divmod(total_bits, s)

        segments = [message_bits[i * segment_size + min(i, remainder): (i + 1) * segment_size + min(i + 1, remainder)] for i in range(s)]
        segment_indices = list(range(s)); 
        
        random.seed(42); 
        random.shuffle(segment_indices)
    
        weights = [(s - i) ** 2 for i in range(s)]
        total_weight = sum(weights)

        distributed_sizes = [max(1, int((w / total_weight) * total_bits)) for w in weights]
        excess = sum(distributed_sizes) - total_bits

        if excess > 0:
            max_idx = distributed_sizes.index(max(distributed_sizes))
            distributed_sizes[max_idx] -= excess

        bit_idx = 0
        new_segments = []
        
        for dest_plane_idx in segment_indices:
            size = distributed_sizes[dest_plane_idx]
            new_segments.append(message_bits[bit_idx:bit_idx+size])
            bit_idx += size

        segments = new_segments
        stego_planes = [None] * s
        bitmaps = [None] * s
        segments_lengths = [0] * s
        total_used = 0

        for orig_segment_idx, dest_plane_idx in enumerate(segment_indices):
            segment = segments[orig_segment_idx]
            plane = local_planes[dest_plane_idx]

            h, w = plane.shape
            stego_plane = plane.copy()
            bitmap = np.zeros_like(plane, dtype=np.uint8)
            num_bits = min(len(segment), h * w)

            linear_indices = np.arange(num_bits)
            y_coords = linear_indices // w
            x_coords = linear_indices % w
            original_pixels = stego_plane[y_coords, x_coords]
            current_lsbs = original_pixels & 1
            msg_bits = np.array(list(segment[:num_bits]), dtype=np.uint8)
            
            needs_change = (current_lsbs != msg_bits)
            if np.any(needs_change):
                new_pixels = (original_pixels & 0xFE) | msg_bits
                stego_plane[y_coords[needs_change], x_coords[needs_change]] = new_pixels[needs_change]

            bitmap[y_coords, x_coords] = 1
            stego_planes[dest_plane_idx] = stego_plane
            bitmaps[dest_plane_idx] = bitmap
            segments_lengths[dest_plane_idx] = len(segment)

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
    probabilities = counts[counts > 0] / len(data_array.ravel())
    
    # Aplicamos a fórmula da entropia
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_mutual_information(bit_plane, image_array):
    """
    Calcula a informação mútua I(X;Y) entre um plano de bit (X) e a imagem (Y).
    Esta versão é específica e precisa para a tarefa, usando binning exato.
    I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    # Se uma das variáveis for constante, a informação mútua é zero.
    if bit_plane.min() == bit_plane.max() or image_array.min() == image_array.max():
        return 0.0

    # H(X) - Entropia do plano de bit
    h_x = calculate_entropy(bit_plane)

    # H(Y) - Entropia da imagem inteira
    h_y = calculate_entropy(image_array)

    # H(X,Y) - Entropia conjunta
    # Usamos um histograma 2D com o número exato de estados para cada variável.
    # Plano de bit (X): 2 estados (0, 1)
    # Imagem (Y): image_array.max() + 1 estados (ex: 256 para 8-bit)
    # Define o número de bins com base no tipo de dados, não no valor máximo.
    if image_array.dtype == np.uint8:
        num_bins_y = 256  # Para imagens de 8 bits (0-255)
    elif image_array.dtype == np.uint16:
        num_bins_y = 65536 # Para imagens de 16 bits (0-65535)
    else:
        # Fallback para outros tipos de dados
        num_bins_y = int(image_array.max() + 1)

    bins = [2, num_bins_y]
    # Define o range explicitamente para garantir que o histograma cubra todos os valores possíveis
    range_y = [0, num_bins_y - 1]
    
    joint_hist = np.histogram2d(bit_plane.ravel(), image_array.ravel(), bins=bins, range=[[0, 1], range_y])[0]
    
    
    joint_probs = joint_hist / joint_hist.sum()
    h_xy = -np.sum(joint_probs[joint_probs > 0] * np.log2(joint_probs[joint_probs > 0]))
    
    # Informação Mútua
    mi = h_x + h_y - h_xy
    
    # Garante que não seja negativo devido a pequenos erros de precisão do float
    return max(0.0, mi)

def adaptive_modalities_decomposition(image_array, beta=0.8, nbits=None):
    """
    Algoritmo 2 (Adaptado): Encontra o ponto de corte 's' usando o cálculo de MI específico.
    """    
    # Determina o número de bits a partir dos metadados ou do próprio array
    nbits = image_array.dtype.itemsize * 8 if nbits is None else nbits
    print(f"   - Profundidade de bits efetiva: {nbits}")

    bit_planes = [(image_array >> i) & 1 for i in range(nbits)]
    
    total_info = calculate_entropy(image_array)
    target_info = beta * total_info
    
    print(f"   - Informação total da imagem: {total_info:.4f}")
    print(f"   - Meta de retenção ({beta*100}%): {target_info:.4f}")
    
    cumulative_info = 0.0
    s = 1  # Ponto de corte padrão
    
    for i in range(nbits):
        current_plane = bit_planes[i]
        
        mi = calculate_mutual_information(current_plane, image_array)
        cumulative_info += mi
                
        if cumulative_info >= target_info:
            s = i + 1
            break
    
    # Separa os planos
    local_planes = bit_planes[:s]   # 's' planos menos significativos
    global_planes = bit_planes[s:]  # O restante dos planos mais significativos
    
    return s, global_planes, local_planes




def main():
    name = "peito"
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

    index_s, global_modality, local_modality = adaptive_modalities_decomposition(image_array, beta=0.8, nbits=bits_stored)
    local_stego, bitmap, total_used, segments_lengths, segment_indices = lsb_embed_multi_plane(local_modality, message_bits)
    image_stego = merge_modalities(global_modality, local_stego)

    # Salva a imagem esteganografada
    create_dicom(image_stego, f"output/{name}_stego.dcm", bits_stored)

    print(f"> dtype: {dicom_data.pixel_array.dtype}, shape: {dicom_data.pixel_array.shape}")
    print(f"> Index s: {index_s}")
    print(f"> Total usado: {total_used}")
    print(f"> Segmentos: {segments_lengths}")
    print(f"> Índices dos segmentos: {segment_indices}")

if __name__ == "__main__":
    main()
