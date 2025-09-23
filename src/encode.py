import numpy as np
import os
from PIL import Image
from tools import (
    load_image,
    create_output_folder,
    extract_bit_planes,
    build_modality,
    build_image_from_modality,
    compress_image_with_algorithm,
    save_compressed_stego_bitstream_multi_ultra_compact,
    message_to_bits,
    find_optimal_cut_point,
    save_stego_dicom
)


def lsb_embed_multi_plane(local_planes, message_bits, s):
    """
    LSB embedding distribuído em múltiplos planos com pesos exponenciais
    
    Args:
        local_planes: Lista de planos de bits locais [plane0, plane1, ..., plane_s-1]
        message_bits: Lista de bits da mensagem completa
        s: Número de planos a usar (n_lsb = s)
    
    Returns:
        tuple: (planos_stego, bitmaps, comprimentos_segmentos)
    """
    import random
    
    # Dividir mensagem em s segmentos usando pesos exponenciais
    total_bits = len(message_bits)
    if total_bits == 0:
        return local_planes.copy(), [np.zeros_like(plane) for plane in local_planes], [0] * s
    
    # Calcular pesos exponenciais: planos LSB (índices menores) recebem mais dados
    # Peso do plano i = 2^(s-i-1), onde i vai de 0 a s-1
    weights = [2**(s-i-1) for i in range(s)]
    total_weight = sum(weights)
    
    # Calcular comprimentos dos segmentos baseados nos pesos
    segments_lengths = []
    allocated_bits = 0
    
    for i in range(s - 1):
        # Proporção do peso atual em relação ao total
        proportion = weights[i] / total_weight
        segment_length = int(total_bits * proportion)
        segments_lengths.append(segment_length)
        allocated_bits += segment_length
    
    # Último segmento pega os bits restantes
    remaining_bits = total_bits - allocated_bits
    segments_lengths.append(remaining_bits)
    
    # Print resumo da distribuição
    print(f"📊 Distribuição exponencial: {segments_lengths} bits em {s} planos")
    
    # Dividir bits da mensagem em segmentos baseados nos pesos
    message_segments = []
    bit_idx = 0
    for length in segments_lengths:
        segment = message_bits[bit_idx:bit_idx + length]
        message_segments.append(segment)
        bit_idx += length
    
    # Usar ordem sequencial baseada nos pesos (sem embaralhamento aleatório)
    segment_indices = list(range(s))  # Ordem: 0, 1, 2, ..., s-1
    
    # segments_lengths já está na ordem correta dos pesos
    actual_segments_lengths = segments_lengths.copy()
    
    stego_planes = []
    bitmaps = []
    
    for i in range(s):
        plane = local_planes[i].copy()
        original_plane = plane.copy()
        segment_idx = segment_indices[i]  # Sempre igual a i agora (ordem sequencial)
        segment_bits = message_segments[segment_idx]
        
        # Embed LSB neste plano usando operações vetorizadas
        h, w = plane.shape
        bitmap = np.zeros_like(plane, dtype=np.uint8)
        
        if len(segment_bits) > 0:
            # Número de bits a processar (limitado pelo tamanho do plano)
            max_bits = min(len(segment_bits), h * w)
            
            # Converter bits do segmento para array numpy
            segment_array = np.array(segment_bits[:max_bits], dtype=np.uint8)
            
            # Flatten da imagem para processamento linear
            plane_flat = plane.flatten()
            
            # Criar máscara de posições onde bits serão embarcados
            positions_mask = np.arange(len(plane_flat)) < max_bits
            
            # LSBs originais das posições a serem modificadas
            original_lsbs = plane_flat[:max_bits] & 1
            
            # Máscara onde LSB precisa ser invertido
            flip_mask = original_lsbs != segment_array
            
            # Aplicar flip apenas onde necessário
            plane_flat[:max_bits] = np.where(flip_mask, 
                                           plane_flat[:max_bits] ^ 1, 
                                           plane_flat[:max_bits])
            
            # Reshape de volta
            plane[:] = plane_flat.reshape(h, w)
            
            # Marcar posições usadas no bitmap
            bitmap_flat = bitmap.flatten()
            bitmap_flat[:max_bits] = 1
            bitmap[:] = bitmap_flat.reshape(h, w)
        
        stego_planes.append(plane)
        bitmaps.append(bitmap)
    
    return stego_planes, bitmaps, actual_segments_lengths, segment_indices



def steganography_encode(local_planes, message, s):
    """
    Esteganografia LSB distribuída em múltiplos planos (n_lsb = s)
    
    Args:
        local_planes: Lista de planos de bits locais
        message: Mensagem a ser inserida
        s: Número de planos a usar (n_lsb = s)
    
    Returns:
        tuple: (planos_stego, bitmaps, comprimentos_segmentos, bits_totais, indices_embaralhamento)
    """
    message_bits = message_to_bits(message)
    stego_planes, bitmaps, segments_lengths, segment_indices = lsb_embed_multi_plane(local_planes, message_bits, s)
    
    return stego_planes, bitmaps, segments_lengths, len(message_bits), segment_indices

def encode_image(image_path, message, beta=0.8, local_algorithm='lzma', global_algorithm='zlib'):
    """
    Pipeline completo de codificação esteganográfica LSB multi-plano
    
    Args:
        image_path: Caminho para a imagem DICOM (.dcm)
        message: Mensagem a ser inserida
        beta: Parâmetro limiar para cálculo automático de s* (padrão: 0.8)
        local_algorithm: Algoritmo de compressão local (padrão: 'lzma')
        global_algorithm: Algoritmo de compressão global (padrão: 'zlib')
    
    Returns:
        dict: Resultados da codificação com s* calculado automaticamente
    """
    # Validar que é arquivo DICOM
    if not image_path.lower().endswith('.dcm'):
        raise ValueError("Apenas arquivos DICOM (.dcm) são suportados para codificação")
    
    # Carregar imagem
    data = load_image(image_path)
    image = data['image']
    nbits = data['bits_stored']
    
    # Validar que é imagem 2D
    if len(image.shape) != 2:
        raise ValueError(f"Apenas imagens DICOM 2D são suportadas. Dimensões encontradas: {image.shape}")
    
    print(f"📊 Processando {os.path.basename(image_path)} - {image.shape}, {nbits} bits/pixel")
    
    # Extrair planos de bits
    planes = extract_bit_planes(image, nbits)
    
    # Calcular s* automaticamente usando informação mútua
    s = find_optimal_cut_point(image, planes, beta=beta)
    
    local_planes, global_planes = build_modality(planes, s=s)
    
    # Arquitetura LSB multi-plano: n_lsb = s (usar todos os planos locais)
    # Aplicar esteganografia LSB multi-plano nos planos locais
    stego_local_planes, bitmaps, segments_lengths, total_bits, segment_indices = steganography_encode(local_planes, message, s)
    
    # Construir imagem local esteganográfica
    local_stego = np.zeros_like(image, dtype=image.dtype)
    for i, plane in enumerate(stego_local_planes):
        local_stego |= (plane.astype(image.dtype) << i)
    
    # Construir imagem global (inalterada)
    global_image = np.zeros_like(image, dtype=image.dtype)
    for i, plane in enumerate(global_planes):
        global_image |= (plane.astype(image.dtype) << (i + len(local_planes)))
    
    # Comprimir componente global
    compressed_global_info = compress_image_with_algorithm(global_image, global_algorithm)
    
    # Reconstruir imagem esteganográfica completa
    stego_image = build_image_from_modality(local_stego, global_image)
    
    # Criar estrutura de pastas organizada
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = create_output_folder(image_base_name)
    
    # Comprimir componente local
    compressed_local_info = compress_image_with_algorithm(local_stego, local_algorithm)
    
    # Preparar parâmetros de esteganografia LSB multi-plano
    stego_params = {
        'method': 'lsb',  # Apenas LSB multi-plano
        'n_lsb': s,  # n_lsb = s
        's': s,
        'bits_used': total_bits,
        'segments_lengths': segments_lengths,  # Comprimentos dos segmentos
        'segment_indices': segment_indices  # Ordem de embaralhamento
    }
    
    # Salvar bitstream comprimido completo com múltiplos bitmaps (formato compacto STG4)
    bitstream_path = save_compressed_stego_bitstream_multi_ultra_compact(
        compressed_local_info, 
        compressed_global_info, 
        output_folder, 
        image_base_name, 
        bitmaps,  # Lista de bitmaps (um por plano)
        stego_params
    )
    
    # Salvar imagem esteganográfica como PNG normalizada
    stego_image_path = os.path.join(output_folder, f"{image_base_name}_stego.png")
    
    if nbits > 8:
        # Para imagens 16-bit, usar normalização adaptativa que preserva contraste
        # Encontrar valores mínimo e máximo reais da imagem
        min_val = np.min(stego_image)
        max_val = np.max(stego_image)
        
        # Normalizar usando o range real da imagem para preservar contraste
        if max_val > min_val:
            stego_normalized = ((stego_image.astype('float64') - min_val) / (max_val - min_val) * 255).astype('uint8')
        else:
            # Caso especial: imagem uniforme
            stego_normalized = np.full_like(stego_image, 128, dtype='uint8')
    else:
        # Para imagens 8-bit, usar diretamente
        stego_normalized = stego_image.astype('uint8')
    
    stego_pil = Image.fromarray(stego_normalized)
    stego_pil.save(stego_image_path)
    
    # Salvar imagem esteganográfica como DICOM
    stego_dicom_path = os.path.join(output_folder, f"{image_base_name}_stego.dcm")
    save_stego_dicom(data['metadata'], stego_image, stego_dicom_path)
    
    print(f"✅ Codificação finalizada - {s} planos, {total_bits} bits")
    
    return {
        'stego_image': stego_image,
        'stego_image_path': stego_image_path,
        'stego_dicom_path': stego_dicom_path,
        'bitmaps': bitmaps,  # Lista de bitmaps
        'output_folder': output_folder,
        'bitstream_path': bitstream_path,
        'bits_used': total_bits,
        'segments_lengths': segments_lengths,
        'local_compressed': compressed_local_info,
        'global_compressed': compressed_global_info,
        'stego_params': stego_params,
        'original_data': data,
        's_optimal': s  # Valor de s* calculado ou usado
    }

def main():
    """Exemplo de uso do encoder com s* automático"""
    name = "torax"
    image_path = f'images/{name}.dcm' 
    message = (
        " \"É uma mensagem longa para testar a implementação de esteganografia\n"
        "  com funcionalidade de inserção e extração LSB multi-plano e deve ser\n"
        "  suficientemente extensa para cobrir múltiplos bits e garantir que o\n"
        "  processo de inserção seja devidamente exercitado. Lorem ipsum dolor\n"
        "  sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt\n"
        "  ut labore et dolore magna aliqua. Lorem ipsum dolor sit amet, consectetur\n"
        "  adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore\n"
        "  magna aliqua.\""
    )
    
    # Algoritmos de compressão testados:
    # png - png = 2.29, jpegls - jpegls = 4.91, jpeg2000 - jpeg2000 = 5.08
    
    # Codificação LSB multi-plano com s* calculado automaticamente via informação mútua
    result = encode_image(
        image_path=image_path,
        message=message,
        beta=0.8,  # Parâmetro limiar conforme paper (s* calculado automaticamente)
        local_algorithm="jpegls",
        global_algorithm="png"
    )
    
    print(f"📁 Salvo: {os.path.basename(result['bitstream_path'])}")

if __name__ == '__main__':
    main()