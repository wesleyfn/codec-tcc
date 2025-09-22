import numpy as np
import os
from tools import (
    load_image,
    create_output_folder,
    extract_bit_planes,
    build_modality,
    build_image_from_modality,
    compress_image_with_algorithm,
    save_compressed_stego_bitstream,
    message_to_bits
)

def _predictor_left_top(img, y, x):
    """Preditor usado no PEE - média entre pixel à esquerda e acima"""
    left = int(img[y, x-1]) if x > 0 else 0
    top = int(img[y-1, x]) if y > 0 else 0
    return (left + top) // 2

def pee_embed(local_img, nbits, message_bits, threshold):
    """
    PEE (Prediction Error Expansion) embedding
    
    Args:
        local_img: Imagem local onde inserir a mensagem
        nbits: Número de bits por pixel
        message_bits: Lista de bits da mensagem
        threshold: Limiar para expansão do erro de predição
    
    Returns:
        tuple: (imagem_stego, bitmap, bits_usados)
    """
    h, w = local_img.shape
    local_int = local_img.astype(np.int32)
    maxval = (1 << nbits) - 1

    bitmap = np.zeros_like(local_img, dtype=np.uint8)
    L_stego = local_int.copy()
    bits = list(message_bits)
    bit_idx = 0

    for y in range(h):
        for x in range(w):
            pred = _predictor_left_top(local_int, y, x)  # usa imagem original
            e = int(local_int[y, x]) - pred             # erro original
            
            if abs(e) <= threshold and bit_idx < len(bits):
                b = bits[bit_idx]
                e_exp = 2*e + b  # expansão do erro
                newp = pred + e_exp
                
                if 0 <= newp <= maxval:
                    L_stego[y, x] = newp
                    bitmap[y, x] = 255  # marca posição no bitmap
                    bit_idx += 1

    return L_stego.astype(local_img.dtype), bitmap, bit_idx

def steganography_encode(local_image, nbits, message, threshold):
    """
    Função principal de codificação esteganográfica
    
    Args:
        local_image: Imagem local para inserir mensagem
        nbits: Número de bits por pixel
        message: Mensagem a ser inserida
        threshold: Limiar PEE
    
    Returns:
        tuple: (imagem_stego, bitmap, bits_usados)
    """
    message_bits = message_to_bits(message)
    L_stego, bitmap, used = pee_embed(local_image, nbits, message_bits, threshold)
    return L_stego, bitmap, used

def encode_image(image_path, message, threshold=2, s=1, local_algorithm='lzma', global_algorithm='zlib'):
    """
    Pipeline completo de codificação esteganográfica
    
    Args:
        image_path: Caminho para a imagem DICOM (.dcm)
        message: Mensagem a ser inserida
        threshold: Limiar PEE (padrão: 2)
        s: Índice de modalidade (padrão: 1)
        local_algorithm: Algoritmo de compressão local (padrão: 'lzma')
        global_algorithm: Algoritmo de compressão global (padrão: 'zlib')
    
    Returns:
        dict: Resultados da codificação
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
    local_planes, global_planes = build_modality(planes, s=s)
    
    # Construir imagens local e global
    local_image = np.zeros_like(image, dtype=image.dtype)
    for i, plane in enumerate(local_planes):
        local_image |= (plane.astype(image.dtype) << i)

    global_image = np.zeros_like(image, dtype=image.dtype)
    for i, plane in enumerate(global_planes):
        global_image |= (plane.astype(image.dtype) << (i + len(local_planes)))
    
    # Comprimir componente global
    compressed_global_info = compress_image_with_algorithm(global_image, global_algorithm)
    
    # Aplicar esteganografia na componente local
    local_stego, bitmap, used = steganography_encode(local_image, len(local_planes), message, threshold)
    
    # Reconstruir imagem esteganográfica
    stego_image = build_image_from_modality(local_stego, global_image)
    
    # Criar estrutura de pastas organizada
    image_base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = create_output_folder(image_base_name)
    
    # Comprimir componente local
    compressed_local_info = compress_image_with_algorithm(local_stego, local_algorithm)
    
    # Preparar parâmetros de esteganografia
    stego_params = {
        'threshold': threshold,
        's': s,
        'bits_used': used
    }
    
    # Salvar bitstream comprimido completo
    bitstream_path = save_compressed_stego_bitstream(
        compressed_local_info, 
        compressed_global_info, 
        output_folder, 
        image_base_name, 
        bitmap, 
        stego_params
    )
    
    print(f"✅ Codificação concluída - {os.path.basename(bitstream_path)}")
    
    return {
        'stego_image': stego_image,
        'bitmap': bitmap,
        'output_folder': output_folder,
        'bitstream_path': bitstream_path,
        'bits_used': used,
        'local_compressed': compressed_local_info,
        'global_compressed': compressed_global_info,
        'stego_params': stego_params,
        'original_data': data
    }

def main():
    """Exemplo de uso do encoder"""
    dir = "images"
    name = "peito"
    image_path = f'{dir}/{name}.dcm' 
    message = (
        " \"É uma mensagem longa para testar a implementação de esteganografia\n"
        "  com funcionalidade de inserção e extração PEE e deve ser suficientemente\n"
        "  extensa para cobrir múltiplos bits e garantir que o processo de inserção\n"
        "  seja devidamente exercitado. Lorem ipsum dolor sit amet, consectetur\n"
        "  adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore\n"
        "  magna aliqua. Lorem ipsum dolor sit amet, consectetur adipiscing elit.\n"
        "  Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\""
    )
    
    # png - png = 2.29
    # jpegls - jpegls = 4.91
    # jpeg2000 - jpeg2000 = 5.08


    result = encode_image(
        image_path=image_path,
        message=message,
        threshold=2,
        s=3,
        local_algorithm="jpeg2000",
        global_algorithm="jpegls"
    )
    
    print(f"\n🎉 Processo de codificação finalizado!")
    print(f"   Bits utilizados: {result['bits_used']}")
    print(f"   Arquivo de saída: {result['bitstream_path']}")

if __name__ == '__main__':
    main()