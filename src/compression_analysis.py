import os
import pydicom
import numpy as np
from tools import load_image

def calculate_bpp_dcm(dcm_filepath):
    """
    Calcula bits por pixel para uma imagem DICOM
    """
    # Carrega a imagem DICOM
    img_data = load_image(dcm_filepath)
    image = img_data['image']
    
    # Obtém dimensões da imagem
    height, width = image.shape
    total_pixels = height * width
    
    # Calcula o tamanho do arquivo em bits
    file_size_bytes = os.path.getsize(dcm_filepath)
    file_size_bits = file_size_bytes * 8
    
    # Calcula bits por pixel
    bpp = file_size_bits / total_pixels
    
    return bpp, file_size_bytes, total_pixels, (height, width)

def calculate_bpp_bin(bin_filepath, image_dimensions):
    """
    Calcula bits por pixel para um arquivo BIN baseado nas dimensões da imagem original
    """
    height, width = image_dimensions
    total_pixels = height * width
    
    # Calcula o tamanho do arquivo BIN em bits
    file_size_bytes = os.path.getsize(bin_filepath)
    file_size_bits = file_size_bytes * 8
    
    # Calcula bits por pixel
    bpp = file_size_bits / total_pixels
    
    return bpp, file_size_bytes

def analyze_compression_ratio(original_dcm_path, compressed_bin_path):
    """
    Analisa a taxa de compressão entre imagem DCM original e arquivo BIN comprimido
    """
    print("=" * 60)
    print("ANÁLISE DE TAXA DE COMPRESSÃO")
    print("=" * 60)
    
    # Analisa imagem DCM original
    print("\n1. IMAGEM DCM ORIGINAL:")
    print("-" * 30)
    bpp_original, size_original, total_pixels, dimensions = calculate_bpp_dcm(original_dcm_path)
    print(f"Arquivo: {os.path.basename(original_dcm_path)}")
    print(f"Dimensões: {dimensions[1]} x {dimensions[0]} pixels")
    print(f"Total de pixels: {total_pixels:,}")
    print(f"Tamanho do arquivo: {size_original:,} bytes ({size_original/1024:.2f} KB)")
    print(f"Bits por pixel (bpp): {bpp_original:.4f}")
    
    # Analisa arquivo BIN comprimido
    print("\n2. ARQUIVO BIN COMPRIMIDO:")
    print("-" * 30)
    bpp_compressed, size_compressed = calculate_bpp_bin(compressed_bin_path, dimensions)
    print(f"Arquivo: {os.path.basename(compressed_bin_path)}")
    print(f"Tamanho do arquivo: {size_compressed:,} bytes ({size_compressed/1024:.2f} KB)")
    print(f"Bits por pixel (bpp): {bpp_compressed:.4f}")
    
    # Calcula taxa de compressão
    print("\n3. ANÁLISE DE COMPRESSÃO:")
    print("-" * 30)
    compression_ratio = size_original / size_compressed
    space_savings = ((size_original - size_compressed) / size_original) * 100
    bpp_reduction = ((bpp_original - bpp_compressed) / bpp_original) * 100
    
    print(f"Taxa de compressão: {compression_ratio:.2f}:1")
    print(f"Economia de espaço: {space_savings:.2f}%")
    print(f"Redução de bpp: {bpp_reduction:.2f}%")
    print(f"Tamanho original: {size_original:,} bytes")
    print(f"Tamanho comprimido: {size_compressed:,} bytes")
    print(f"Bytes economizados: {size_original - size_compressed:,} bytes")
    
    return {
        'bpp_original': bpp_original,
        'bpp_compressed': bpp_compressed,
        'size_original': size_original,
        'size_compressed': size_compressed,
        'compression_ratio': compression_ratio,
        'space_savings': space_savings,
        'bpp_reduction': bpp_reduction
    }

def main():
    """
    Executa análise de compressão para o exemplo peito.dcm e peito_stego_data.bin
    """
    # Caminhos dos arquivos
    original_dcm = "/home/wesleyn/Documents/codec-tcc/images/peito.dcm"
    compressed_bin = "/home/wesleyn/Documents/codec-tcc/output/peito/peito_stego_data.bin"
    
    # Verifica se os arquivos existem
    if not os.path.exists(original_dcm):
        print(f"Erro: Arquivo não encontrado: {original_dcm}")
        return
    
    if not os.path.exists(compressed_bin):
        print(f"Erro: Arquivo não encontrado: {compressed_bin}")
        return
    
    # Executa análise
    results = analyze_compression_ratio(original_dcm, compressed_bin)
    
    print("\n" + "=" * 60)
    print("RESUMO DOS RESULTADOS:")
    print("=" * 60)
    print(f"A compressão alcançou uma taxa de {results['compression_ratio']:.2f}:1")
    print(f"Economia de {results['space_savings']:.2f}% do espaço original")
    print(f"BPP reduzido de {results['bpp_original']:.4f} para {results['bpp_compressed']:.4f}")

if __name__ == "__main__":
    main()