#!/usr/bin/env python3
"""
Teste dos algoritmos de compressão incluindo PNG
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools import load_image, compress_image_with_algorithm
import numpy as np

def test_all_compression_algorithms(image_path):
    """Testa todos os algoritmos de compressão disponíveis"""
    print("=" * 70)
    print("TESTE DE ALGORITMOS DE COMPRESSÃO")
    print("=" * 70)
    
    # Carregar imagem
    img_data = load_image(image_path)
    image = img_data['image']
    
    print(f"📁 Imagem: {os.path.basename(image_path)}")
    print(f"📏 Dimensões: {image.shape}")
    print(f"🎯 Tipo: {image.dtype}")
    print(f"📊 Tamanho original: {image.nbytes / 1024:.2f} KB")
    print("-" * 70)
    
    # Algoritmos para testar
    algorithms = ['zlib', 'gzip', 'bz2', 'lzma', 'png', 'gdcm']
    results = {}
    
    for algorithm in algorithms:
        print(f"\n🔄 Testando {algorithm.upper()}...")
        try:
            compressed_info = compress_image_with_algorithm(image, algorithm)
            
            # Calcular métricas
            original_size = image.nbytes
            compressed_size = len(compressed_info['compressed_data'])
            compression_ratio = original_size / compressed_size
            space_savings = ((original_size - compressed_size) / original_size) * 100
            bpp = (compressed_size * 8) / (image.shape[0] * image.shape[1])
            
            results[algorithm] = {
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'space_savings': space_savings,
                'bpp': bpp,
                'success': True
            }
            
            print(f"   ✅ Taxa de compressão: {compression_ratio:.2f}:1")
            print(f"   💾 Economia de espaço: {space_savings:.2f}%")
            print(f"   📐 BPP: {bpp:.4f}")
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results[algorithm] = {'success': False, 'error': str(e)}
    
    # Resumo comparativo
    print("\n" + "=" * 70)
    print("RESUMO COMPARATIVO")
    print("=" * 70)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        # Ordenar por taxa de compressão (melhor = maior)
        sorted_by_ratio = sorted(successful_results.items(), 
                               key=lambda x: x[1]['compression_ratio'], reverse=True)
        
        print(f"{'Algoritmo':<10} {'Taxa':<8} {'Economia':<10} {'BPP':<8} {'Tamanho':<12}")
        print("-" * 70)
        
        for algo, data in sorted_by_ratio:
            print(f"{algo.upper():<10} {data['compression_ratio']:.2f}:1   "
                  f"{data['space_savings']:.1f}%      {data['bpp']:.4f}  "
                  f"{data['compressed_size']/1024:.1f} KB")
        
        # Melhor resultado
        best_algo, best_data = sorted_by_ratio[0]
        print(f"\n🏆 MELHOR RESULTADO: {best_algo.upper()}")
        print(f"   Taxa de compressão: {best_data['compression_ratio']:.2f}:1")
        print(f"   BPP: {best_data['bpp']:.4f}")
        print(f"   Economia: {best_data['space_savings']:.2f}%")
    
    return results

def main():
    # Testar com a imagem peito.dcm
    image_path = "/home/wesleyn/Documents/codec-tcc/images/peito.dcm"
    
    if not os.path.exists(image_path):
        print(f"❌ Arquivo não encontrado: {image_path}")
        # Listar arquivos disponíveis
        images_dir = "/home/wesleyn/Documents/codec-tcc/images/"
        if os.path.exists(images_dir):
            print("📁 Arquivos disponíveis:")
            for f in os.listdir(images_dir):
                if f.endswith('.dcm'):
                    print(f"   - {f}")
        return
    
    results = test_all_compression_algorithms(image_path)
    
    print(f"\n✅ Teste concluído! PNG foi {'adicionado com sucesso' if results.get('png', {}).get('success', False) else 'adicionado mas com problemas'}.")

if __name__ == "__main__":
    main()