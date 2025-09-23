#!/usr/bin/env python3
"""
Benchmark das otimizações de performance
"""
import sys
import os
sys.path.append('src')

import time
import numpy as np
from tools import message_to_bits, bits_to_message
from encode import encode_image
from decode import decode_from_bitstream

def benchmark_message_conversion():
    """Benchmark das funções de conversão de mensagem"""
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
    
    print("=== BENCHMARK CONVERSÃO MENSAGEM ===")
    
    # Benchmark message_to_bits
    start = time.time()
    for _ in range(1000):
        bits = message_to_bits(message)
    end = time.time()
    print(f"message_to_bits (1000x): {(end-start)*1000:.2f}ms")
    
    # Benchmark bits_to_message
    start = time.time()
    for _ in range(1000):
        recovered = bits_to_message(bits)
    end = time.time()
    print(f"bits_to_message (1000x): {(end-start)*1000:.2f}ms")
    
    # Verificar se a conversão está correta
    print(f"Conversão correta: {message == recovered}")
    print(f"Tamanho original: {len(message)} chars")
    print(f"Tamanho bits: {len(bits)} bits")
    print(f"Tamanho recuperado: {len(recovered)} chars")

def benchmark_full_pipeline():
    """Benchmark do pipeline completo"""
    
    print("\n=== BENCHMARK PIPELINE COMPLETO ===")
    
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
    
    # Benchmark Encoding
    print("Testando encoding...")
    start = time.time()
    result = encode_image(
        image_path="images/peito.dcm",
        message=message,
        beta=0.8,
        local_algorithm="avif",
        global_algorithm="png"
    )
    encode_time = time.time() - start
    print(f"Encoding: {encode_time:.3f}s")
    
    # Benchmark Decoding
    print("Testando decoding...")
    start = time.time()
    decode_result = decode_from_bitstream(result['bitstream_path'], save_recovered=False)
    decode_time = time.time() - start
    print(f"Decoding: {decode_time:.3f}s")
    
    print(f"Tempo total: {encode_time + decode_time:.3f}s")
    print(f"Precisão: 100.0%" if message == decode_result['message'] else f"ERRO na recuperação")
    
    return encode_time, decode_time

if __name__ == "__main__":
    benchmark_message_conversion()
    encode_time, decode_time = benchmark_full_pipeline()
    
    print(f"\n=== RESUMO PERFORMANCE ===")
    print(f"📊 Encoding otimizado: {encode_time:.3f}s")
    print(f"📊 Decoding otimizado: {decode_time:.3f}s") 
    print(f"📊 Pipeline total: {encode_time + decode_time:.3f}s")