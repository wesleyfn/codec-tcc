import pickle
import numpy as np

# Carregar dados de steganografia para análise
print("🔍 Analisando dados de steganografia...")

stego_data_path = "output/peito/peito_stego_data.bin"

try:
    with open(stego_data_path, 'rb') as f:
        stego_data = pickle.load(f)
    
    print(f"✅ Dados carregados com sucesso!")
    
    # Analisar componente local
    local_comp = stego_data['local_component']
    print(f"\n📊 COMPONENTE LOCAL:")
    print(f"   Shape: {local_comp['image'].shape}")
    print(f"   Dtype: {local_comp['image'].dtype}")
    print(f"   Range: [{local_comp['image'].min()}, {local_comp['image'].max()}]")
    print(f"   Original dtype: {local_comp['original_dtype']}")
    print(f"   Bits per pixel: {local_comp['nbits']}")
    
    # Analisar bitmap
    bitmap = local_comp['bitmap']
    marked_positions = np.sum(bitmap == 255)
    total_positions = bitmap.size
    print(f"   Bitmap - Posições marcadas: {marked_positions} / {total_positions}")
    print(f"   Percentual marcado: {marked_positions/total_positions*100:.2f}%")
    
    # Analisar threshold usado no encoding
    if 'threshold' in local_comp:
        print(f"   Threshold usado: {local_comp['threshold']}")
    else:
        print(f"   ⚠️  Threshold não salvo nos dados")
    
    # Analisar estatísticas se disponíveis
    if 'stats' in local_comp:
        stats = local_comp['stats']
        print(f"   Stats encoding:")
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                print(f"      {key}: {value}")
    
    # Analisar mensagem
    message = stego_data['message']
    print(f"\n📝 MENSAGEM:")
    print(f"   Tamanho: {len(message)} caracteres")
    print(f"   Primeiros 50 chars: '{message[:50]}...'")
    
    # Calcular bits necessários
    message_bits = len(message) * 8  # assumindo ASCII
    eof_bits = 32  # EOF marker típico
    total_bits = message_bits + eof_bits
    
    print(f"   Bits necessários: {total_bits} (msg: {message_bits} + EOF: {eof_bits})")
    print(f"   Posições disponíveis: {marked_positions}")
    print(f"   Taxa de ocupação: {total_bits/marked_positions*100:.2f}%")
    
    if total_bits > marked_positions:
        print(f"   ❌ PROBLEMA: Mais bits necessários que posições disponíveis!")
    
except Exception as e:
    print(f"❌ Erro ao carregar dados: {e}")
    print(f"   Tipo: {type(e).__name__}")