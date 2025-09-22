import sys
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from encode import encode_image
import pandas as pd
import time
import os
from pathlib import Path

def benchmark_compression_algorithms():
    """
    Testa todas as combinações de algoritmos de compressão e gera gráficos comparativos
    """
    
    print("🧪 BENCHMARK: Comparação de Algoritmos de Compressão")
    print("="*80)
    
    # Algoritmos de compressão disponíveis
    algorithms = ['png', 'avif', 'jpegxl', 'jpegls']
    
    # Configurações de teste
    test_configs = [
        {'s': 1, 'image': 'images/peito.dcm', 'name': 'Peito s=1'},
        {'s': 2, 'image': 'images/peito.dcm', 'name': 'Peito s=2'},
        {'s': 3, 'image': 'images/peito.dcm', 'name': 'Peito s=3'},
    ]
    
    # Mensagem de teste
    message = "Teste de benchmark para comparar algoritmos de compressão"
    
    # Resultados
    results = []
    failed_combinations = []
    
    total_tests = len(algorithms) * len(algorithms) * len(test_configs)
    current_test = 0
    
    print(f"📊 Executando {total_tests} testes...")
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"🔬 Testando: {config['name']}")
        print(f"{'='*60}")
        
        for local_alg in algorithms:
            for global_alg in algorithms:
                current_test += 1
                combo_name = f"{local_alg}+{global_alg}"
                
                print(f"[{current_test:2d}/{total_tests}] {combo_name:<20} ", end="")
                
                try:
                    # Medir tempo de encoding
                    start_time = time.time()
                    
                    result = encode_image(
                        image_path=config['image'],
                        message=message,
                        s=config['s'],
                        local_algorithm=local_alg,
                        global_algorithm=global_alg,
                        threshold=2
                    )
                    
                    encode_time = time.time() - start_time
                    
                    # Extrair métricas dos dados comprimidos
                    local_compressed = result.get('local_compressed', {})
                    global_compressed = result.get('global_compressed', {})
                    
                    # Extrair taxas de compressão dos dados comprimidos
                    local_ratio = local_compressed.get('compression_ratio', 1.0) * 100  # Converter para %
                    global_ratio = global_compressed.get('compression_ratio', 1.0) * 100  # Converter para %
                    
                    # Extrair outras métricas
                    bits_used = result.get('bits_used', 0)
                    
                    # Calcular tamanho total aproximado
                    local_size = len(local_compressed.get('data', b'')) if 'data' in local_compressed else 0
                    global_size = len(global_compressed.get('data', b'')) if 'data' in global_compressed else 0
                    total_size = local_size + global_size
                    
                    # Calcular taxa de compressão combinada ponderada pelo tamanho
                    if local_ratio > 0 and global_ratio > 0:
                        combined_ratio = (local_ratio + global_ratio) / 2
                    else:
                        combined_ratio = max(local_ratio, global_ratio, 1.0)
                    
                    results.append({
                        'config': config['name'],
                        's': config['s'],
                        'local_algorithm': local_alg,
                        'global_algorithm': global_alg,
                        'combination': combo_name,
                        'local_ratio': local_ratio,
                        'global_ratio': global_ratio,
                        'combined_ratio': combined_ratio,
                        'total_size_mb': total_size / (1024*1024),
                        'bits_used': bits_used,
                        'encode_time': encode_time,
                        'efficiency': (100 - combined_ratio) / encode_time if encode_time > 0 else 0
                    })
                    
                    print(f"✅ L:{local_ratio:5.1f}% G:{global_ratio:5.1f}% T:{encode_time:4.1f}s")
                    
                except Exception as e:
                    failed_combinations.append({
                        'config': config['name'],
                        'combination': combo_name,
                        'error': str(e)
                    })
                    print(f"❌ ERRO: {str(e)[:50]}")
    
    if not results:
        print("❌ Nenhum teste foi bem-sucedido!")
        return
    
    # Converter para DataFrame
    df = pd.DataFrame(results)
    
    # Debug: Mostrar estatísticas dos dados
    print(f"\n📊 ESTATÍSTICAS DOS DADOS:")
    print(f"Total de resultados: {len(df)}")
    print(f"Taxa compressão - Min: {df['combined_ratio'].min():.1f}%, Max: {df['combined_ratio'].max():.1f}%")
    print(f"Tempo encoding - Min: {df['encode_time'].min():.3f}s, Max: {df['encode_time'].max():.3f}s")
    print(f"Eficiência - Min: {df['efficiency'].min():.1f}, Max: {df['efficiency'].max():.1f}")
    print(f"Valores únicos de s: {sorted(df['s'].unique())}")
    
    # Verificar se há zeros problemáticos
    zero_ratios = (df['combined_ratio'] == 0).sum()
    zero_times = (df['encode_time'] == 0).sum()
    zero_efficiency = (df['efficiency'] == 0).sum()
    
    if zero_ratios > 0:
        print(f"⚠️  AVISO: {zero_ratios} resultados com taxa de compressão = 0%")
    if zero_times > 0:
        print(f"⚠️  AVISO: {zero_times} resultados com tempo = 0s")
    if zero_efficiency > 0:
        print(f"⚠️  AVISO: {zero_efficiency} resultados com eficiência = 0")
    
    # Mostrar falhas se houver
    if failed_combinations:
        print(f"\n⚠️  {len(failed_combinations)} combinações falharam:")
        for fail in failed_combinations[:5]:  # Mostrar apenas primeiras 5
            print(f"   {fail['combination']} ({fail['config']}): {fail['error'][:60]}")
    
    print(f"\n📊 Gerando gráficos com {len(results)} resultados...")
    
    # Criar gráficos
    create_compression_charts(df)
    
    # Mostrar top 5 combinações
    print_top_combinations(df)

def create_compression_charts(df):
    """Cria gráficos de comparação dos algoritmos - APENAS 3 GRÁFICOS PRINCIPAIS"""
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Criar figura com 3 subplots em linha
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Heatmap de taxa de compressão combinada
    plt.subplot(1, 3, 1)
    pivot_combined = df.pivot_table(values='combined_ratio', 
                                   index='local_algorithm', 
                                   columns='global_algorithm', 
                                   aggfunc='mean')
    sns.heatmap(pivot_combined, annot=True, fmt='.2f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Taxa Compressão'}, square=True)
    plt.title('Taxa de Compressão Combinada\n(Menor = Melhor)', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritmo Global', fontsize=12)
    plt.ylabel('Algoritmo Local', fontsize=12)
    
    # 2. Heatmap de tempo de encoding
    plt.subplot(1, 3, 2)
    pivot_time = df.pivot_table(values='encode_time', 
                               index='local_algorithm', 
                               columns='global_algorithm', 
                               aggfunc='mean')
    sns.heatmap(pivot_time, annot=True, fmt='.2f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Tempo (s)'}, square=True)
    plt.title('Tempo de Encoding\n(Menor = Melhor)', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritmo Global', fontsize=12)
    plt.ylabel('Algoritmo Local', fontsize=12)
    
    # 3. Heatmap de eficiência (compressão/tempo)
    plt.subplot(1, 3, 3)
    pivot_eff = df.pivot_table(values='efficiency', 
                              index='local_algorithm', 
                              columns='global_algorithm', 
                              aggfunc='mean')
    sns.heatmap(pivot_eff, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': 'Eficiência'}, square=True)
    plt.title('Eficiência (Compressão/Tempo)\n(Maior = Melhor)', fontsize=14, fontweight='bold')
    plt.xlabel('Algoritmo Global', fontsize=12)
    plt.ylabel('Algoritmo Local', fontsize=12)
    
    plt.tight_layout(pad=3.0)
    
    # Salvar gráfico
    output_path = 'compression_benchmark.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {output_path}")
    
    plt.show()

def print_top_combinations(df):
    """Mostra as melhores combinações"""
    
    print(f"\n{'='*80}")
    print(f"🏆 MELHORES COMBINAÇÕES")
    print(f"{'='*80}")
    
    # Top 5 melhor compressão
    print(f"\n🎯 TOP 5 MELHOR COMPRESSÃO (menor %):")
    top_compression = df.nsmallest(5, 'combined_ratio')
    for i, (_, row) in enumerate(top_compression.iterrows(), 1):
        print(f"  {i}. {row['combination']:<15} (s={row['s']}) - "
              f"{row['combined_ratio']:5.1f}% em {row['encode_time']:4.1f}s")
    
    # Top 5 mais rápido
    print(f"\n⚡ TOP 5 MAIS RÁPIDO:")
    top_speed = df.nsmallest(5, 'encode_time')
    for i, (_, row) in enumerate(top_speed.iterrows(), 1):
        print(f"  {i}. {row['combination']:<15} (s={row['s']}) - "
              f"{row['encode_time']:4.1f}s com {row['combined_ratio']:5.1f}%")
    
    # Top 5 melhor eficiência
    print(f"\n🚀 TOP 5 MELHOR EFICIÊNCIA (compressão/tempo):")
    top_efficiency = df.nlargest(5, 'efficiency')
    for i, (_, row) in enumerate(top_efficiency.iterrows(), 1):
        print(f"  {i}. {row['combination']:<15} (s={row['s']}) - "
              f"Eficiência: {row['efficiency']:5.1f}")
    
    # Recomendação geral
    print(f"\n💡 RECOMENDAÇÕES:")
    
    best_overall = df.loc[df['efficiency'].idxmax()]
    print(f"  🥇 Melhor geral: {best_overall['combination']} (s={best_overall['s']})")
    print(f"     Compressão: {best_overall['combined_ratio']:.1f}%, "
          f"Tempo: {best_overall['encode_time']:.1f}s, "
          f"Eficiência: {best_overall['efficiency']:.1f}")
    
    best_compression = df.loc[df['combined_ratio'].idxmin()]
    print(f"  🎯 Melhor compressão: {best_compression['combination']} (s={best_compression['s']})")
    print(f"     Taxa: {best_compression['combined_ratio']:.1f}%")
    
    fastest = df.loc[df['encode_time'].idxmin()]
    print(f"  ⚡ Mais rápido: {fastest['combination']} (s={fastest['s']})")
    print(f"     Tempo: {fastest['encode_time']:.1f}s")

if __name__ == "__main__":
    # Verificar se matplotlib está disponível
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError as e:
        print(f"❌ Dependência não encontrada: {e}")
        print("💡 Instale com: pip install matplotlib seaborn pandas")
        sys.exit(1)
    
    benchmark_compression_algorithms()