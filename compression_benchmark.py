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
    
    # Configurações de teste - s de 1 até 7
    test_configs = [
        {'s': 1, 'image': 'images/torax.dcm', 'name': 'Torax s=1'},
        {'s': 2, 'image': 'images/torax.dcm', 'name': 'Torax s=2'},
        {'s': 3, 'image': 'images/torax.dcm', 'name': 'Torax s=3'},
        {'s': 4, 'image': 'images/torax.dcm', 'name': 'Torax s=4'},
        {'s': 5, 'image': 'images/torax.dcm', 'name': 'Torax s=5'},
        {'s': 6, 'image': 'images/torax.dcm', 'name': 'Torax s=6'},
        {'s': 7, 'image': 'images/torax.dcm', 'name': 'Torax s=7'},
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
    """Cria gráficos detalhados mostrando performance por valor de s"""
    
    # Configurar estilo
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Criar figura com múltiplos subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Gráfico de linha: Taxa de compressão por s para cada combinação
    plt.subplot(3, 2, 1)
    for combination in df['combination'].unique():
        combo_data = df[df['combination'] == combination].sort_values('s')
        plt.plot(combo_data['s'], combo_data['combined_ratio'], 
                marker='o', linewidth=2, label=combination, markersize=6)
    
    plt.xlabel('Parâmetro s', fontweight='bold')
    plt.ylabel('Taxa de Compressão (%)', fontweight='bold')
    plt.title('Taxa de Compressão por Parâmetro s\n(Menor = Melhor)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(df['s'].unique()))
    
    # 2. Gráfico de linha: Tempo de encoding por s
    plt.subplot(3, 2, 2)
    for combination in df['combination'].unique():
        combo_data = df[df['combination'] == combination].sort_values('s')
        plt.plot(combo_data['s'], combo_data['encode_time'], 
                marker='s', linewidth=2, label=combination, markersize=6)
    
    plt.xlabel('Parâmetro s', fontweight='bold')
    plt.ylabel('Tempo de Encoding (s)', fontweight='bold')
    plt.title('Tempo de Encoding por Parâmetro s\n(Menor = Melhor)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(df['s'].unique()))
    
    # 3. Gráfico de linha: Eficiência por s
    plt.subplot(3, 2, 3)
    for combination in df['combination'].unique():
        combo_data = df[df['combination'] == combination].sort_values('s')
        plt.plot(combo_data['s'], combo_data['efficiency'], 
                marker='^', linewidth=2, label=combination, markersize=6)
    
    plt.xlabel('Parâmetro s', fontweight='bold')
    plt.ylabel('Eficiência (Compressão/Tempo)', fontweight='bold')
    plt.title('Eficiência por Parâmetro s\n(Maior = Melhor)', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(sorted(df['s'].unique()))
    
    # 4. Heatmap: Melhor combinação por s
    plt.subplot(3, 2, 4)
    # Encontrar melhor combinação para cada s
    best_by_s = df.loc[df.groupby('s')['efficiency'].idxmax()]
    best_matrix = best_by_s.pivot_table(values='efficiency', 
                                       index='combination', 
                                       columns='s', 
                                       fill_value=0)
    
    sns.heatmap(best_matrix, annot=True, fmt='.1f', cmap='RdYlGn',
                cbar_kws={'label': 'Eficiência'})
    plt.title('Melhor Método por Parâmetro s\n(Eficiência)', fontsize=14, fontweight='bold')
    plt.xlabel('Parâmetro s', fontweight='bold')
    plt.ylabel('Combinação', fontweight='bold')
    
    # 5. Gráfico de barras: Ranking médio por combinação
    plt.subplot(3, 2, 5)
    avg_performance = df.groupby('combination').agg({
        'combined_ratio': 'mean',
        'encode_time': 'mean', 
        'efficiency': 'mean'
    }).round(2)
    
    # Ordenar por eficiência
    avg_performance = avg_performance.sort_values('efficiency', ascending=False)
    
    x_pos = range(len(avg_performance))
    bars = plt.bar(x_pos, avg_performance['efficiency'], 
                   color=sns.color_palette("husl", len(avg_performance)))
    
    plt.xlabel('Combinação de Algoritmos', fontweight='bold')
    plt.ylabel('Eficiência Média', fontweight='bold')
    plt.title('Ranking Geral por Eficiência\n(Média de todos os s)', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, avg_performance.index, rotation=45, ha='right')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 6. Tabela de detalhes por s
    plt.subplot(3, 2, 6)
    plt.axis('off')
    
    # Criar tabela com melhores resultados por s
    summary_data = []
    for s_val in sorted(df['s'].unique()):
        s_data = df[df['s'] == s_val]
        best_compression = s_data.loc[s_data['combined_ratio'].idxmin()]
        best_speed = s_data.loc[s_data['encode_time'].idxmin()]
        best_efficiency = s_data.loc[s_data['efficiency'].idxmax()]
        
        summary_data.append([
            f"s={s_val}",
            f"{best_compression['combination']}\n({best_compression['combined_ratio']:.1f}%)",
            f"{best_speed['combination']}\n({best_speed['encode_time']:.1f}s)",
            f"{best_efficiency['combination']}\n({best_efficiency['efficiency']:.1f})"
        ])
    
    table = plt.table(cellText=summary_data,
                     colLabels=['S', 'Melhor Compressão', 'Mais Rápido', 'Mais Eficiente'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Colorir header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Resumo dos Melhores Métodos por s', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout(pad=3.0)
    
    # Salvar gráfico
    output_path = 'compression_benchmark_detailed.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico detalhado salvo em: {output_path}")
    
    plt.show()

def print_top_combinations(df):
    """Mostra as melhores combinações com análise detalhada por s"""
    
    print(f"\n{'='*100}")
    print(f"🏆 ANÁLISE DETALHADA POR PARÂMETRO S")
    print(f"{'='*100}")
    
    # Análise por cada valor de s
    for s_val in sorted(df['s'].unique()):
        print(f"\n{'─'*60}")
        print(f"📊 RESULTADOS PARA s={s_val}")
        print(f"{'─'*60}")
        
        s_data = df[df['s'] == s_val].copy()
        s_data = s_data.sort_values('efficiency', ascending=False)
        
        print(f"\n🥇 RANKING POR EFICIÊNCIA (s={s_val}):")
        for i, (_, row) in enumerate(s_data.iterrows(), 1):
            print(f"  {i:2d}. {row['combination']:<12} - "
                  f"Compressão: {row['combined_ratio']:5.1f}%, "
                  f"Tempo: {row['encode_time']:5.1f}s, "
                  f"Eficiência: {row['efficiency']:5.1f}")
        
        # Destaques para s específico
        best_comp = s_data.loc[s_data['combined_ratio'].idxmin()]
        fastest = s_data.loc[s_data['encode_time'].idxmin()]
        most_eff = s_data.loc[s_data['efficiency'].idxmax()]
        
        print(f"\n   🎯 Melhor compressão: {best_comp['combination']} ({best_comp['combined_ratio']:.1f}%)")
        print(f"   ⚡ Mais rápido: {fastest['combination']} ({fastest['encode_time']:.1f}s)")
        print(f"   🚀 Mais eficiente: {most_eff['combination']} ({most_eff['efficiency']:.1f})")
    
    # Resumo geral
    print(f"\n{'='*100}")
    print(f"📈 TENDÊNCIAS GERAIS")
    print(f"{'='*100}")
    
    # Melhor combinação por critério geral
    print(f"\n🏆 CAMPEÕES GERAIS (considerando todos os s):")
    
    best_overall = df.loc[df['efficiency'].idxmax()]
    print(f"  🥇 Mais eficiente geral: {best_overall['combination']} (s={best_overall['s']})")
    print(f"     Eficiência: {best_overall['efficiency']:.1f}, "
          f"Compressão: {best_overall['combined_ratio']:.1f}%, "
          f"Tempo: {best_overall['encode_time']:.1f}s")
    
    best_compression = df.loc[df['combined_ratio'].idxmin()]
    print(f"  🎯 Melhor compressão geral: {best_compression['combination']} (s={best_compression['s']})")
    print(f"     Compressão: {best_compression['combined_ratio']:.1f}%")
    
    fastest = df.loc[df['encode_time'].idxmin()]
    print(f"  ⚡ Mais rápido geral: {fastest['combination']} (s={fastest['s']})")
    print(f"     Tempo: {fastest['encode_time']:.1f}s")
    
    # Análise de comportamento por s
    print(f"\n📊 COMPORTAMENTO POR PARÂMETRO S:")
    for combination in df['combination'].unique():
        combo_data = df[df['combination'] == combination].sort_values('s')
        if len(combo_data) > 1:
            # Calcular tendência
            comp_trend = "↑" if combo_data['combined_ratio'].iloc[-1] > combo_data['combined_ratio'].iloc[0] else "↓"
            time_trend = "↑" if combo_data['encode_time'].iloc[-1] > combo_data['encode_time'].iloc[0] else "↓"
            eff_trend = "↑" if combo_data['efficiency'].iloc[-1] > combo_data['efficiency'].iloc[0] else "↓"
            
            print(f"  {combination:<12}: "
                  f"Compressão {comp_trend} "
                  f"({combo_data['combined_ratio'].iloc[0]:.1f}% → {combo_data['combined_ratio'].iloc[-1]:.1f}%), "
                  f"Tempo {time_trend} "
                  f"({combo_data['encode_time'].iloc[0]:.1f}s → {combo_data['encode_time'].iloc[-1]:.1f}s), "
                  f"Eficiência {eff_trend}")
    
    # Recomendações
    print(f"\n💡 RECOMENDAÇÕES:")
    
    # Média por combinação
    avg_performance = df.groupby('combination').agg({
        'combined_ratio': 'mean',
        'encode_time': 'mean', 
        'efficiency': 'mean'
    }).round(2)
    
    best_avg = avg_performance.loc[avg_performance['efficiency'].idxmax()]
    print(f"  🌟 Para uso geral: {avg_performance['efficiency'].idxmax()}")
    print(f"     Eficiência média: {best_avg['efficiency']:.1f}")
    
    if 'jpegls' in avg_performance.index:
        jpegls_combos = [combo for combo in avg_performance.index if 'jpegls' in combo]
        if jpegls_combos:
            best_jpegls = max(jpegls_combos, key=lambda x: avg_performance.loc[x, 'efficiency'])
            print(f"  🏥 Para imagens médicas: {best_jpegls}")
            print(f"     Eficiência: {avg_performance.loc[best_jpegls, 'efficiency']:.1f}")
    
    fastest_avg = avg_performance.loc[avg_performance['encode_time'].idxmin()]
    print(f"  ⚡ Para processamento rápido: {avg_performance['encode_time'].idxmin()}")
    print(f"     Tempo médio: {fastest_avg['encode_time']:.1f}s")

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