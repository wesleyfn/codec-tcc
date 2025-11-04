import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Carregar os dados
df = pd.read_csv('tcc_results/results_parallel.csv')

# Converter tamanhos para MB para melhor legibilidade
df['Original_Size_MB'] = df['Original_Size_Bytes'] / (1024 * 1024)
df['Final_Size_MB'] = df['Final_Bin_Size_Bytes'] / (1024 * 1024)
df['Stego_Capacity_MB'] = df['Stego_Capacity_Bits'] / (8 * 1024 * 1024)

# Criar visualizações
plt.style.use('default')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Comparação de CR por Codec e Beta
cr_comparison = df.groupby(['Parameter_Codec', 'Parameter_Beta'])['CR'].mean().unstack()
cr_comparison.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
axes[0,0].set_title('Taxa de Compressão (CR) por Codec')
axes[0,0].set_ylabel('CR')
axes[0,0].grid(True, alpha=0.3)
axes[0,0].legend(title='Beta')

# 2. Comparação de PSNR por Codec e Beta
psnr_comparison = df.groupby(['Parameter_Codec', 'Parameter_Beta'])['Stego_Image_PSNR_dB'].mean().unstack()
psnr_comparison.plot(kind='bar', ax=axes[0,1], color=['skyblue', 'lightcoral'])
axes[0,1].set_title('Qualidade (PSNR) por Codec')
axes[0,1].set_ylabel('PSNR (dB)')
axes[0,1].grid(True, alpha=0.3)
axes[0,1].legend(title='Beta')

# 3. Comparação de Bpp por Codec e Beta
bpp_comparison = df.groupby(['Parameter_Codec', 'Parameter_Beta'])['Bpp'].mean().unstack()
bpp_comparison.plot(kind='bar', ax=axes[0,2], color=['skyblue', 'lightcoral'])
axes[0,2].set_title('Bits por Pixel (Bpp) por Codec')
axes[0,2].set_ylabel('Bpp')
axes[0,2].grid(True, alpha=0.3)
axes[0,2].legend(title='Beta')

# 4. Tempos de Codificação
encode_time = df.groupby(['Parameter_Codec', 'Parameter_Beta'])['Encoding_Time_s'].mean().unstack()
encode_time.plot(kind='bar', ax=axes[1,0], color=['skyblue', 'lightcoral'])
axes[1,0].set_title('Tempo de Codificação por Codec')
axes[1,0].set_ylabel('Tempo (s)')
axes[1,0].grid(True, alpha=0.3)
axes[1,0].legend(title='Beta')

# 5. Tempos de Decodificação
decode_time = df.groupby(['Parameter_Codec', 'Parameter_Beta'])['Decoding_Time_s'].mean().unstack()
decode_time.plot(kind='bar', ax=axes[1,1], color=['skyblue', 'lightcoral'])
axes[1,1].set_title('Tempo de Decodificação por Codec')
axes[1,1].set_ylabel('Tempo (s)')
axes[1,1].grid(True, alpha=0.3)
axes[1,1].legend(title='Beta')

# 6. SSIM por Codec
ssim_comparison = df.groupby(['Parameter_Codec', 'Parameter_Beta'])['Stego_Image_SSIM'].mean().unstack()
ssim_comparison.plot(kind='bar', ax=axes[1,2], color=['skyblue', 'lightcoral'])
axes[1,2].set_title('Similaridade Estrutural (SSIM) por Codec')
axes[1,2].set_ylabel('SSIM')
axes[1,2].grid(True, alpha=0.3)
axes[1,2].legend(title='Beta')

plt.tight_layout()
plt.show()

# Gráfico adicional: Comparação por Modalidade
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# CR por Modalidade
modalidade_cr = df.groupby(['Modality', 'Parameter_Codec'])['CR'].mean().unstack()
modalidade_cr.plot(kind='bar', ax=axes[0])
axes[0].set_title('CR por Modalidade e Codec')
axes[0].set_ylabel('CR')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# PSNR por Modalidade
modalidade_psnr = df.groupby(['Modality', 'Parameter_Codec'])['Stego_Image_PSNR_dB'].mean().unstack()
modalidade_psnr.plot(kind='bar', ax=axes[1])
axes[1].set_title('PSNR por Modalidade e Codec')
axes[1].set_ylabel('PSNR (dB)')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# Tamanho Final por Modalidade
modalidade_size = df.groupby(['Modality', 'Parameter_Codec'])['Final_Size_MB'].mean().unstack()
modalidade_size.plot(kind='bar', ax=axes[2])
axes[2].set_title('Tamanho Final por Modalidade e Codec')
axes[2].set_ylabel('Tamanho (MB)')
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Tabelas Resumo Detalhadas
print("=" * 80)
print("ANÁLISE COMPARATIVA DETALHADA DOS CODECS")
print("=" * 80)

# Tabela 1: Estatísticas Gerais por Codec
print("\n1. ESTATÍSTICAS GERAIS POR CODEC:")
stats_summary = df.groupby('Parameter_Codec').agg({
    'CR': ['mean', 'std'],
    'Stego_Image_PSNR_dB': ['mean', 'std'],
    'Stego_Image_SSIM': ['mean', 'std'],
    'Bpp': ['mean', 'std'],
    'Encoding_Time_s': ['mean', 'std'],
    'Decoding_Time_s': ['mean', 'std'],
    'Final_Size_MB': ['mean', 'std']
}).round(3)

print(stats_summary)

# Tabela 2: Comparação Beta 0.5 vs 0.8
print("\n2. COMPARAÇÃO BETA 0.5 vs 0.8:")
beta_comparison = df.groupby(['Parameter_Codec', 'Parameter_Beta']).agg({
    'CR': 'mean',
    'Stego_Image_PSNR_dB': 'mean',
    'Encoding_Time_s': 'mean',
    'Final_Size_MB': 'mean'
}).round(3)
print(beta_comparison)

# Tabela 3: Análise por Modalidade
print("\n3. ANÁLISE POR MODALIDADE:")
modality_stats = df.groupby(['Modality', 'Parameter_Codec']).agg({
    'CR': 'mean',
    'Stego_Image_PSNR_dB': 'mean',
    'Final_Size_MB': 'mean'
}).round(3)
print(modality_stats)

# Tabela 4: Ranking de Performance
print("\n4. RANKING DE PERFORMANCE:")

# Ranking por CR
cr_ranking = df.groupby('Parameter_Codec')['CR'].mean().sort_values(ascending=False)
print("\nRanking por Taxa de Compressão (CR):")
for i, (codec, cr) in enumerate(cr_ranking.items(), 1):
    print(f"{i}. {codec}: {cr:.3f}")

# Ranking por Velocidade de Codificação
encode_ranking = df.groupby('Parameter_Codec')['Encoding_Time_s'].mean().sort_values(ascending=True)
print("\nRanking por Velocidade de Codificação:")
for i, (codec, time) in enumerate(encode_ranking.items(), 1):
    print(f"{i}. {codec}: {time:.3f}s")

# Ranking por Velocidade de Decodificação
decode_ranking = df.groupby('Parameter_Codec')['Decoding_Time_s'].mean().sort_values(ascending=True)
print("\nRanking por Velocidade de Decodificação:")
for i, (codec, time) in enumerate(decode_ranking.items(), 1):
    print(f"{i}. {codec}: {time:.3f}s")

# Gráfico de Dispersão: CR vs PSNR
plt.figure(figsize=(12, 8))
colors = {'jxl': 'red', 'j2k': 'blue', 'jls': 'green', 'png': 'orange'}
markers = {'jxl': 'o', 'j2k': 's', 'jls': '^', 'png': 'D'}

for codec in df['Parameter_Codec'].unique():
    subset = df[df['Parameter_Codec'] == codec]
    for beta in [0.5, 0.8]:
        beta_subset = subset[subset['Parameter_Beta'] == beta]
        plt.scatter(beta_subset['CR'], beta_subset['Stego_Image_PSNR_dB'],
                  c=colors[codec], marker=markers[codec], 
                  s=60, alpha=0.7, label=f'{codec} β={beta}' if beta == 0.5 else "")

plt.xlabel('Taxa de Compressão (CR)')
plt.ylabel('PSNR (dB)')
plt.title('Relação entre Compressão e Qualidade')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Gráfico de Calor: Performance por Modalidade
heatmap_data = df.pivot_table(values='CR', 
                            index='Modality', 
                            columns='Parameter_Codec', 
                            aggfunc='mean')

plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Taxa de Compressão (CR) por Modalidade e Codec')
plt.tight_layout()
plt.show()

# Análise Final Comparativa
print("\n" + "=" * 80)
print("CONCLUSÕES E RECOMENDAÇÕES")
print("=" * 80)

# Calcular métricas consolidadas
consolidated_stats = df.groupby('Parameter_Codec').agg({
    'CR': 'mean',
    'Encoding_Time_s': 'mean', 
    'Decoding_Time_s': 'mean',
    'Stego_Image_PSNR_dB': 'mean',
    'Final_Size_MB': 'mean'
}).round(3)

print("\nPerformance Consolidada:")
print(consolidated_stats)

print("\nRecomendações Baseadas nos Resultados:")
print("• Para máxima compressão: JXL")
print("• Para velocidade de codificação: JLS") 
print("• Para velocidade de decodificação: PNG")
print("• Para equilíbrio geral: J2K")
print("• Efeito do Beta: Reduz PSNR em ~7dB com ganho mínimo em compressão")

# Executar análise estatística detalhada
print("ANÁLISE ESTATÍSTICA DETALHADA")

# Calcular diferenças entre beta 0.5 e 0.8
beta_differences = []
for codec in df['Parameter_Codec'].unique():
    beta_05 = df[(df['Parameter_Codec'] == codec) & (df['Parameter_Beta'] == 0.5)]
    beta_08 = df[(df['Parameter_Codec'] == codec) & (df['Parameter_Beta'] == 0.8)]
    
    cr_diff = beta_08['CR'].mean() - beta_05['CR'].mean()
    psnr_diff = beta_08['Stego_Image_PSNR_dB'].mean() - beta_05['Stego_Image_PSNR_dB'].mean()
    size_diff = beta_08['Final_Size_MB'].mean() - beta_05['Final_Size_MB'].mean()
    
    beta_differences.append({
        'Codec': codec,
        'ΔCR': cr_diff,
        'ΔPSNR': psnr_diff,
        'ΔSize_MB': size_diff
    })

beta_diff_df = pd.DataFrame(beta_differences)
print("\nDiferenças Beta 0.8 vs 0.5:")
print(beta_diff_df.round(4))

# Análise de eficiência por bits stored
print("\nAnálise por Profundidade de Bits:")
bits_analysis = df.groupby(['Bits_Stored', 'Parameter_Codec']).agg({
    'CR': 'mean',
    'Bpp': 'mean',
    'Stego_Image_PSNR_dB': 'mean'
}).round(3)
print(bits_analysis)