import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- Configurações Gráficas ---
plt.style.use('default')
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'figure.dpi': 150})
# ------------------------------

def generate_analysis_plots(df):
    
    # --- 1. PREPARAÇÃO DE DADOS ---
    
    # Renomear codecs
    df['Parameter_Codec'] = df['Parameter_Codec'].replace({
        'j2k': 'JPEG 2000', 
        'jls': 'JPEG-LS',
        'jxl': 'JPEG XL'
    })
    
    # Renomear e ordenar modalidades
    modality_order = ['dx_8b', 'dx_16b', 'mr_16b', 'mg_16b']
    df['Modality'] = pd.Categorical(df['Modality'], categories=modality_order, ordered=True)
    df['Modality'] = df['Modality'].cat.rename_categories({
        'dx_8b': 'DX 8-bit',
        'dx_16b': 'DX 16-bit', 
        'mr_16b': 'MR 16-bit',
        'mg_16b': 'MG 16-bit'
    })
    
    # Calcular BPP original (para eficiência)
    df['Bpp_Original'] = df['Bits_Stored']
    df['Eficiencia_Bpp'] = (df['Bpp_Original'] - df['Bpp']) / df['Bpp_Original'] * 100
    
    # Agrupamento para médias e desvios
    grouped_data = df.groupby(['Parameter_Codec', 'Modality', 'Parameter_Beta'])
    summary = grouped_data[['Bpp', 'Total_Encoding_Time_s', 'Decoding_Time_s', 'Stego_Image_PSNR_dB']].agg(['mean', 'std']).reset_index()


    # --- 2. GRÁFICO BPP VS CODEC ---

    fig, ax = plt.subplots(figsize=(10, 5))
    
    pivot_bpp = summary.pivot_table(
        index='Parameter_Codec', 
        columns=['Modality', 'Parameter_Beta'], 
        values=('Bpp', 'mean')
    )
    
    # Plotar como barras agrupadas
    pivot_bpp.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Bits por Pixel (Bpp) por Codec e Modalidade', fontsize=12, fontweight='bold')
    ax.set_xlabel('Codec', fontsize=10)
    ax.set_ylabel('Bpp Final', fontsize=10)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Modalidade (Beta)', loc='upper left', ncol=2, fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('analysis_bpp_by_codec.png')
    plt.close()
    print("Gráfico: analysis_bpp_by_codec.png gerado.")


    # --- 3. GRÁFICO TEMPO DE ENCODE VS CODEC ---

    fig, ax = plt.subplots(figsize=(10, 5))
    
    pivot_time = summary.pivot_table(
        index='Parameter_Codec', 
        columns=['Modality', 'Parameter_Beta'], 
        values=('Total_Encoding_Time_s', 'mean')
    )
    
    pivot_time.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Tempo Médio de Codificação (s)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Codec', fontsize=10)
    ax.set_ylabel('Tempo Total (s)', fontsize=10)
    ax.tick_params(axis='x', rotation=0)
    ax.legend(title='Modalidade (Beta)', loc='upper left', ncol=2, fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('analysis_time_by_codec.png')
    plt.close()
    print("Gráfico: analysis_time_by_codec.png gerado.")
    
    
    # --- 4. GRÁFICO PSNR (Qualidade) VS MODALIDADE ---
    
    # Agrupar apenas por Modalidade e Beta (PSNR não varia com o codec, pois é calculado antes da compressão)
    psnr_summary = df.groupby(['Modality', 'Parameter_Beta'])['Stego_Image_PSNR_dB'].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))
    
    psnr_summary.plot(kind='bar', ax=ax, width=0.7)
    
    ax.set_title('PSNR da Imagem Esteganografada por Modalidade', fontsize=12, fontweight='bold')
    ax.set_xlabel('Modalidade', fontsize=10)
    ax.set_ylabel('PSNR Médio (dB)', fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Beta', loc='lower right', fontsize=8)
    ax.set_ylim(40, psnr_summary.values.max() + 5) # Começa em 40 dB
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('analysis_psnr_by_modality.png')
    plt.close()
    print("Gráfico: analysis_psnr_by_modality.png gerado.")


    # --- 5. TABELAS DE RESUMO ---
    print("\n" + "="*80)
    print("--- TABELA DE BITS POR PIXEL (BPP) MÉDIO ---")
    print(summary.pivot_table(index='Parameter_Codec', columns=['Modality', 'Parameter_Beta'], values=('Bpp', 'mean'), aggfunc='mean').round(3).to_markdown())
    
    print("\n" + "="*80)
    print("--- TABELA DE TEMPO MÉDIO DE CODIFICAÇÃO (s) ---")
    print(summary.pivot_table(index='Parameter_Codec', columns=['Modality', 'Parameter_Beta'], values=('Total_Encoding_Time_s', 'mean'), aggfunc='mean').round(3).to_markdown())
    
    print("\n" + "="*80)
    print("--- TABELA DE PSNR MÉDIO (dB) ---")
    print(psnr_summary.round(2).to_markdown())
    print("="*80 + "\n")


if __name__ == '__main__':
    # A localização do arquivo CSV deve ser ajustada conforme a estrutura de pastas do seu projeto
    try:
        df = pd.read_csv('tcc_results/results_sequential.csv')
    except FileNotFoundError:
        print("ERRO: Arquivo 'tcc_results/results_sequential.csv' não encontrado. Execute 'main_tcc.py' primeiro.")
        exit()
        
    generate_analysis_plots(df)