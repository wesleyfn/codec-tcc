import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- 1. Configuração Global e Estilo ---
def setup_style():
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # CORES: JPEG XL Vermelho (#D62728), Outros Cinza
    custom_codec_colors = {
        'JPEG XL': '#D62728',
        'JPEG-LS': '#7F7F7F',
        'JPEG 2000': '#C7C7C7'
    }
    
    return {
        'codec_palette': custom_codec_colors
    }

PALETTES = setup_style()

def load_data():
    possible_paths = ['tcc_results/results_sequential.csv', 'results_sequential.csv']
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Dados carregados de: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        print("ERRO: CSV não encontrado.")
        return None

    codec_map = {'jxl': 'JPEG XL', 'jls': 'JPEG-LS', 'j2k': 'JPEG 2000'}
    if 'Codec' in df.columns:
        df['Codec'] = df['Codec'].map(lambda x: codec_map.get(x, x))
    return df

def add_labels_horizontal(ax, fmt='%.3f'):
    """Adiciona rótulos em barras horizontais."""
    for c in ax.containers:
        # Tenta colocar o texto dentro da barra (branco/negrito)
        # Se a barra for muito curta, o matplotlib ajusta ou você pode forçar 'edge'
        ax.bar_label(c, fmt=fmt, label_type='edge', padding=3, color='black', fontsize=9, weight='normal')

def plot_time_seconds_horizontal(df):
    """
    Fig 5: Tempo em Segundos (Barras Horizontais)
    """
    df_plot = df[df['Beta'] == 0.2].copy()
    
    # Conversão: ms/MB -> s/MB
    df_plot['Encoding_Time_s'] = df_plot['Encoding_Speed_ms_MB'] / 1000.0
    df_plot['Decoding_Time_s'] = df_plot['Decoding_Speed_ms_MB'] / 1000.0
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # Mais largo para barras horizontais
    order = ['CT', 'DX', 'MG']
    hue_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000']
    
    # --- GRÁFICO 1: CODIFICAÇÃO (Esquerda) ---
    sns.barplot(
        data=df_plot, 
        y='Modality',           # Y é a Categoria (Horizontal)
        x='Encoding_Time_s',    # X é o Valor
        hue='Codec', 
        order=order, 
        hue_order=hue_order, 
        ax=axes[0], 
        palette=PALETTES['codec_palette'], 
        errorbar=('ci', 95), 
        capsize=.05,
        orient='h'              # Força orientação horizontal
    )
    
    axes[0].set_title('(a) Tempo de Codificação (s/MB)', fontsize=12, pad=10)
    axes[0].set_xlabel('Segundos (Menor é Melhor)', fontsize=11)
    axes[0].set_ylabel('Modalidade', fontsize=11)
    axes[0].legend_.remove() # Remove legenda do primeiro para não duplicar
    add_labels_horizontal(axes[0], fmt='%.3f')

    # --- GRÁFICO 2: DECODIFICAÇÃO (Direita) ---
    sns.barplot(
        data=df_plot, 
        y='Modality', 
        x='Decoding_Time_s', 
        hue='Codec', 
        order=order, 
        hue_order=hue_order, 
        ax=axes[1], 
        palette=PALETTES['codec_palette'], 
        errorbar=('ci', 95), 
        capsize=.05,
        orient='h'
    )
    
    axes[1].set_title('(b) Tempo de Decodificação (s/MB)', fontsize=12, pad=10)
    axes[1].set_xlabel('Segundos (Menor é Melhor)', fontsize=11)
    axes[1].set_ylabel('') # Remove label Y do segundo gráfico para limpar
    axes[1].set_yticks([]) # Opcional: Remove os nomes das modalidades do gráfico da direita se estiverem alinhados
    
    # Legenda apenas no gráfico da direita
    axes[1].legend(loc='lower right', title='Codec', fontsize=10)
    add_labels_horizontal(axes[1], fmt='%.3f')
    
    plt.tight_layout()
    plt.savefig('fig_5_tempo_segundos_horizontal.png')
    print("Salvo: fig_5_tempo_segundos_horizontal.png")

# --- MANTENDO OS OUTROS GRÁFICOS (Versão Resumida) ---
# Você pode manter suas outras funções aqui (plot_efficiency, etc.)

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_time_seconds_horizontal(df)
        print("\nGráfico horizontal de tempo (segundos) gerado com sucesso!")