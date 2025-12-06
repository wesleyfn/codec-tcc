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
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    return {
        'codec_palette': 'Set2',
        'beta_palette': 'viridis'
    }

PALETTES = setup_style()

def load_data():
    possible_paths = [
        'tcc_results/results_sequential.csv',
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Dados carregados de: {path}")
            df = pd.read_csv(path)
            break
            
    if df is None:
        print("ERRO: 'results_sequential.csv' não encontrado.")
        return None

    codec_map = {'jxl': 'JPEG XL', 'jls': 'JPEG-LS', 'j2k': 'JPEG 2000'}
    if 'Codec' in df.columns:
        df['Codec'] = df['Codec'].map(lambda x: codec_map.get(x, x))
        
    return df

# --- 2. Helper de Labels Inteligente ---
def add_labels(ax, fmt='%.2f', centered=True):
    """
    Adiciona valores nas barras.
    centered=True -> No meio (Branco, Negrito) -> Para gráficos com desvio padrão
    centered=False -> No topo (Preto, Normal) -> Para gráficos exatos
    """
    for c in ax.containers:
        if centered:
            ax.bar_label(c, fmt=fmt, label_type='center', color='white', weight='bold', fontsize=10)
        else:
            ax.bar_label(c, fmt=fmt, label_type='edge', padding=2, color='black', fontsize=10)

# --- 3. Funções de Plotagem ---

def plot_efficiency(df):
    """Fig 1: Bpp e CR (Com Desvio -> Centro)"""
    df_plot = df[df['Beta'] == 0.2].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    order = ['CT', 'DX', 'MG']
    hue_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000']
    
    # Bpp
    sns.barplot(data=df_plot, x='Modality', y='Bpp', hue='Codec', 
                order=order, hue_order=hue_order, ax=axes[0], 
                palette=PALETTES['codec_palette'], errorbar=('ci', 95), capsize=.05, alpha=0.8)
    axes[0].set_title('(a) Bits por Pixel (bpp)')
    axes[0].set_ylabel('Bits por Pixel (bpp)')
    axes[0].legend(loc='upper right', frameon=True, fontsize=10)
    axes[0].set_xlabel('Modalidade')
    add_labels(axes[0], fmt='%.2f', centered=True) # Com desvio -> Centro

    
    # CR
    sns.barplot(data=df_plot, x='Modality', y='CR', hue='Codec', 
                order=order, hue_order=hue_order, ax=axes[1], 
                palette=PALETTES['codec_palette'], errorbar=('ci', 95), capsize=.05, alpha=0.8)
    axes[1].set_title('(b) Taxa de Compressão (CR)')
    axes[1].set_ylabel('Taxa de Compressão (CR)')
    axes[1].legend(loc='upper left', frameon=True, fontsize=10)
    axes[1].set_xlabel('Modalidade')
    add_labels(axes[1], fmt='%.1f', centered=True) # Com desvio -> Centro
    
    plt.tight_layout()
    plt.savefig('fig_1_eficiencia_compressao.png')
    print("Salvo: fig_1_eficiencia_compressao.png")

def plot_performance(df):
    """Fig 2: Velocidade de Codificação (Com Desvio -> Centro)"""
    df_plot = df[df['Beta'] == 0.2].copy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    order = ['CT', 'DX', 'MG']
    hue_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000']

    # Encoding
    sns.barplot(data=df_plot, y='Modality', x='Encoding_Speed_ms_MB', hue='Codec', 
                order=order, hue_order=hue_order, ax=axes[0], 
                palette=PALETTES['codec_palette'], errorbar=('ci', 95), capsize=.05, alpha=0.8)
    axes[0].set_title('(a) Velocidade de Codificação por Modalidade')
    axes[0].set_xlabel('Tempo (ms/MB)')
    axes[0].legend(loc='lower right', frameon=True, fontsize=10)
    axes[0].set_ylabel('Modalidade')
    add_labels(axes[0], fmt='%.0f', centered=True)


    # Decoding
    sns.barplot(data=df_plot, y='Modality', x='Decoding_Speed_ms_MB', hue='Codec', 
                order=order, hue_order=hue_order, ax=axes[1], 
                palette=PALETTES['codec_palette'], errorbar=('ci', 95), capsize=.05, alpha=0.8)
    axes[1].set_title('(b) Velocidade de Decodificação por Modalidade')
    axes[1].set_xlabel('Tempo (ms/MB)')
    axes[1].legend(loc='lower right', frameon=True, fontsize=10)
    axes[1].set_ylabel('Modalidade')
    add_labels(axes[1], fmt='%.0f', centered=True)
    
    plt.tight_layout()
    plt.savefig('fig_3_desempenho.png')
    print("Salvo: fig_3_desempenho.png")

def plot_quality_tradeoff(df):
    """Fig 3: Qualidade por Beta (Com Desvio -> Centro)"""
    df_plot = df[df['Beta'] == 0.2].copy()
    df_plot = df[df['Codec'] == 'JPEG XL'].groupby(['Modality', 'Beta'])[['MSE', 'PSNR_dB', 'SSIM']].mean().reset_index()
    df_plot['Beta'] = df_plot['Beta'].astype(str)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    order = ['CT', 'DX', 'MG']
    
    # MSE
    sns.barplot(data=df_plot, x='Modality', y='MSE', hue='Beta',
                order=order, ax=ax, palette='Reds')
    ax.set_title('Erro Quadrático (MSE)')
    ax.set_ylabel('MSE')
    ax.set_xlabel('Modalidade')
    add_labels(ax, fmt='%.6f', centered=False) # Com desvio -> Centro
    ax.legend(title='Beta', loc='upper right', frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig('fig_3_qualidade_beta.png')
    print("Salvo: fig_3_qualidade_beta.png")

def plot_ssim_combined(df):
    """Fig 4: SSIM (Painel A com Desvio -> Centro; Painel B Fixo -> Topo)"""
    df['Restored_SSIM'] = 1.0 
    
    df['Beta_Label'] = df['Beta'].apply(lambda x: f"Beta {x}")
    modality_order = ['CT', 'DX', 'MG']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # --- GRÁFICO 2: Reversibilidade (Sem Desvio - Valor Fixo 1.0) ---
    sns.barplot(
        data=df,
        x='Modality',
        y='Restored_SSIM',
        hue='Codec',
        order=modality_order,
        palette='Blues',
        ax=ax,
        errorbar=None
    )
    
    ax.set_title('Reversibilidade da Imagem Recuperada', fontsize=13)
    ax.set_xlabel('Modalidade')
    ax.set_ylabel('SSIM (Ideal = 1.0)')
    ax.set_ylim(0.0, 1.1) 
    ax.legend(title='Codec', loc='lower right', fontsize=10)
    
    # Sem desvio -> Topo (Edge)
    add_labels(ax, fmt='%.1f', centered=False)

    plt.tight_layout()
    plt.savefig('fig_ssim_recupered.png')
    print("Salvo: fig_ssim_recupered.png")

def plot_psnr_ssim_stego(df):
    """Gera gráfico combinado de PSNR e SSIM da Imagem Portadora"""
    
    # Agrupa por Modalidade e Beta (média dos codecs)
    # A qualidade esteganográfica (LGE) é independente do codec lossless usado depois
    df_plot = df.groupby(['Modality', 'Beta'])[['PSNR_dB', 'SSIM']].mean().reset_index()
    df_plot['Beta'] = df_plot['Beta'].astype(str)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    order = ['CT', 'DX', 'MG']
    
    # --- GRÁFICO 1 (ESQUERDA): PSNR ---
    sns.barplot(
        data=df_plot, 
        x='Modality', 
        y='PSNR_dB', 
        hue='Beta', 
        order=order, 
        ax=axes[0], 
        palette='Blues'
    )
    axes[0].set_title('(a) Pico de Relação Sinal-Ruído (PSNR)', fontsize=13)
    axes[0].set_xlabel('Modalidade', fontsize=12)
    axes[0].set_ylabel('PSNR (dB)', fontsize=12)
    axes[0].set_ylim(30, 100) # Ajuste de escala para visualização
    axes[0].legend(title='Beta', loc='lower right', fontsize=10)
    
    add_labels(axes[0], fmt='%.1f', centered=False) # Labels no centro (branco)
    
    # --- GRÁFICO 2 (DIREITA): SSIM ---
    sns.barplot(
        data=df_plot, 
        x='Modality', 
        y='SSIM', 
        hue='Beta', 
        order=order, 
        ax=axes[1], 
        palette='Greens'
    )
    axes[1].set_title('(b) Similaridade Estrutural (SSIM)', fontsize=13)
    axes[1].set_xlabel('Modalidade', fontsize=12)
    axes[1].set_ylabel('SSIM', fontsize=12)
    axes[1].set_ylim(0.95, 1.002) # Zoom para ver as diferenças sutis
    axes[1].legend(title='Beta', loc='lower right', fontsize=10)
    
    add_labels(axes[1], fmt='%.4f', centered=False) # Labels no centro com 4 casas decimais

    # Finalização
    plt.tight_layout()
    
    output_filename = 'fig_4_qualidade_psnr_ssim.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Salvo: {output_filename}")

if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_efficiency(df)
        plot_performance(df)
        plot_quality_tradeoff(df)
        plot_ssim_combined(df)
        plot_psnr_ssim_stego(df)

        print("\nTodos os gráficos foram atualizados!")