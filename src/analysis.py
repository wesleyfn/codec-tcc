import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_bpp_cr_comparison():
    # 1. Configuração do Caminho do Arquivo
    # Tenta achar o CSV na pasta atual ou na subpasta tcc_results
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('tcc_results', 'results_sequential.csv')
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV não encontrado em '{csv_path}'")
        return

    # 2. Carregar e Filtrar Dados
    df = pd.read_csv(csv_path)
    
    # Filtramos por Beta = 0.2 para mostrar o cenário de alta qualidade (padrão)
    # Se quiser ver o Beta 0.8, mude aqui.
    df_plot = df[df['Beta'] == 0.2].copy()

    # 3. Configuração Estética
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Ordem fixa para garantir que as barras fiquem na mesma sequência sempre
    modality_order = ['CT', 'DX', 'MG']
    codec_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000'] 

    # --- GRÁFICO 1 (ESQUERDA): BITS POR PIXEL (Bpp) ---
    sns.barplot(
        data=df_plot,
        x='Modality',
        y='Bpp',
        hue='Codec',
        order=modality_order,
        hue_order=codec_order,
        ax=axes[0],
        errorbar=('ci', 95), # Intervalo de confiança
        capsize=.1,

    )
    
    axes[0].set_title('Bits por Pixel (Bpp) - Menor é Melhor', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modalidade', fontsize=12)
    axes[0].set_ylabel('Bits por Pixel', fontsize=12)
    axes[0].legend(title='Codec', loc='upper right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)


    # --- GRÁFICO 2 (DIREITA): TAXA DE COMPRESSÃO (CR) ---
    sns.barplot(
        data=df_plot,
        x='Modality',
        y='CR',
        hue='Codec',
        order=modality_order,
        hue_order=codec_order,
        ax=axes[1],
        errorbar=('ci', 95),
        capsize=.1,
    )
    
    axes[1].set_title('Taxa de Compressão (CR) - Maior é Melhor', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Modalidade', fontsize=12)
    axes[1].set_ylabel('Taxa de Compressão (x:1)', fontsize=12)
    axes[1].legend(title='Codec', loc='upper left')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Finalização
    plt.tight_layout()

    # 5. Salvar
    output_filename = 'grafico_bpp_cr_lado_a_lado.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Sucesso! Gráfico salvo como: {output_filename}")
    plt.show()


    


def plot_encoding_speed():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        # Tenta achar em subpastas se necessário
        for root, dirs, files in os.walk('.'):
            if 'results_sequential.csv' in files:
                csv_path = os.path.join(root, 'results_sequential.csv')
                break
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV não encontrado.")
        return

    # 2. Carregar e Filtrar Dados
    df = pd.read_csv(csv_path)
    
    # Filtramos por Beta = 0.2 para consistência (menor distorção)
    df_plot = df[df['Beta'] == 0.2].copy()

    # 3. Configuração Estética
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Ordem fixa para consistência
    modality_order = ['CT', 'DX', 'MG']
    codec_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000'] 

    # --- GRÁFICO: VELOCIDADE DE CODIFICAÇÃO (ms/MB) ---
    ax = sns.barplot(
        data=df_plot,
        y='Modality',
        x='Encoding_Speed_ms_MB',
        hue='Codec',
        order=modality_order,
        hue_order=codec_order,
        errorbar=('ci', 95), # Intervalo de confiança
        capsize=.1,
    )
    
    # Títulos e Rótulos
    plt.ylabel('Modalidade', fontsize=12)
    plt.xlabel('Tempo de Processamento (ms/MB)', fontsize=12)
    plt.legend(title='Codec', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)


    # 4. Salvar e Mostrar
    plt.tight_layout()
    output_filename = 'grafico_velocidade_codificacao.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Gráfico gerado: {output_filename}")
    plt.show()

def plot_decoding_speed():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        for root, dirs, files in os.walk('.'):
            if 'results_sequential.csv' in files:
                csv_path = os.path.join(root, 'results_sequential.csv')
                break
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV não encontrado.")
        return

    # 2. Carregar e Filtrar Dados
    df = pd.read_csv(csv_path)
    df_plot = df[df['Beta'] == 0.2].copy()

    # 3. Configuração Estética
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))

    modality_order = ['CT', 'DX', 'MG']
    codec_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000'] 

    # --- GRÁFICO: VELOCIDADE DE DECODIFICAÇÃO (ms/MB) ---
    ax = sns.barplot(
        data=df_plot,
        y='Modality',
        x='Decoding_Speed_ms_MB', # <--- MUDANÇA AQUI
        hue='Codec',
        order=modality_order,
        hue_order=codec_order,
        errorbar=('ci', 95),
        capsize=.1,
        palette='pastel' # Cor diferente para distinguir do gráfico de codificação
    )
    
    plt.ylabel('Modalidade', fontsize=12)
    plt.xlabel('Tempo de Decodificação (ms/MB)', fontsize=12)
    plt.legend(title='Codec', loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 4. Salvar
    plt.tight_layout()
    output_filename = 'grafico_velocidade_decodificacao.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Sucesso! Gráfico salvo como: {output_filename}")
    plt.show()

def plot_quality_by_beta():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        for root, dirs, files in os.walk('.'):
            if 'results_sequential.csv' in files:
                csv_path = os.path.join(root, 'results_sequential.csv')
                break
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV não encontrado.")
        return

    # 2. Carregar Dados
    df = pd.read_csv(csv_path)
    
    # 3. Preparação dos Dados
    # Como a qualidade depende do Beta e não do Codec (todos são lossless),
    # podemos simplificar removendo a redundância dos codecs para o gráfico não ficar poluído.
    # Vamos pegar a média dos codecs para cada combinação de (Imagem, Beta)
    df_plot = df.groupby(['Image_File', 'Modality', 'Beta'])[['MSE', 'PSNR_dB', 'SSIM']].mean().reset_index()
    
    # Converter Beta para texto para ficar bonito na legenda
    df_plot['Beta_Label'] = df_plot['Beta'].apply(lambda x: f"Beta = {x}")

    # 4. Configuração Estética
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    
    modality_order = ['CT', 'DX', 'MG']

    # --- GRÁFICO 1: MSE (Menor é Melhor) ---
    sns.barplot(
        data=df_plot,
        x='Modality',
        y='MSE',
        hue='Beta_Label',
        order=modality_order,
        ax=axes[0],
        palette='Reds',
        errorbar=('ci', 95),
        capsize=.1
    )
    axes[0].set_title('Erro Quadrático Médio (MSE)\nQuanto menor, melhor', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('MSE')
    
    # --- GRÁFICO 2: PSNR (Maior é Melhor) ---
    sns.barplot(
        data=df_plot,
        x='Modality',
        y='PSNR_dB',
        hue='Beta_Label',
        order=modality_order,
        ax=axes[1],
        palette='Blues',
        errorbar=('ci', 95),
        capsize=.1
    )
    axes[1].set_title('Pico de Relação Sinal-Ruído (PSNR)\nQuanto maior, melhor', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('PSNR (dB)')

    # --- GRÁFICO 3: SSIM (Maior é Melhor) ---
    sns.barplot(
        data=df_plot,
        x='Modality',
        y='SSIM',
        hue='Beta_Label',
        order=modality_order,
        ax=axes[2],
        palette='Greens',
        errorbar=('ci', 95),
        capsize=.1
    )
    axes[2].set_title('Similaridade Estrutural (SSIM)\nQuanto mais próximo de 1.0, melhor', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('SSIM')
    axes[2].set_ylim(0.9, 1.005) # Zoom para ver a diferença, já que é tudo muito alto

    # Ajustes Finais
    plt.suptitle('Impacto do Parâmetro Beta na Qualidade da Imagem Portadora', fontsize=16, y=1.05)
    plt.tight_layout()
    
    # Salvar
    output_filename = 'comparativo_beta_qualidade.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"Sucesso! Gráfico salvo como: {output_filename}")
    plt.show()

if __name__ == "__main__":
    plot_bpp_cr_comparison()
    plot_encoding_speed()
    plot_decoding_speed()
    plot_quality_by_beta()