import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_bpp_cr_comparison():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('tcc_results', 'results_sequential.csv')
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV 'results_sequential.csv' não encontrado.")
        return

    # 2. Carregar e Filtrar Dados
    df = pd.read_csv(csv_path)
    df_plot = df[df['Beta'] == 0.2].copy()

    # 3. Gerar e Imprimir Tabela de Resumo
    summary_table = df_plot.groupby(['Modality', 'Codec']).agg(
        Mean_Bpp=('Bpp', 'mean'),
        Mean_CR=('CR', 'mean'),
        Mean_Metadata_Size_Bytes=('Metadata_Size_Bytes', 'mean')
    ).reset_index()
    
    summary_table['Mean_Bpp'] = summary_table['Mean_Bpp'].map('{:.3f}'.format)
    summary_table['Mean_CR'] = summary_table['Mean_CR'].map('{:.2f}'.format)
    summary_table['Mean_Metadata_Size_Bytes'] = summary_table['Mean_Metadata_Size_Bytes'].map('{:,.0f}'.format).str.replace(',', '.')

    modality_order = ['CT', 'DX', 'MG']
    summary_table['Modality'] = pd.Categorical(summary_table['Modality'], categories=modality_order, ordered=True)
    summary_table = summary_table.sort_values(['Modality', 'Codec'])

    print("\n" + "="*80)
    print("Tabela 1: Comparativo de Bpp, Taxa de Compressão (CR) e Tamanho Médio dos Metadados")
    print("Agrupado por Modalidade e Codec (Beta = 0.2)")
    print("-"*80)
    print(summary_table.to_string(index=False))
    print("="*80)

    # 4. Configuração Estética do Gráfico
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    codec_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000'] 

    sns.barplot(
        data=df_plot, x='Modality', y='Bpp', hue='Codec',
        order=modality_order, hue_order=codec_order, ax=axes[0],
        errorbar=('ci', 95), capsize=.1,
    )
    axes[0].set_title('Bits por Pixel (Bpp) - Menor é Melhor', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Modalidade', fontsize=12)
    axes[0].set_ylabel('Bits por Pixel', fontsize=12)
    axes[0].legend(title='Codec', loc='upper right')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    sns.barplot(
        data=df_plot, x='Modality', y='CR', hue='Codec',
        order=modality_order, hue_order=codec_order, ax=axes[1],
        errorbar=('ci', 95), capsize=.1,
    )
    axes[1].set_title('Taxa de Compressão (CR) - Maior é Melhor', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Modalidade', fontsize=12)
    axes[1].set_ylabel('Taxa de Compressão (x:1)', fontsize=12)
    axes[1].legend(title='Codec', loc='upper left')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # 5. Salvar
    plt.tight_layout()
    output_filename = 'grafico_bpp_cr_lado_a_lado.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nSucesso! Gráfico salvo como: {output_filename}")
    plt.show()

def plot_encoding_speed():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('tcc_results', 'results_sequential.csv')
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV 'results_sequential.csv' não encontrado.")
        return

    # 2. Carregar e Filtrar Dados
    df = pd.read_csv(csv_path)
    df_plot = df[df['Beta'] == 0.2].copy()

    # 3. Gerar e Imprimir Tabela de Resumo
    summary_table = df_plot.groupby(['Modality', 'Codec']).agg(
        Mean_Encoding_Speed_ms_MB=('Encoding_Speed_ms_MB', 'mean')
    ).reset_index()

    summary_table['Mean_Encoding_Speed_ms_MB'] = summary_table['Mean_Encoding_Speed_ms_MB'].map('{:.2f}'.format)
    
    modality_order = ['CT', 'DX', 'MG']
    summary_table['Modality'] = pd.Categorical(summary_table['Modality'], categories=modality_order, ordered=True)
    summary_table = summary_table.sort_values(['Modality', 'Codec'])

    print("\n" + "="*80)
    print("Tabela 2: Velocidade de Codificação (ms/MB)")
    print("Agrupado por Modalidade e Codec (Beta = 0.2)")
    print("-"*80)
    print(summary_table.to_string(index=False))
    print("="*80)

    # 4. Configuração Estética
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    codec_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000'] 

    ax = sns.barplot(
        data=df_plot, y='Modality', x='Encoding_Speed_ms_MB', hue='Codec',
        order=modality_order, hue_order=codec_order,
        errorbar=('ci', 95), capsize=.1,
    )
    
    ax.set_title('Velocidade de Codificação - Menor é Melhor', fontsize=14, fontweight='bold')
    plt.ylabel('Modalidade', fontsize=12)
    plt.xlabel('Tempo de Processamento (ms/MB)', fontsize=12)
    plt.legend(title='Codec', loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 5. Salvar e Mostrar
    plt.tight_layout()
    output_filename = 'grafico_velocidade_codificacao.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nGráfico gerado: {output_filename}")
    plt.show()

def plot_decoding_speed():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('tcc_results', 'results_sequential.csv')
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV 'results_sequential.csv' não encontrado.")
        return

    # 2. Carregar e Filtrar Dados
    df = pd.read_csv(csv_path)
    df_plot = df[df['Beta'] == 0.2].copy()

    # 3. Gerar e Imprimir Tabela de Resumo
    summary_table = df_plot.groupby(['Modality', 'Codec']).agg(
        Mean_Decoding_Speed_ms_MB=('Decoding_Speed_ms_MB', 'mean')
    ).reset_index()

    summary_table['Mean_Decoding_Speed_ms_MB'] = summary_table['Mean_Decoding_Speed_ms_MB'].map('{:.2f}'.format)
    
    modality_order = ['CT', 'DX', 'MG']
    summary_table['Modality'] = pd.Categorical(summary_table['Modality'], categories=modality_order, ordered=True)
    summary_table = summary_table.sort_values(['Modality', 'Codec'])
    
    print("\n" + "="*80)
    print("Tabela 3: Velocidade de Decodificação (ms/MB)")
    print("Agrupado por Modalidade e Codec (Beta = 0.2)")
    print("-"*80)
    print(summary_table.to_string(index=False))
    print("="*80)

    # 4. Configuração Estética
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    codec_order = ['JPEG XL', 'JPEG-LS', 'JPEG 2000'] 

    ax = sns.barplot(
        data=df_plot, y='Modality', x='Decoding_Speed_ms_MB', hue='Codec',
        order=modality_order, hue_order=codec_order,
        errorbar=('ci', 95), capsize=.1, palette='pastel'
    )
    
    ax.set_title('Velocidade de Decodificação - Menor é Melhor', fontsize=14, fontweight='bold')
    plt.ylabel('Modalidade', fontsize=12)
    plt.xlabel('Tempo de Decodificação (ms/MB)', fontsize=12)
    plt.legend(title='Codec', loc='lower right')
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # 5. Salvar
    plt.tight_layout()
    output_filename = 'grafico_velocidade_decodificacao.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nSucesso! Gráfico salvo como: {output_filename}")
    plt.show()

def plot_quality_by_beta():
    # 1. Configuração do Caminho do Arquivo
    csv_path = 'results_sequential.csv'
    if not os.path.exists(csv_path):
        csv_path = os.path.join('tcc_results', 'results_sequential.csv')
    
    if not os.path.exists(csv_path):
        print(f"Erro: Arquivo CSV 'results_sequential.csv' não encontrado.")
        return

    # 2. Carregar Dados
    df = pd.read_csv(csv_path)

    # 3. Gerar e Imprimir Tabela de Resumo
    summary_table = df.groupby(['Modality', 'Beta']).agg(
        Mean_MSE=('MSE', 'mean'),
        Mean_PSNR_dB=('PSNR_dB', 'mean'),
        Mean_SSIM=('SSIM', 'mean'),
        Mean_Metadata_Size_Bytes=('Metadata_Size_Bytes', 'mean')
    ).reset_index()

    summary_table['Mean_MSE'] = summary_table['Mean_MSE'].map('{:.6f}'.format)
    summary_table['Mean_PSNR_dB'] = summary_table['Mean_PSNR_dB'].map('{:.2f}'.format)
    summary_table['Mean_SSIM'] = summary_table['Mean_SSIM'].map('{:.4f}'.format)
    summary_table['Mean_Metadata_Size_Bytes'] = summary_table['Mean_Metadata_Size_Bytes'].map('{:,.0f}'.format).str.replace(',', '.')

    modality_order = ['CT', 'DX', 'MG']
    summary_table['Modality'] = pd.Categorical(summary_table['Modality'], categories=modality_order, ordered=True)
    summary_table = summary_table.sort_values(['Modality', 'Beta'])
    
    print("\n" + "="*80)
    print("Tabela 4: Impacto do Beta na Qualidade da Imagem e Tamanho dos Metadados")
    print("Agrupado por Modalidade e Beta")
    print("-"*80)
    print(summary_table.to_string(index=False))
    print("="*80)
    
    # 4. Preparação dos Dados para o Gráfico
    df_plot = df.groupby(['Image_File', 'Modality', 'Beta'])[['MSE', 'PSNR_dB', 'SSIM']].mean().reset_index()
    df_plot['Beta_Label'] = df_plot['Beta'].apply(lambda x: f"Beta = {x}")

    # 5. Configuração Estética
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.barplot(
        data=df_plot, x='Modality', y='MSE', hue='Beta_Label',
        order=modality_order, ax=axes[0], palette='Reds',
        errorbar=('ci', 95), capsize=.1
    )
    axes[0].set_title('Erro Quadrático Médio (MSE)\nMenor é Melhor', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Modalidade', fontsize=10)
    axes[0].set_ylabel('MSE', fontsize=10)
    axes[0].legend(title='Beta')
    
    sns.barplot(
        data=df_plot, x='Modality', y='PSNR_dB', hue='Beta_Label',
        order=modality_order, ax=axes[1], palette='Blues',
        errorbar=('ci', 95), capsize=.1
    )
    axes[1].set_title('Pico de Relação Sinal-Ruído (PSNR)\nMaior é Melhor', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Modalidade', fontsize=10)
    axes[1].set_ylabel('PSNR (dB)', fontsize=10)
    axes[1].legend(title='Beta')

    sns.barplot(
        data=df_plot, x='Modality', y='SSIM', hue='Beta_Label',
        order=modality_order, ax=axes[2], palette='Greens',
        errorbar=('ci', 95), capsize=.1
    )
    axes[2].set_title('Similaridade Estrutural (SSIM)\nMais próximo de 1.0 é melhor', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Modalidade', fontsize=10)
    axes[2].set_ylabel('SSIM', fontsize=10)
    axes[2].set_ylim(0.95, 1.001)
    axes[2].legend(title='Beta')

    # 6. Ajustes Finais
    plt.suptitle('Impacto do Parâmetro Beta na Qualidade da Imagem Portadora', fontsize=16, y=1.03)
    plt.tight_layout(pad=2.0)
    
    # 7. Salvar
    output_filename = 'graficos_qualidade_stego.png'
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"\nSucesso! Gráfico salvo como: {output_filename}")
    plt.show()


if __name__ == "__main__":
    plot_bpp_cr_comparison()
    plot_encoding_speed()
    plot_decoding_speed()
    plot_quality_by_beta()