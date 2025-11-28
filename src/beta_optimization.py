import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Importa as funções do seu codec original
# Certifique-se de que o arquivo 'codec.py' está na mesma pasta
try:
    from codec import process_single_image
except ImportError:
    print("ERRO: O arquivo 'codec.py' não foi encontrado no diretório atual.")
    exit()

# --- CONFIGURAÇÃO ---
# Caminhos para uma imagem de cada modalidade
# Ajuste conforme seus arquivos reais
IMG_PATHS = {
    "CT": "images/CT/000.dcm",
    "DX": "images/DX/000.dcm",
    "MG": "images/MG/000.dcm"
}

OUTPUT_DIR = "beta_test_results/"
OUTPUT_PLOT = "analise_beta_otimo.png"
CODEC_FIXED = 'jxl'       # Codec fixo para isolar a variável Beta
BLOCK_SIZE = 4
PERCENTILE = 90
TARGET_BIT_DEPTH = 16

# Intervalo de Beta para testar (0.1 a 0.9)
BETAS_TO_TEST = [round(x * 0.1, 1) for x in range(1, 10)]

def run_beta_analysis():
    print(f"--- Iniciando Análise de Beta (0.1 - 0.9) ---")
    
    all_results = []
    
    # Cria diretório de saída se não existir
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for modality, img_path in IMG_PATHS.items():
        if not os.path.exists(img_path):
            print(f"AVISO: Imagem {modality} não encontrada em {img_path}. Pulando.")
            continue
            
        print(f"\n> Processando {modality} ({os.path.basename(img_path)})...")
        
        # Chama o processamento do codec.py iterando os Betas
        # Nota: process_single_image retorna uma lista de dicionários
        try:
            # Passamos a lista completa de Betas para a função
            results = process_single_image(
                img_path, 
                OUTPUT_DIR, 
                [CODEC_FIXED], 
                BETAS_TO_TEST, 
                BLOCK_SIZE, 
                PERCENTILE, 
                TARGET_BIT_DEPTH, 
                debug_mode=False
            )
            
            # Adiciona a tag da modalidade para o gráfico
            for res in results:
                res['Modality_Label'] = modality
                all_results.append(res)
                
        except Exception as e:
            print(f"Erro ao processar {modality}: {e}")

    if not all_results:
        print("Nenhum resultado gerado.")
        return

    df = pd.DataFrame(all_results)
    
    # Salva CSV bruto para referência
    df.to_csv("beta_analysis_raw.csv", index=False)
    
    plot_results(df)

def plot_results(df):
    # Configuração do Plot (3 subplots, um para cada modalidade)
    modalities = df['Modality_Label'].unique()
    fig, axes = plt.subplots(1, len(modalities), figsize=(15, 6), sharey='row')
    
    if len(modalities) == 1: axes = [axes] # Garante que seja lista se for só 1

    print("\n> Gerando gráfico...")

    for i, mod in enumerate(modalities):
        ax1 = axes[i]
        subset = df[df['Modality_Label'] == mod].sort_values('Beta')
        
        # Eixo Y1: PSNR (Qualidade)
        color_psnr = 'tab:blue'
        l1 = ax1.plot(subset['Beta'], subset['PSNR_dB'], 
                 marker='o', color=color_psnr, label='PSNR (dB)', linewidth=2)
        ax1.set_xlabel('Parâmetro Beta (β)', fontsize=12)
        ax1.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color_psnr)
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.set_title(f"Modalidade: {mod}", fontweight='bold')
        
        # Eixo Y2: Taxa de Compressão ou BPP (Eficiência/Capacidade)
        # Como o BPP varia com o Beta (mais dados escondidos = maior entropia no arquivo final),
        # usamos a Taxa de Compressão (CR) inversa ou o tamanho do arquivo.
        # Aqui usaremos Bpp (Bits per Pixel) do arquivo final.
        # Quanto MAIOR o Bpp, mais informação (ruído/mensagem) foi preservada.
        
        ax2 = ax1.twinx()
        color_bpp = 'tab:red'
        l2 = ax2.plot(subset['Beta'], subset['Bpp'], 
                 marker='s', linestyle='--', color=color_bpp, label='Bpp (Final)', alpha=0.7)
        
        if i == len(modalities) - 1: # Rotulo apenas no último para não poluir
            ax2.set_ylabel('Bits por Pixel (Bpp)', color=color_bpp, fontsize=12)
        ax2.tick_params(axis='y', labelcolor=color_bpp)

        # --- ANOTAÇÕES E DESTAQUES ---
        # Destacar o Beta que dá o melhor equilíbrio (Ex: PSNR > 50 e Max Bpp)
        # Lógica simples: Beta onde o PSNR cai abaixo de um limiar crítico (ex: 60dB)
        # ou onde o Bpp estabiliza.
        
        # Anotação de valores no gráfico
        for x, y_psnr, y_bpp in zip(subset['Beta'], subset['PSNR_dB'], subset['Bpp']):
            if x in [0.2, 0.5, 0.8]: # Anota apenas alguns pontos chave para não poluir
                ax1.annotate(f"{y_psnr:.1f}dB", (x, y_psnr), textcoords="offset points", xytext=(0,10), ha='center', color=color_psnr, fontsize=8)
                ax2.annotate(f"{y_bpp:.1f}", (x, y_bpp), textcoords="offset points", xytext=(0,-15), ha='center', color=color_bpp, fontsize=8)

    plt.suptitle(f"Análise de Sensibilidade do Beta: PSNR vs. Bpp (Codec: {CODEC_FIXED.upper()})", fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Gráfico salvo como: {OUTPUT_PLOT}")

if __name__ == "__main__":
    # Suprime avisos de bibliotecas de imagem
    warnings.filterwarnings("ignore")
    run_beta_analysis()