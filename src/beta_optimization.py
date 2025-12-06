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
    
    # Exibe a tabela de resultados no console
    print("\n--- Tabela de Resultados da Análise de Beta ---")
    with pd.option_context('display.max_rows', None, 'display.width', 1000):
        print(df[['Modality_Label', 'Beta', 'PSNR_dB', 'SSIM', 'MSE', 'Bpp', 'CR']].round(4))
    print("------------------------------------------------\n")
    
    plot_results(df)

def plot_results(df):
    """Plota os resultados de PSNR vs. Beta para todas as modalidades em um único gráfico."""
    fig, ax = plt.subplots(figsize=(12, 7))

    print("\n> Gerando gráfico...")

    # Define um ciclo de cores para as modalidades
    modality_colors = {
        'CT': '#1f77b4',  # Azul
        'DX': '#ff7f0e',  # Laranja
        'MG': '#2ca02c'   # Verde
    }

    for mod in df['Modality_Label'].unique():
        subset = df[df['Modality_Label'] == mod].sort_values('Beta')
        color = modality_colors[mod]
        
        # Plota a linha para a modalidade atual
        ax.plot(subset['Beta'], subset['PSNR_dB'], 
                marker='o', color=color, label=mod, linewidth=2)

        # Anotação de valores em pontos chave
        for x, y_psnr in zip(subset['Beta'], subset['PSNR_dB']):
            if x in [0.2, 0.5, 0.8]:
                ax.annotate(f"{y_psnr:.1f}", (x, y_psnr), textcoords="offset points", xytext=(0,10), ha='center', color=color, fontsize=8, weight='bold')

    ax.set_xlabel('Parâmetro Beta (β)', fontsize=12)
    ax.set_ylabel('PSNR (dB)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title='Modalidade', loc='lower right')
    plt.suptitle(f"Análise de Sensibilidade do Beta (Codec: {CODEC_FIXED.upper()})", fontsize=16, y=0.96)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Gráfico salvo como: {OUTPUT_PLOT}")

    

if __name__ == "__main__":
    # Suprime avisos de bibliotecas de imagem
    warnings.filterwarnings("ignore")
    run_beta_analysis()