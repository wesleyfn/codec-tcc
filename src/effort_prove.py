import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import imagecodecs
import pydicom
import numpy as np

# --- CONFIGURAÇÃO ---
# Coloque aqui o caminho de uma imagem DX (ruidosa) e uma MR (suave/controle)
IMG_CT = "images/CT/000.dcm" 
IMG_DX = "images/DX/000.dcm" 
IMG_MG = "images/MG/000.dcm" 
OUTPUT_FILE = "prova_effort_jxl_annotated.png"

def test_efforts_multi(img_path_list, label):
    
    all_results = []
    
    print(f"\n--- Testando {len(img_path_list)} imagens para {label} ---")
    
    for img_path in img_path_list:
        if not os.path.exists(img_path):
            print(f"Aviso: Arquivo não encontrado: {img_path}. Pulando.")
            continue

        # Ler imagem e preparar
        try:
            ds = pydicom.dcmread(img_path)
            arr = ds.pixel_array
        except Exception as e:
            print(f"Erro ao ler DICOM {os.path.basename(img_path)}: {e}. Pulando.")
            continue
            
        # Garantir formato uint16 para JXL
        if arr.dtype == np.int16:
            arr = arr.astype(np.uint16)
            
        if arr.ndim > 2:
            arr = np.squeeze(arr)

        original_size = arr.nbytes
        
        for eff in range(1, 10):
            t0 = time.time()
            
            # Encoding Lossless JXL (numthreads=1 para estabilidade)
            encoded = imagecodecs.jpegxl_encode(
                arr, 
                lossless=True, 
                effort=eff, 
                numthreads=0, 
                photometric='GRAY', 
                usecontainer=False
            )
            dt = time.time() - t0
            
            size = len(encoded)
            cr = original_size / size
            
            print(f"Effort {eff}: Size={size/1024:.1f} KB | CR={cr:.2f}x | Time={dt:.3f}s")
            
            # Adiciona o resultado
            all_results.append({
                'Effort': eff,
                'Size_KB': size / 1024,
                'CR': cr,
                'Time_s': dt,
                'Modality': label,
                'File': os.path.basename(img_path)
            })
            
    return pd.DataFrame(all_results)

# 1. Rodar Testes em Múltiplas Imagens e Consolidar
FILE_PATHS = {
    "CT": [IMG_CT, "images/CT/001.dcm", "images/CT/002.dcm", "images/CT/003.dcm", "images/CT/004.dcm", "images/CT/005.dcm", "images/CT/006.dcm", "images/CT/007.dcm", "images/CT/008.dcm", "images/CT/009.dcm"],
    "DX": [IMG_DX, "images/DX/001.dcm", "images/DX/002.dcm", "images/DX/003.dcm", "images/DX/004.dcm", "images/DX/005.dcm", "images/DX/006.dcm", "images/DX/007.dcm", "images/DX/008.dcm", "images/DX/009.dcm"],
    "MG": [IMG_MG, "images/MG/001.dcm", "images/MG/002.dcm", "images/MG/003.dcm", "images/MG/004.dcm", "images/MG/005.dcm", "images/MG/006.dcm", "images/MG/007.dcm", "images/MG/008.dcm", "images/MG/009.dcm"]
}

dataframes = []
for label, paths in FILE_PATHS.items():
    if paths:
        df_modality = test_efforts_multi(paths, label)
        dataframes.append(df_modality)

if not dataframes:
    print("Nenhum dado válido para plotar.")
    exit()

df_raw = pd.concat(dataframes, ignore_index=True)

# 2. AGREGAÇÃO: Calcular Média da Performance por Effort/Modality
df_mean = df_raw.groupby(['Modality', 'Effort'])[['Size_KB', 'Time_s', 'CR']].mean().reset_index()


# 3. ANÁLISE: Identificar Pontos Ótimos (CR Max Value)
analysis_results = []
for label in df_mean['Modality'].unique():
    subset = df_mean[df_mean['Modality'] == label]
    
    # Melhor CR (Min Size_KB é Max CR) -> Usamos idxmax() no CR
    best_cr_row = subset.loc[subset['CR'].idxmax()]
    
    # Melhor Tempo (Min Time_s)
    best_time_row = subset.loc[subset['Time_s'].idxmin()]

    analysis_results.append({
        'Modality': label,
        'Best_CR_Effort': best_cr_row['Effort'],
        'Best_CR_Size_KB': best_cr_row['Size_KB'],
        'Best_CR_Value': best_cr_row['CR'], # <--- CAMPO CR VALUE AQUI
        'Best_Time_Effort': best_time_row['Effort'],
        'Best_Time_Time_s': best_time_row['Time_s'],
    })
df_analysis = pd.DataFrame(analysis_results)


# 4. PLOTAGEM COM ANOTAÇÕES
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx() # Eixo secundário para o Tempo

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
lines_to_plot = []

# Loop para plotar as curvas MÉDIAS e anotar
for i, label in enumerate(df_mean['Modality'].unique()):
    subset = df_mean[df_mean['Modality'] == label]
    color = color_cycle[i]
    
    # Curva de Tamanho (Eixo Y1)
    line_size, = ax1.plot(subset['Effort'], subset['Size_KB'], marker='o', color=color, 
                          label=f"{label} (Size)")
    
    # Curva de Tempo (Eixo Y2)
    line_time, = ax2.plot(subset['Effort'], subset['Time_s'], marker='.', linestyle='--', alpha=0.6, color=color, 
                          label=f"{label} (Time)")
    
    # --- NOVO: Labels de Valor (KB) acima de cada ponto ---
    for x, y in zip(subset['Effort'], subset['Size_KB']):
        ax1.text(x, y + 15, f"{y:.0f}", ha='center', va='bottom', fontsize=7, color='black')

    
    # --- ANOTAÇÕES ÓTIMAS ---
    best_row = df_analysis[df_analysis['Modality'] == label].iloc[0]
    
    # CR Max Annotation (Eixo Primário - KB)
    x_cr = best_row['Best_CR_Effort']
    y_cr = best_row['Best_CR_Size_KB']
    best_cr_value = best_row['Best_CR_Value'] # Valor CR
    
    ax1.plot(x_cr, y_cr, 'X', color='black', markersize=8, zorder=5) # Marca CR Max (Black X)
    ax1.annotate(f"CR Max\n({best_cr_value:.2f}x)", # <--- MOSTRA O VALOR CR
                 (x_cr, y_cr), 
                 textcoords="offset points", 
                 xytext=(10, -20), ha='left', fontsize=8, 
                 bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8, edgecolor='none'))


# --- CONFIGURAÇÕES FINAIS DO GRÁFICO ---
ax1.set_xlabel('Parâmetro "Effort" do JPEG XL (1-9)', fontsize=12)
ax1.set_ylabel('Tamanho do Arquivo (KB)', color='tab:blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(range(1, 10))
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.set_ylabel('Tempo de Codificação (s)', color='tab:red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:red')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2, fontsize=8)


plt.title("Otimização de Esforço JXL: Média de Tamanho vs. Tempo por Modalidade", fontsize=14)
plt.tight_layout()
plt.savefig("prova_effort_jxl_mean_annotated.png")

print("\nGráfico de prova (curvas médias) salvo como 'prova_effort_jxl_mean_annotated.png'")