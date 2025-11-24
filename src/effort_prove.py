import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import imagecodecs
import pydicom
import numpy as np

# --- CONFIGURAÇÃO ---
# Coloque aqui o caminho de uma imagem DX (ruidosa) e uma MR (suave/controle)
IMG_CT = "images/CT_16b/000.dcm"  # Ajuste para o seu arquivo
IMG_DX = "images/DX_16b/000.dcm"  # Ajuste para o seu arquivo
IMG_MG = "images/MG_16b/000.dcm"  # Ajuste para o seu arquivo
OUTPUT_FILE = "prova_effort_jxl_annotated.png"

def test_efforts(img_path, label):
    if not os.path.exists(img_path):
        print(f"Arquivo não encontrado: {img_path}")
        return []

    # Ler imagem
    ds = pydicom.dcmread(img_path)
    arr = ds.pixel_array
    
    # Garantir formato correto para JXL
    if arr.dtype == np.int16:
        arr = arr.astype(np.uint16) # JXL prefere uint
    
    # Correção de estrutura (remover dimensão singleton, se houver)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        
    results = []
    original_size = arr.nbytes 
    
    print(f"\n--- Testando {label} ({os.path.basename(img_path)}) ---")
    
    # Testar efforts de 1 (Lightning) a 9 (Tortoise)
    efforts = range(1, 10)
    
    for eff in efforts:
        t0 = time.time()
        # Encoding Lossless JXL
        encoded = imagecodecs.jpegxl_encode(
            arr, 
            lossless=True, 
            effort=eff, 
            numthreads=0, # Usar 1 thread para maior estabilidade
            photometric='GRAY', 
            usecontainer=False
        )
        dt = time.time() - t0
        
        size = len(encoded)
        cr = original_size / size
        
        print(f"Effort {eff}: Size={size/1024:.1f} KB | CR={cr:.2f}x | Time={dt:.3f}s")
        
        results.append({
            'Effort': eff,
            'Size_KB': size / 1024,
            'CR': cr,
            'Time_s': dt,
            'Modality': label
        })
        
    return results

# 1. Rodar Testes e Consolidar Dados
data_ct = test_efforts(IMG_CT, "CT")
data_dx = test_efforts(IMG_DX, "DX")
data_mg = test_efforts(IMG_MG, "MG")

if not data_dx and not data_mg and not data_ct:
    exit()

df = pd.DataFrame(data_dx + data_mg + data_ct)


# 2. ANÁLISE: Identificar Pontos Ótimos
analysis_results = []
for label in df['Modality'].unique():
    subset = df[df['Modality'] == label]
    
    # Melhor CR (Min Size_KB)
    best_size_row = subset.loc[subset['Size_KB'].idxmin()]
    
    # Melhor Tempo (Min Time_s)
    best_time_row = subset.loc[subset['Time_s'].idxmin()]

    analysis_results.append({
        'Modality': label,
        'Best_CR_Effort': best_size_row['Effort'],
        'Best_CR_Size_KB': best_size_row['Size_KB'],
        'Best_Time_Effort': best_time_row['Effort'],
        'Best_Time_Time_s': best_time_row['Time_s'],
    })
df_analysis = pd.DataFrame(analysis_results)

# 3. SAÍDA NO CONSOLE
print("\n--- RESUMO DE OTIMIZAÇÃO (BEST EFFORT) ---")
print(df_analysis.to_markdown(index=False, floatfmt=".3f"))


# 4. PLOTAGEM COM ANOTAÇÕES
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx() # Eixo secundário para o Tempo

# Define o ciclo de cores para usar na plotagem
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = {mod: color_cycle[i % len(color_cycle)] for i, mod in enumerate(df['Modality'].unique())}


# --- PLOTAGEM DAS CURVAS (Tamanho e Tempo) ---
for label in df['Modality'].unique():
    subset = df[df['Modality'] == label]
    color = color_map[label]
    
    # Curva de Tamanho (Eixo Y1)
    ax1.plot(subset['Effort'], subset['Size_KB'], marker='o', color=color, label=f"{label} (Size)")
    
    # Curva de Tempo (Eixo Y2)
    ax2.plot(subset['Effort'], subset['Time_s'], marker='.', linestyle='--', alpha=0.6, color=color, label=f"{label} (Time)")


    # --- ANOTAÇÃO DOS VALORES (Labels) ---
    for x, y in zip(subset['Effort'], subset['Size_KB']):
        # Label de valor: Tamanho arredondado (para evitar sobrecarga)
        ax1.text(x, y + 15, f"{y:.0f}", ha='center', va='bottom', fontsize=7, color='black')


# --- ANOTAÇÃO DOS PONTOS ÓTIMOS (Best CR & Best Time) ---
for i, row in df_analysis.iterrows():
    modality_label = row['Modality']
    
    # --- Ponto Ótimo de Tamanho (CR Max) ---
    x_cr = row['Best_CR_Effort']
    y_cr = row['Best_CR_Size_KB']
    
    # --- Ponto Ótimo de Tempo (Time Min) ---
    x_time = row['Best_Time_Effort']
    y_time_s = row['Best_Time_Time_s']

    # CR Max Annotation (Eixo Primário - KB)
    ax1.plot(x_cr, y_cr, 'X', color='black', markersize=8, zorder=5) # Marca CR Max
    ax1.annotate(f"CR Max\n({y_cr:.0f} KB)", 
                 (x_cr, y_cr), 
                 textcoords="offset points", 
                 xytext=(10, -20), 
                 ha='left', 
                 fontsize=8, 
                 bbox=dict(boxstyle="round,pad=0.2", fc="yellow", alpha=0.8, edgecolor='none'))

    # Time Min Annotation (Eixo Secundário - s)
    ax2.plot(x_time, y_time_s, 'v', color='darkred', markersize=8, zorder=5) # Marca Time Min
    ax2.annotate(f"Time Min", 
                 (x_time, y_time_s), 
                 textcoords="offset points", 
                 xytext=(-20, 10), 
                 ha='right', 
                 fontsize=8, 
                 color='darkred',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, edgecolor='none'))


# --- CONFIGURAÇÕES FINAIS DO GRÁFICO ---
ax1.set_xlabel('Parâmetro "Effort" do JPEG XL (1-9)', fontsize=12)
ax1.set_ylabel('Tamanho do Arquivo (KB)', color='tab:blue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_xticks(range(1, 10))
ax1.grid(True, linestyle='--', alpha=0.5)

ax2.set_ylabel('Tempo de Codificação (s)', color='tab:red', fontsize=12)
ax2.tick_params(axis='y', labelcolor='tab:red')

# Unificar a legenda (Manual)
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', ncol=2, fontsize=8)


plt.title("Otimização de Esforço JXL: Tamanho vs. Tempo por Modalidade", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_FILE)
plt.show()
print(f"\nGráfico de prova salvo como '{OUTPUT_FILE}'")