import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_csv('tcc_results/results_parallel.csv')

# Calcular BPP teórico original (para comparação)
# Assumindo que Bits_Stored é a profundidade de bits original
df['Bpp_Original'] = df['Bits_Stored']
df['Bpp_Comprimido'] = df['Bpp']

# Calcular eficiência de compressão
df['Eficiencia_Bpp'] = (df['Bpp_Original'] - df['Bpp_Comprimido']) / df['Bpp_Original'] * 100

# Calcular eficiência de compressão em bytes
df['Compression_Efficiency_Bytes'] = (df['Original_Size_Bytes'] - df['Compressed_Image_Size_Bytes']) / df['Original_Size_Bytes'] * 100

# Ordenar Codecs
df = df.sort_values(by='Bpp_Comprimido', ascending=False)

# Ordernar Modalidades
modality_order = ['dx_8b', 'dx_16b', 'mg_16b']
df['Modality'] = pd.Categorical(df['Modality'], categories=modality_order, ordered=True)

# Renomear codecs para melhor visualização
df['Parameter_Codec'] = df['Parameter_Codec'].replace({
    'png': 'PNG',
    'j2k': 'JPEG 2000', 
    'jls': 'JPEG-LS',
    'jxl': 'JPEG XL'
})

# Renomear modalidades para melhor visualização (CORREÇÃO DO WARNING)
df['Modality'] = df['Modality'].cat.rename_categories({
    'dx_8b': 'DX_8',
    'dx_16b': 'DX_16', 
    'mg_16b': 'MG_16',
})


###################################################################################################################################################################################################
sns.set_theme(style="whitegrid", palette="Set2", context="notebook")


# Calcular média e desvio padrão para anotação
stats = df.groupby(['Parameter_Codec', 'Modality'])['Bpp_Comprimido'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df, 
    x='Bpp_Comprimido', 
    y='Parameter_Codec',
    hue='Modality', 
    alpha=0.8, 
    kind='bar',
    aspect=1.8,
    legend=True, 
    legend_out=False,
    palette=['#f03c02', '#a30006', '#601848'],
    errorbar=('sd')
)
plt.xlabel('Bits por Pixel (bpp)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')


#####

figure = sns.catplot(
    data=df, 
    x='Eficiencia_Bpp', 
    y='Parameter_Codec',
    hue='Modality', 
    alpha=0.8, 
    kind='bar',
    aspect=1.8,
    palette=['#f03c02', '#a30006', '#601848'],
    errorbar=None,
    legend_out=False,
    legend=True
)
plt.xlabel('Eficiência de Compressão (%)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de BPP Comprimido ---")
bpp_table = df.groupby(['Parameter_Codec', 'Modality'])['Bpp_Comprimido'].agg(['mean', 'std']).unstack()
print(bpp_table)

plt.show()




# --- Gráfico e Tabela de Tempo de Codificação ---

encoding_data = df.groupby(['Parameter_Codec', 'Modality'])['Total_Encoding_Time_s'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df,
    x='Total_Encoding_Time_s',
    y='Parameter_Codec',
    hue='Modality',
    alpha=0.8,
    kind='bar',
    aspect=1.8,
    legend=True,
    legend_out=False,
    palette=['#f03c02', '#a30006', '#601848'],
    errorbar=('sd')
)
plt.xlabel('Tempo de Codificação (s)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de Tempo Médio de Codificação (s) ---")
encoding_table = encoding_data.pivot(index='Parameter_Codec', columns='Modality', values='mean')
print(encoding_table)

plt.show()

# --- Gráfico e Tabela de Tempo de Decodificação ---

decoding_data = df.groupby(['Parameter_Codec', 'Modality'])['Decoding_Time_s'].agg(['mean', 'std']).reset_index()

c = sns.catplot(
    data=df,
    x='Decoding_Time_s',
    y='Parameter_Codec',
    hue='Modality',
    alpha=0.8,
    kind='bar',
    aspect=1.8,
    legend=True,
    legend_out=False,
    palette=['#f03c02', '#a30006', '#601848'],
    errorbar=None
)
plt.xlabel('Tempo de Decodificação (s)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de Tempo Médio de Decodificação (s) ---")
decoding_table = decoding_data.pivot(index='Parameter_Codec', columns='Modality', values='mean')
print(decoding_table)

plt.show()


# --- Gráfico e Tabela de PSNR ---

psnr_data = df.groupby(['Parameter_Beta', 'Modality'])['Stego_Image_PSNR_dB'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df,
    x='Stego_Image_PSNR_dB',
    y='Modality',
    hue='Parameter_Beta',
    kind='bar',
    alpha=0.8,
    aspect=1.8,
    legend=True,
    legend_out=False,
    palette=['#f03c02', '#a30006', '#601848'],
    errorbar=('sd'),
)
plt.xlabel('PSNR da Imagem Esteganografada (dB)')
plt.ylabel('Modalidade')
plt.legend(title='Beta')
figure.set(xlim=(20, None)) # Ajusta o eixo X para melhor visualização

print("\n--- Tabela de PSNR Médio (dB) ---")
psnr_table = psnr_data.pivot(index='Parameter_Beta', columns='Modality', values='mean')
print(psnr_table)

plt.show()


# --- Gráfico de Dispersão Facetado: BPP vs. Tempo de Codificação ---

modalities = df['Modality'].cat.categories
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

# Plotar os dados para cada modalidade
for i, modality in enumerate(modalities):
    ax = axes_flat[i]
    data_modality = df[df['Modality'] == modality]
    sns.scatterplot(
        data=data_modality,
        x='Bpp_Comprimido',
        y='Total_Encoding_Time_s',
        hue='Parameter_Codec',
        s=120,
        alpha=0.8,
        ax=ax,
        legend=False  # Desativa legendas individuais
    )
    ax.set_title(f'Modalidade: {modality}')
    ax.set_xlabel('Bits por Pixel (bpp)')
    ax.set_ylabel('Tempo de Codificação (s)')

# Criar a legenda no quarto subplot
handles, labels = axes_flat[0].get_legend_handles_labels()
legend_ax = axes_flat[3]
legend_ax.legend(handles, labels, loc='center', title='Codec', fontsize='large', title_fontsize='x-large')
legend_ax.axis('off')  # Esconde os eixos do subplot da legenda

fig.suptitle('BPP vs. Tempo de Codificação por Modalidade', fontsize=16, y=1.02)
fig.tight_layout(pad=3.0)

plt.show()


# --- Gráfico de Dispersão Facetado: BPP vs. Tempo de Decodificação ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

# Plotar os dados para cada modalidade
for i, modality in enumerate(modalities):
    ax = axes_flat[i]
    data_modality = df[df['Modality'] == modality]
    sns.scatterplot(
        data=data_modality,
        x='Bpp_Comprimido',
        y='Decoding_Time_s',
        hue='Parameter_Codec',
        s=120,
        alpha=0.8,
        ax=ax,
        legend=False
    )
    ax.set_title(f'Modalidade: {modality}')
    ax.set_xlabel('Bits por Pixel (bpp)')
    ax.set_ylabel('Tempo de Decodificação (s)')

# Criar a legenda no quarto subplot
handles, labels = axes_flat[0].get_legend_handles_labels()
legend_ax = axes_flat[3]
legend_ax.legend(handles, labels, loc='center', title='Codec', fontsize='large', title_fontsize='x-large')
legend_ax.axis('off')

fig.suptitle('BPP vs. Tempo de Decodificação por Modalidade', fontsize=16, y=1.02)
fig.tight_layout(pad=3.0)

plt.show()
