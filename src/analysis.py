import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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


# --- Gráfico e Tabela de Tempo de Compressão ---

encoding_data = df.groupby(['Parameter_Codec', 'Modality'])['Compression_Time_s'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df,
    x='Compression_Time_s',
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
plt.xlabel('Tempo de Compressão (s)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de Tempo Médio de Compressão (s) ---")
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

