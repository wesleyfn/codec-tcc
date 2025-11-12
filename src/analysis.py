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
modality_order = ['dx_8b', 'dx_16b', 'mr_16b']
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
    'mr_16b': 'MG_16',
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
    legend=True, 
    aspect=1.8,
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

#####

# Preparar dados para o gráfico de desempenho computacional
df_time = df.melt(
    id_vars=['Parameter_Codec', 'Modality'], 
    value_vars=['Total_Encoding_Time_s', 'Decoding_Time_s'],
    var_name='Tipo de Tempo',
    value_name='Tempo (s)'
)

df_time['Tipo de Tempo'] = df_time['Tipo de Tempo'].replace({
    'Total_Encoding_Time_s': 'Codificação',
    'Decoding_Time_s': 'Decodificação'
})

# Criar uma nova coluna para o hue, combinando Modalidade e Tipo de Tempo
df_time['Group'] = df_time['Modality'].astype(str) + ' - ' + df_time['Tipo de Tempo']

# Ordenar os grupos para uma melhor visualização na legenda
modalities = df['Modality'].cat.categories
time_types = ['Codificação', 'Decodificação']
hue_order = [f'{mod} - {time}' for mod in modalities for time in time_types]

# Gráfico de Desempenho Computacional Unificado
time_figure = sns.catplot(
    data=df_time,
    x='Tempo (s)',
    y='Parameter_Codec',
    hue='Group',
    hue_order=hue_order,
    kind='bar',
    aspect=1.8,
    height=6,
    palette='Paired',
    errorbar=None,
    legend_out=False
)

time_figure.set_axis_labels('Tempo (s)', 'Codec')
time_figure.legend.set_title('Modalidade e Operação')
time_figure.fig.suptitle('Desempenho Computacional por Codec, Modalidade e Operação', y=1.03, size=14)
time_figure.tight_layout()

plt.show()

#####



