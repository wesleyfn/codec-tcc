import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_csv('../results_parallel.csv')

# Calcular BPP teórico original (para comparação)
# Assumindo que Bits_Stored é a profundidade de bits original
df['Bpp_Original'] = df['Bits_Stored']
df['Bpp_Comprimido'] = df['Bpp']

# Calcular eficiência de compressão
df['Eficiencia_Bpp'] = (df['Bpp_Original'] - df['Bpp_Comprimido']) / df['Bpp_Original'] * 100

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
sns.set_theme(style="whitegrid", palette="Set2")


figure = sns.catplot(
    data=df, 
    x='Parameter_Codec', y='Bpp_Comprimido',
    hue='Modality', 
    alpha=0.8, 
    kind='bar',
    legend=True, 
    aspect=1.5,
    legend_out=False
)
plt.ylabel('Bits por Pixel (bpp)')
plt.xlabel('Codec')

plt.show()


for figure in figure.axes.flat:
    for p in figure.patches:
        height = p.get_height()
        if height > 0:
            figure.text(p.get_x() + p.get_width() / 2., height / 2, f'{height:.2f}', ha="center", va="center", color="white", weight="bold")
