import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Carregar os dados
df = pd.read_csv('tcc_results/results_sequential.csv')

# Calcular Bpp teórico original (para comparação)
# Assumindo que Bits_Stored é a profundidade de bits original
df['Bpp_Original'] = df['Bits_Stored']

# Calcular eficiência de compressão
df['Eficiencia_Bpp'] = (df['Bpp_Original'] - df['Bpp']) / df['Bpp_Original'] * 100

# Ordenar Codecs
df = df.sort_values(by='Bpp', ascending=False)

# Ordernar Modalidades
modality_order = ['CT','DX','MG']
df['Modality'] = pd.Categorical(df['Modality'], categories=modality_order)



###################################################################################################################################################################################################
sns.set_theme(style="whitegrid", palette="Set2", context="notebook")


# Calcular média e desvio padrão para anotação
stats = df.groupby(['Codec', 'Modality'])['Bpp'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df, 
    x='Codec', 
    y='Bpp',
    hue='Modality', 
    alpha=0.8, 
    kind='bar',
    aspect=1.8,
    legend=True, 
    legend_out=False,
    
    errorbar=('sd')
)

# Adiciona rótulos de desvio padrão em cima de cada barra
for i, c in enumerate(figure.ax.containers):
    modality = modality_order[i]
    # Filtra as estatísticas para a modalidade atual e ordena pelo codec
    labels = stats[stats['Modality'] == modality].sort_values('Codec')['std'].map(lambda x: f'±{x:.2f}')
    # Cria o bar label e move um pouco pra esquerda
    for bar, label in zip(c, labels):
        x = (bar.get_x() + bar.get_width() / 2 ) - 0.025
        y = bar.get_height() + 0.1
        figure.ax.text(x, y, label, ha='center', va='bottom', rotation=90, fontsize=8, color='black')

figure.ax.set_ylim(top=df['Bpp'].max() * 1.2) # Aumenta o espaço para os rótulos
plt.xlabel('Codec')
plt.ylabel('Bits por Pixel (Bpp)')
plt.legend(title='Modalidade')


#####

figure = sns.catplot(
    data=df, 
    x='Eficiencia_Bpp', 
    y='Codec',
    hue='Modality', 
    alpha=0.8, 
    kind='bar',
    aspect=1.8,
    errorbar=None,
    legend_out=False,
    legend=True
)
plt.xlabel('Eficiência de Compressão (%)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de Bpp Comprimido ---")
Bpp_table = df.groupby(['Codec', 'Modality'])['Bpp'].agg(['mean', 'std']).unstack()
print(Bpp_table)

plt.show()




# --- Gráfico e Tabela de Tempo de Codificação ---

encoding_data = df.groupby(['Codec', 'Modality'])['Encoding_Speed_ms_MB'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df,
    x='Encoding_Speed_ms_MB',
    y='Codec',
    hue='Modality',
    alpha=0.8,
    kind='bar',
    aspect=1.8,
    legend=True,
    legend_out=False,
    errorbar=None
)
plt.xlabel('Velocidade de Codificação (ms/MB)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de Velocidade Média de Codificação (ms/MB) ---")
encoding_table = encoding_data.pivot(index='Codec', columns='Modality', values='mean')
print(encoding_table)

plt.show()

# --- Gráfico e Tabela de Tempo de Decodificação ---

decoding_data = df.groupby(['Codec', 'Modality'])['Decoding_Speed_ms_MB'].agg(['mean', 'std']).reset_index()

c = sns.catplot(
    data=df,
    x='Decoding_Speed_ms_MB',
    y='Codec',
    hue='Modality',
    alpha=0.8,
    kind='bar',
    aspect=1.8,
    legend=True,
    legend_out=False,
    errorbar=None
)
plt.xlabel('Velocidade de Decodificação (ms/MB)')
plt.ylabel('Codec')
plt.legend(title='Modalidade')

print("\n--- Tabela de Velocidade Média de Decodificação (ms/MB) ---")
decoding_table = decoding_data.pivot(index='Codec', columns='Modality', values='mean')
print(decoding_table)

plt.show()


# --- Gráfico e Tabela de PSNR ---

psnr_data = df.groupby(['Beta', 'Modality'])['PSNR_dB'].agg(['mean', 'std']).reset_index()

figure = sns.catplot(
    data=df,
    y='PSNR_dB',
    x='Modality',
    hue='Beta',
    kind='bar',
    alpha=0.8,
    aspect=1.8,
    legend=True,
    legend_out=False,
    errorbar=('sd'),
)
plt.ylabel('PSNR da Imagem Esteganografada (dB)')
plt.xlabel('Modalidade')
plt.legend(title='Beta')

# Adiciona rótulos de desvio padrão em cima de cada barra
betas = sorted(psnr_data['Beta'].unique())
for i, c in enumerate(figure.ax.containers):
    beta_value = betas[i]
    # Filtra as estatísticas para o beta atual e ordena pela modalidade
    labels = psnr_data[psnr_data['Beta'] == beta_value].set_index('Modality').loc[modality_order]['std'].map(lambda x: f'±{x:.2f}')
    
    # Adiciona os rótulos a cada barra no container
    for bar, label in zip(c, labels):
        x = (bar.get_x() + bar.get_width() / 2) - 0.025
        y = bar.get_height() + 0.1
        figure.ax.text(x, y, label, ha='center', va='bottom', rotation=90, fontsize=8, color='black')

print("\n--- Tabela de PSNR Médio (dB) ---")
psnr_table = psnr_data.pivot(index='Beta', columns='Modality', values='mean')
print(psnr_table)


plt.show()


""" # --- Gráfico de Dispersão Facetado: Bpp vs. Tempo de Codificação ---

modalities = df['Modality'].cat.categories
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

# Plotar os dados para cada modalidade
for i, modality in enumerate(modalities):
    ax = axes_flat[i]
    data_modality = df[df['Modality'] == modality]
    sns.scatterplot(
        data=data_modality,
        x='Bpp',
        y='Total_Encoding_Time_s',
        hue='Codec',
        s=120,
        alpha=0.8,
        ax=ax,
        legend=False  # Desativa legendas individuais
    )
    ax.set_title(f'Modalidade: {modality}')
    ax.set_xlabel('Bits por Pixel (Bpp)')
    ax.set_ylabel('Tempo de Codificação (s)')

# Criar a legenda no quarto subplot
handles, labels = axes_flat[0].get_legend_handles_labels()
legend_ax = axes_flat[3]
legend_ax.legend(handles, labels, loc='center', title='Codec', fontsize='large', title_fontsize='x-large')
legend_ax.axis('off')  # Esconde os eixos do subplot da legenda

fig.suptitle('Bpp vs. Tempo de Codificação por Modalidade', fontsize=16, y=1.02)
fig.tight_layout(pad=3.0)

plt.show()


# --- Gráfico de Dispersão Facetado: Bpp vs. Tempo de Decodificação ---
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes_flat = axes.flatten()

# Plotar os dados para cada modalidade
for i, modality in enumerate(modalities):
    ax = axes_flat[i]
    data_modality = df[df['Modality'] == modality]
    sns.scatterplot(
        data=data_modality,
        x='Bpp',
        y='Decoding_Time_s',
        hue='Codec',
        s=120,
        alpha=0.8,
        ax=ax,
        legend=False
    )
    ax.set_title(f'Modalidade: {modality}')
    ax.set_xlabel('Bits por Pixel (Bpp)')
    ax.set_ylabel('Tempo de Decodificação (s)')

# Criar a legenda no quarto subplot
handles, labels = axes_flat[0].get_legend_handles_labels()
legend_ax = axes_flat[3]
legend_ax.legend(handles, labels, loc='center', title='Codec', fontsize='large', title_fontsize='x-large')
legend_ax.axis('off')

fig.suptitle('Bpp vs. Tempo de Decodificação por Modalidade', fontsize=16, y=1.02)
fig.tight_layout(pad=3.0)

plt.show() """