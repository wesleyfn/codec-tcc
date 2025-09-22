# Codec TCC - Projeto de Esteganografia

Este projeto implementa técnicas de esteganografia para imagens DICOM usando PEE (Prediction Error Expansion).

## Configuração do Ambiente

### 1. Criar e ativar ambiente virtual
```bash
# Criar ambiente virtual
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/Mac
source venv/bin/activate
```

### 2. Instalar dependências
```bash
pip install -r requirements.txt
```

### 3. Desativar ambiente virtual (quando terminar)
```bash
deactivate
```

## Dependências do Projeto

- **numpy**: Cálculos numéricos e manipulação de arrays
- **pillow**: Manipulação de imagens
- **pydicom**: Leitura e escrita de arquivos DICOM  
- **gdcm**: Compressão de imagens DICOM

## Estrutura do Projeto

```
├── src/
│   ├── encode.py     # Codificação esteganográfica
│   ├── decode.py     # Decodificação esteganográfica  
│   ├── tools.py      # Funções utilitárias
│   └── mse.py        # Cálculo de MSE
├── images/           # Imagens DICOM de entrada
├── output/           # Resultados da codificação
├── venv/             # Ambiente virtual (ignorado pelo git)
└── requirements.txt  # Dependências do projeto
```

## Como usar

1. Ative o ambiente virtual
2. Execute o encoder: `python src/encode.py`  
3. Execute o decoder: `python src/decode.py`

## Importante

- Sempre use o ambiente virtual para executar o projeto
- Não instale os pacotes globalmente
- Use `pip freeze > requirements.txt` para atualizar dependências se necessário