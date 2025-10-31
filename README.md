# Codec Híbrido DICOM (Compressão Lossless + Esteganografia Adaptativa)

Este repositório contém o protótipo de software desenvolvido para o Trabalho de Conclusão de Curso (TCC) intitulado: **"Desenvolvimento de um Codec Híbrido para Compressão Lossless e Esteganografia Adaptativa em Imagens DICOM"**.

O objetivo deste projeto é criar e avaliar um codec pragmático que integra compressão sem perdas de imagens médicas DICOM com esteganografia adaptativa e reversível, visando um equilíbrio entre eficiência de compressão, segurança da informação e viabilidade computacional.

## O Problema

Imagens médicas no padrão DICOM (`Digital Imaging and Communications in Medicine`) apresentam um desafio duplo:

1.  **Volume (Compressão):** São arquivos volumosos que exigem compressão eficiente para reduzir custos de armazenamento e tempo de transmissão.
2.  **Privacidade (Segurança):** Contêm metadados highly sensíveis (Nome, ID do Paciente, datas) protegidos por leis como a LGPD e HIPAA, que precisam ser protegidos.

Abordagens de ponta (Estado da Arte), como a de Zheng et al. (2025), unem compressão e segurança usando arquiteturas complexas (LLMs, VAEs), tornando-as computacionalmente caras e lentas. Este projeto propõe uma solução prática que atinge objetivos similares com ferramentas acessíveis.

## Funcionalidades (Features)

* **Esteganografia de Metadados:** Extrai automaticamente todos os metadados DICOM originais, serializa-os em JSON e os utiliza como a "mensagem secreta".
* **Adaptação Dupla (Segurança):** A esteganografia não é aleatória; ela é duplamente adaptativa para maximizar a segurança estatística (indetectabilidade):
    1.  **Adaptação de Plano de Bits:** Decompõe a imagem em modalidades "global" (estrutura) e "local" (textura/ruído) usando *Bit Plane Slicing* (BPS) adaptativo baseado em informação mútua. A mensagem é embutida apenas nos planos locais.
    2.  **Adaptação Espacial:** Cria um mapa de capacidade analisando a variância (complexidade) de blocos da imagem. A mensagem é embutida apenas nas regiões de alta complexidade.
* **Reversibilidade Total (Lossless):** Utiliza um *embedding map* (`bitmaps_blob`) para rastrear cada bit modificado, garantindo que a imagem original possa ser reconstruída bit-a-bit após a extração.
* **Compressão Lossless:** Comprime a imagem *stego* intermediária usando codecs modernos (JPEG XL) ou benchmarks (JPEG 2000, JPEG-LS, Deflate/PNG).
* **Formato Contenedor:** Empacota todos os dados (cabeçalho de parâmetros, *embedding map* comprimido com `zlib`, e imagem comprimida) em um arquivo binário autossuficiente (`.bin`).
* **Negação Plausível (Isca):** Opcionalmente, gera um arquivo DICOM (`_stego.dcm`) com metadados falsos e anônimos ("decoy"), enquanto os metadados reais estão ocultos nos pixels.

## Como Funciona (Arquitetura)

O protótipo opera em dois fluxos principais: codificação e decodificação.

### Fluxo de Codificação (`run_steganography`)

1.  **Entrada:** Arquivo DICOM original (`.dcm`).
2.  **Extração:** O `pixel_array` é extraído e os metadados são serializados em `metadata_json`.
3.  **Análise:**
    * `decompose_image_adaptively` (BPS) separa a imagem em `global_planes` e `local_planes`.
    * `create_embedding_capacity_map` identifica os `allowed_indices` (pixels complexos).
4.  **Embutimento:** `embed_message_in_planes` insere os bits do `metadata_json` nos `local_planes` (apenas nos `allowed_indices`), gerando os `stego_planes` e o `embedding_map`.
5.  **Reconstrução:** `merge_bit_planes` combina os `global_planes` (intactos) com os `stego_planes` (modificados) para criar o `stego_image_array`.
6.  **Compressão:**
    * `compress_image_data` comprime o `stego_image_array` (ex: JPEG XL).
    * `zlib.compress` comprime o `embedding_map`.
7.  **Empacotamento:** `create_steganography_container` grava a assinatura `STGC`, o cabeçalho (`header`), o `bitmaps_blob` e os dados da imagem comprimida no arquivo `.bin` final.

### Fluxo de Decodificação (`decode_steganography_container`)

1.  **Entrada:** Arquivo `.bin` (container).
2.  **Leitura:** `parse_steganography_file` lê a assinatura `STGC`, o cabeçalho (`metadata`), o `bitmaps_data` e o `stego_image_data`.
3.  **Descompressão:**
    * `decompress_image_data` decodifica o `stego_image_data` (ex: `djxl`) para obter o `stego_array`.
    * `zlib.decompress` decodifica o `bitmaps_data` para obter o `embedding_map`.
4.  **Extração e Reversão:**
    * `extract_message_and_restore_planes` usa o `embedding_map` para ler os bits do `stego_array` (recriando o `metadata_json`) e simultaneamente reverter os pixels modificados, gerando os `restored_local_planes`.
5.  **Reconstrução:** `merge_bit_planes` combina os `global_planes` com os `restored_local_planes` para obter o `restored_image_array` (idêntico ao original).
6.  **Saída:** `restore_dicom_metadata` aplica o `metadata_json` extraído a um novo dataset DICOM, recriando o arquivo DICOM original.