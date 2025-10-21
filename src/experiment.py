import os
import subprocess
import re
import shutil
import sys
from codec import run_steganography_process # Importa a função que criamos

# --- CONFIGURAÇÃO DOS TESTES ---

# Imagem original que será usada em todos os testes
ORIGINAL_IMAGE_PATH = "images/dentes.dcm" 

# Mensagem a ser escondida
MESSAGE_TO_HIDE = "Este é um teste de esteganografia para avaliar a qualidade da imagem com diferentes parâmetros de configuração e compressão."

# Diretório para salvar os resultados temporários
OUTPUT_DIR = "experimentos_output"

# --- PARÂMETROS A SEREM TESTADOS (ADICIONE OS VALORES QUE QUISER) ---
BETA_VALUES = [0.4, 0.6, 0.8]
BLOCK_SIZES = [4, 8, 16]
THRESHOLD_FACTORS = [0.7, 1.0, 1.3]


def parse_mse_output(output_text):
    """Extrai os valores de MSE, PSNR e SSIM da saída de texto do mse.py."""
    try:
        mse = float(re.search(r"MSE:\s+([\d\.]+)", output_text).group(1))
        psnr = float(re.search(r"PSNR:\s+([\d\.]+)", output_text).group(1))
        ssim = float(re.search(r"SSIM:\s+([\d\.]+)", output_text).group(1))
        return mse, psnr, ssim
    except (AttributeError, ValueError):
        # Retorna None se alguma métrica não for encontrada
        return None, None, None

def run_experiments():
    """Executa o ciclo de testes com todas as combinações de parâmetros."""
    # Obter o diretório onde este script (experiment_runner.py) está localizado
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construir o caminho completo e seguro para o mse.py
    mse_script_path = os.path.join(script_dir, "mse.py")

    if not os.path.exists(ORIGINAL_IMAGE_PATH):
        print(f" Erro: Imagem original '{ORIGINAL_IMAGE_PATH}' não encontrada.")
        return

    # Limpa e cria o diretório de saída
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    
    results = []
    total_runs = len(BETA_VALUES) * len(BLOCK_SIZES) * len(THRESHOLD_FACTORS)
    current_run = 0

    print("🚀 Iniciando bateria de testes de esteganografia...")
    print(f"Total de combinações a serem testadas: {total_runs}")
    print("="*60)

    for beta in BETA_VALUES:
        for block_size in BLOCK_SIZES:
            for threshold in THRESHOLD_FACTORS:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Testando: Beta={beta}, BlockSize={block_size}, Threshold={threshold}")
                
                base_filename = f"b{beta}_bs{block_size}_tf{threshold}"
                
                # 1. Executa o processo de esteganografia
                bin_file_size, stego_dcm_path = run_steganography_process(
                    ORIGINAL_IMAGE_PATH,
                    MESSAGE_TO_HIDE,
                    OUTPUT_DIR,
                    base_filename,
                    beta,
                    block_size,
                    threshold
                )
                
                # Pula para a próxima iteração se a mensagem for muito grande
                if bin_file_size is None:
                    continue

                # 2. Executa a análise com mse.py
                print(f"   -> Analisando com mse.py...")
                mse_cmd = [sys.executable, mse_script_path, ORIGINAL_IMAGE_PATH, stego_dcm_path]
                try:
                    process_result = subprocess.run(
                        mse_cmd, 
                        capture_output=True, 
                        text=True, 
                        check=True,
                        encoding='utf-8', # Mantém para decodificação correta no script principal
                        env=env  # Adiciona o ambiente configurado com UTF-8
                    )
                    
                    # 3. Extrai os resultados da análise
                    mse, psnr, ssim = parse_mse_output(process_result.stdout)
                    
                    if psnr is not None:
                        print(f"   -> Resultado: PSNR={psnr:.2f} dB, Tamanho={bin_file_size} bytes")
                        results.append({
                            "beta": beta,
                            "block_size": block_size,
                            "threshold": threshold,
                            "psnr": psnr,
                            "ssim": ssim,
                            "mse": mse,
                            "file_size": bin_file_size
                        })
                    else:
                        print("   -> Erro ao extrair métricas do mse.py")

                except subprocess.CalledProcessError as e:
                    print(f"   -> Erro ao executar mse.py: {e}")
                    print("--- Saída de Erro do Subprocesso ---")
                    print(e.stderr)
                    print("------------------------------------")

    return results

def display_results(results):
    """Exibe os resultados coletados em uma tabela organizada."""
    if not results:
        print("\nNenhum resultado foi gerado. Verifique as configurações e possíveis erros.")
        return

    print("\n\n" + "="*80)
    print("🏆 RESULTADOS FINAIS DA ANÁLISE COMPARATIVA 🏆")
    print("="*80)

    # Ordena os resultados pelo PSNR (maior para menor) como principal critério de qualidade
    sorted_results = sorted(results, key=lambda x: x['psnr'], reverse=True)
    
    print(f"{'Beta':<6} {'Block':<6} {'Thrsh':<6} | {'PSNR (dB)':<12} {'SSIM':<10} {'MSE':<12} {'Tamanho (bytes)':<15}")
    print("-" * 80)
    
    for r in sorted_results:
        print(f"{r['beta']:<6} {r['block_size']:<6} {r['threshold']:<6.1f} | {r['psnr']:<12.2f} {r['ssim']:<10.4f} {r['mse']:<12.4f} {r['file_size']:,}")

    print("-" * 80)
    print("\n💡 Interpretação:")
    print("   - PSNR Alto: Melhor qualidade de imagem (mais imperceptível).")
    print("   - SSIM Próximo de 1: Estrutura da imagem muito bem preservada.")
    print("   - Tamanho Baixo: Mais eficiente em termos de espaço.")
    print("\n   A 'combinação perfeita' é um equilíbrio: busque o maior PSNR/SSIM com o menor tamanho de arquivo possível.")


if __name__ == "__main__":
    test_results = run_experiments()
    display_results(test_results)
    
    # Pergunta se o usuário quer limpar os arquivos gerados
    cleanup = input("\nLimpar arquivos gerados no diretório '{}'? (s/n): ".format(OUTPUT_DIR))
    if cleanup.lower() == 's':
        shutil.rmtree(OUTPUT_DIR)
        print(f"Diretório '{OUTPUT_DIR}' removido.")