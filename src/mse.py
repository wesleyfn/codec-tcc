import numpy as np
from PIL import Image
import os
import sys
import pydicom

class AnalisadorMSE:
    def __init__(self):
        self.resultados = []
    
    def carregar_imagem(self, caminho):
        """
        Carrega imagem suportando PNG, DICOM e outros formatos
        Retorna array numpy e valor máximo possível
        """
        if caminho.lower().endswith('.dcm'):
            # Carregar DICOM
            dcm = pydicom.dcmread(caminho)
            img_array = dcm.pixel_array
            
            # Handle multi-frame DICOM
            if len(img_array.shape) > 2:
                img_array = img_array[0]
            
            # Converter signed para unsigned se necessário
            if img_array.dtype == np.int16:
                img_array = img_array.astype(np.uint16)
            
            # Determinar bits efetivos
            bits_stored = getattr(dcm, 'BitsStored', img_array.dtype.itemsize * 8)
            max_valor = (1 << bits_stored) - 1
            
            print(f"   DICOM: {img_array.shape}, {img_array.dtype}, {bits_stored} bits, range: {img_array.min()}-{img_array.max()}")
            
            return img_array.astype(np.float64), max_valor, bits_stored
            
        else:
            # Carregar PNG/outros formatos
            img = Image.open(caminho)
            
            # Detectar modo da imagem
            if img.mode == 'I;16':
                # PNG 16-bit
                img_array = np.array(img, dtype=np.uint16)
                max_valor = 65535
                bits_stored = 16
            elif img.mode in ['L', 'P']:
                # Grayscale 8-bit
                img_array = np.array(img.convert('L'), dtype=np.uint8)
                max_valor = 255
                bits_stored = 8
            elif img.mode in ['RGB', 'RGBA']:
                # RGB para grayscale
                img_gray = img.convert('L')
                img_array = np.array(img_gray, dtype=np.uint8)
                max_valor = 255
                bits_stored = 8
            else:
                # Tentar conversão genérica
                img_array = np.array(img)
                if img_array.dtype == np.uint16:
                    max_valor = 65535
                    bits_stored = 16
                else:
                    max_valor = 255
                    bits_stored = 8
            
            print(f"   IMG: {img_array.shape}, {img_array.dtype}, {bits_stored} bits, range: {img_array.min()}-{img_array.max()}")
            
            return img_array.astype(np.float64), max_valor, bits_stored
        
    def calcular_mse(self, imagem1, imagem2):
        """
        Calcula Mean Squared Error entre duas imagens
        MSE = (1/MN) * Σ(I1[i,j] - I2[i,j])²
        
        Suporta imagens de 8 a 16 bits (PNG, DICOM)
        """
        # Carrega imagens com suporte a 16 bits
        if isinstance(imagem1, str):
            img1, max_val1, bits1 = self.carregar_imagem(imagem1)
        else:
            img1 = np.array(imagem1, dtype=np.float64)
            max_val1 = img1.max()
            bits1 = 16 if max_val1 > 255 else 8
            
        if isinstance(imagem2, str):
            img2, max_val2, bits2 = self.carregar_imagem(imagem2)
        else:
            img2 = np.array(imagem2, dtype=np.float64)
            max_val2 = img2.max()
            bits2 = 16 if max_val2 > 255 else 8
        
        # Verifica compatibilidade
        if img1.shape != img2.shape:
            raise ValueError(f"Dimensões diferentes: {img1.shape} vs {img2.shape}")
        
        # Normalizar para mesmo range se necessário
        if max_val1 != max_val2:
            print(f"   ⚠️  Normalizando ranges diferentes: {max_val1} vs {max_val2}")
            # Normaliza ambas para [0,1] e depois para o maior range
            max_range = max(max_val1, max_val2)
            img1_norm = (img1 / max_val1) * max_range
            img2_norm = (img2 / max_val2) * max_range
        else:
            img1_norm = img1
            img2_norm = img2
            max_range = max_val1
        
        # Calcula MSE
        diferenca = img1_norm - img2_norm
        mse = np.mean(diferenca ** 2)
        
        return mse, max_range
    
    def calcular_psnr(self, mse, max_valor=None):
        """
        Calcula PSNR adaptado para imagens de até 16 bits
        PSNR = 10 * log10(MAX²/MSE)
        
        Se max_valor não for fornecido, assume 255 (8 bits)
        Para DICOM 16 bits, o max_valor pode ir até 65535
        """
        if mse == 0:
            return float('inf')  # Imagens idênticas
        
        if max_valor is None:
            max_valor = 255  # Default para 8 bits
        
        psnr = 10 * np.log10((max_valor ** 2) / mse)
        return psnr
    
    def calcular_ssim_simples(self, imagem1, imagem2):
        """
        Calcula SSIM adaptado para imagens de até 16 bits
        """
        # Carrega imagens com suporte a 16 bits
        if isinstance(imagem1, str):
            img1, max_val1, _ = self.carregar_imagem(imagem1)
        else:
            img1 = np.array(imagem1, dtype=np.float64)
            max_val1 = img1.max()
            
        if isinstance(imagem2, str):
            img2, max_val2, _ = self.carregar_imagem(imagem2)
        else:
            img2 = np.array(imagem2, dtype=np.float64)
            max_val2 = img2.max()
        
        # Usar o maior range para as constantes SSIM
        max_range = max(max_val1, max_val2)
        
        # Normalizar se necessário
        if max_val1 != max_val2:
            img1_norm = (img1 / max_val1) * max_range
            img2_norm = (img2 / max_val2) * max_range
        else:
            img1_norm = img1
            img2_norm = img2
        
        # Parâmetros SSIM
        mu1 = np.mean(img1_norm)
        mu2 = np.mean(img2_norm)
        sigma1 = np.var(img1_norm)
        sigma2 = np.var(img2_norm)
        sigma12 = np.mean((img1_norm - mu1) * (img2_norm - mu2))
        
        # Constantes SSIM adaptadas para o range da imagem
        c1 = (0.01 * max_range) ** 2
        c2 = (0.03 * max_range) ** 2
        
        # Fórmula SSIM
        numerador = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
        denominador = (mu1**2 + mu2**2 + c1) * (sigma1 + sigma2 + c2)
        
        ssim = numerador / denominador
        return ssim
    
    def analisar_par_imagens(self, imagem_original, imagem_stego, nome_par=""):
        """
        Analisa um par de imagens (original vs stego) com suporte a 16 bits
        """
        print(f"🔍 Analisando: {nome_par}")
        print("─" * 50)
        
        try:
            # Carrega e verifica as imagens com suporte a 16 bits
            img_orig, max_val_orig, tipo_orig = self.carregar_imagem(imagem_original)
            img_stego, max_val_stego, tipo_stego = self.carregar_imagem(imagem_stego)
            
            print(f"📁 Original:  {os.path.basename(imagem_original)} - {img_orig.shape} - {tipo_orig} (max: {max_val_orig})")
            print(f"📁 Stego:     {os.path.basename(imagem_stego)} - {img_stego.shape} - {tipo_stego} (max: {max_val_stego})")
            
            # Calcula métricas adaptadas para 16 bits
            mse, max_range = self.calcular_mse(imagem_original, imagem_stego)
            psnr = self.calcular_psnr(mse, max_range)
            ssim = self.calcular_ssim_simples(imagem_original, imagem_stego)
            
            # Análise estatística com dados já carregados
            array_orig = np.array(img_orig, dtype=np.float64)
            array_stego = np.array(img_stego, dtype=np.float64)
            
            diferenca_media = np.mean(np.abs(array_orig - array_stego))
            diferenca_max = np.max(np.abs(array_orig - array_stego))
            pixels_diferentes = np.sum(array_orig != array_stego)
            total_pixels = array_orig.size
            percentual_mudanca = (pixels_diferentes / total_pixels) * 100
            
            # Resultados
            print(f"\n📊 MÉTRICAS DE QUALIDADE:")
            print(f"   MSE:  {mse:.4f}")
            print(f"   PSNR: {psnr:.2f} dB")
            print(f"   SSIM: {ssim:.4f}")
            
            print(f"\n📈 ANÁLISE DE DIFERENÇAS:")
            print(f"   Diferença média:     {diferenca_media:.2f}")
            print(f"   Diferença máxima:    {diferenca_max:.0f}")
            print(f"   Pixels alterados:    {pixels_diferentes:,}/{total_pixels:,}")
            print(f"   Percentual mudança:  {percentual_mudanca:.2f}%")
            
            # Interpretação
            print(f"\n💡 INTERPRETAÇÃO:")
            if mse == 0:
                print("   ✅ Imagens IDÊNTICAS")
            elif psnr > 40:
                print("   ✅ Qualidade EXCELENTE (steganografia imperceptível)")
            elif psnr > 30:
                print("   ✅ Qualidade BOA (alterações mínimas)")
            elif psnr > 20:
                print("   ⚠️  Qualidade REGULAR (alterações visíveis)")
            else:
                print("   ❌ Qualidade BAIXA (alterações significativas)")
            
            if ssim > 0.95:
                print("   ✅ Estrutura MUITO bem preservada")
            elif ssim > 0.8:
                print("   ✅ Estrutura bem preservada")
            else:
                print("   ⚠️  Estrutura parcialmente alterada")
            
            # Armazena resultados
            resultado = {
                'nome': nome_par,
                'original': imagem_original,
                'stego': imagem_stego,
                'mse': mse,
                'psnr': psnr,
                'ssim': ssim,
                'diferenca_media': diferenca_media,
                'diferenca_max': diferenca_max,
                'percentual_mudanca': percentual_mudanca
            }
            
            self.resultados.append(resultado)
            return resultado
            
        except Exception as e:
            print(f"❌ Erro na análise: {str(e)}")
            return None
    

    
    def analisar_multiplos_pares(self, pares_imagens):
        """
        Analisa múltiplos pares de imagens e gera comparação
        pares_imagens: lista de tuplas (original, stego, nome)
        """
        print("🔬 ANÁLISE COMPARATIVA DE MÚLTIPLOS PARES")
        print("=" * 80)
        
        resultados = []
        for original, stego, nome in pares_imagens:
            if os.path.exists(original) and os.path.exists(stego):
                resultado = self.analisar_par_imagens(original, stego, nome)
                if resultado:
                    resultados.append(resultado)
                print()
            else:
                print(f"❌ Arquivos não encontrados para {nome}")
                print()
        
        # Resumo comparativo
        if resultados:
            print("📋 RESUMO COMPARATIVO:")
            print("─" * 80)
            print(f"{'NOME':<20} {'MSE':<10} {'PSNR':<10} {'SSIM':<10} {'QUALIDADE':<15}")
            print("─" * 80)
            
            for r in resultados:
                qualidade = "EXCELENTE" if r['psnr'] > 40 else "BOA" if r['psnr'] > 30 else "REGULAR"
                print(f"{r['nome']:<20} {r['mse']:<10.4f} {r['psnr']:<10.2f} {r['ssim']:<10.4f} {qualidade:<15}")
        
        return resultados

    def gerar_relatorio(self, salvar_arquivo=True):
        """
        Gera relatório completo das análises
        """
        if not self.resultados:
            print("❌ Nenhuma análise realizada ainda!")
            return
        
        print(f"\n📊 RELATÓRIO COMPLETO - {len(self.resultados)} ANÁLISES")
        print("═" * 70)
        
        # Estatísticas gerais
        mses = [r['mse'] for r in self.resultados]
        psnrs = [r['psnr'] for r in self.resultados if r['psnr'] != float('inf')]
        ssims = [r['ssim'] for r in self.resultados]
        
        print(f"\n📈 ESTATÍSTICAS GERAIS:")
        print(f"   MSE médio:  {np.mean(mses):.4f} (min: {np.min(mses):.4f}, max: {np.max(mses):.4f})")
        if psnrs:
            print(f"   PSNR médio: {np.mean(psnrs):.2f} dB (min: {np.min(psnrs):.2f}, max: {np.max(psnrs):.2f})")
        print(f"   SSIM médio: {np.mean(ssims):.4f} (min: {np.min(ssims):.4f}, max: {np.max(ssims):.4f})")
        
        # Tabela resumo
        print(f"\n📋 TABELA RESUMO:")
        print(f"{'Nome':<20} {'MSE':<12} {'PSNR':<10} {'SSIM':<8} {'Mudança%':<10}")
        print("─" * 70)
        
        for r in self.resultados:
            psnr_str = f"{r['psnr']:.2f}" if r['psnr'] != float('inf') else "∞"
            print(f"{r['nome']:<20} {r['mse']:<12.4f} {psnr_str:<10} {r['ssim']:<8.4f} {r['percentual_mudanca']:<10.2f}")
        
        # Salva em arquivo se solicitado
        if salvar_arquivo:
            nome_arquivo = "relatorio_mse.txt"
            with open(nome_arquivo, 'w', encoding='utf-8') as f:
                f.write("RELATÓRIO DE ANÁLISE MSE - IMAGENS STEGO\n")
                f.write("=" * 50 + "\n\n")
                
                for r in self.resultados:
                    f.write(f"Análise: {r['nome']}\n")
                    f.write(f"Original: {r['original']}\n")
                    f.write(f"Stego: {r['stego']}\n")
                    f.write(f"MSE: {r['mse']:.4f}\n")
                    f.write(f"PSNR: {r['psnr']:.2f} dB\n")
                    f.write(f"SSIM: {r['ssim']:.4f}\n")
                    f.write(f"Percentual mudança: {r['percentual_mudanca']:.2f}%\n")
                    f.write("-" * 30 + "\n")
                
                f.write(f"\nEstatísticas gerais:\n")
                f.write(f"MSE médio: {np.mean(mses):.4f}\n")
                if psnrs:
                    f.write(f"PSNR médio: {np.mean(psnrs):.2f} dB\n")
                f.write(f"SSIM médio: {np.mean(ssims):.4f}\n")
            
            print(f"\n💾 Relatório salvo em: {nome_arquivo}")

def main():
    if len(sys.argv) != 3:
        print("💡 Uso do Analisador MSE:")
        print("   python mse.py <imagem_original> <imagem_stego>")
        print("")
        print("📁 Exemplo:")
        print("   python mse.py imagem.png imagem_stego.png")
        return
    
    # Análise de par específico
    imagem_original = sys.argv[1]
    imagem_stego = sys.argv[2]
    nome_analise = os.path.splitext(os.path.basename(imagem_original))[0]
    
    # Verifica se os arquivos existem
    if not os.path.exists(imagem_original):
        print(f"❌ Arquivo não encontrado: {imagem_original}")
        return
    
    if not os.path.exists(imagem_stego):
        print(f"❌ Arquivo não encontrado: {imagem_stego}")
        return
    
    # Analisa o par
    analisador = AnalisadorMSE()
    resultado = analisador.analisar_par_imagens(imagem_original, imagem_stego, nome_analise)
    
    if resultado:
        print(f"\n✅ Análise MSE concluída com sucesso!")
        print(f"📊 MSE: {resultado['mse']:.4f}")
        print(f"📊 PSNR: {resultado['psnr']:.2f} dB")
        print(f"📊 SSIM: {resultado['ssim']:.4f}")

if __name__ == "__main__":
    main()
