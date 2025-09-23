import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- 1. Simulação e Preparação dos Dados ---
# Como não temos os dados reais, vamos criar um DataFrame simulado
# que se assemelha à estrutura descrita.

def criar_dados_simulados(num_amostras=5000):
    """Cria um DataFrame com dados simulados para o problema."""
    # Data range (aproximadamente 2008 até agora, em dias úteis)
    datas = pd.to_datetime(np.arange(0, num_amostras), unit='D', origin='2008-01-01')
    
    # Fatores de Risco (o que queremos gerar)
    # 8 parâmetros para curvas, 8 para superfícies, 2 para preços
    dados = {
        **{f'p_curva_{i+1}': np.random.randn(num_amostras) for i in range(8)},
        **{f'p_vol_{i+1}': np.random.rand(num_amostras) for i in range(8)},
        'preco_ativo_A': np.random.uniform(50, 150, num_amostras) * (1 + np.sin(np.arange(num_amostras)/100)),
        'preco_ativo_B': np.random.uniform(20, 80, num_amostras) * (1 + np.cos(np.arange(num_amostras)/50))
    }
    
    # Variáveis Econômicas (condições para a geração)
    condicoes = {
        'inflacao_br': np.random.uniform(2, 12, num_amostras),
        'juros_eua': np.random.uniform(0, 5, num_amostras),
        'ibov': np.random.uniform(70000, 130000, num_amostras),
        'balanca_comercial': np.random.uniform(-5, 10, num_amostras)
    }
    
    df = pd.DataFrame({**dados, **condicoes}, index=datas)
    
    # Adicionar alguma correlação simples para tornar o problema mais realista
    df['p_curva_1'] += df['juros_eua'] * 0.5
    df['preco_ativo_A'] -= df['inflacao_br'] * 2
    
    print("Dados simulados criados com sucesso.")
    print(f"Shape do DataFrame: {df.shape}")
    print("Colunas de Fatores de Risco (Exemplo):", list(dados.keys())[:3])
    print("Colunas de Condições (Exemplo):", list(condicoes.keys())[:3])
    
    return df

# Nomes das colunas para fácil acesso
colunas_fatores_risco = [f'p_curva_{i+1}' for i in range(8)] + \
                        [f'p_vol_{i+1}' for i in range(8)] + \
                        ['preco_ativo_A', 'preco_ativo_B']

colunas_condicionais = ['inflacao_br', 'juros_eua', 'ibov', 'balanca_comercial']

# Criar os dados
df = criar_dados_simulados()

# Normalização dos dados: Essencial para redes neurais.
# Usaremos MinMaxScaler para colocar todos os valores no intervalo [-1, 1],
# que funciona bem com a função de ativação tanh na saída do gerador.
scaler_fatores = MinMaxScaler(feature_range=(-1, 1))
scaler_condicoes = MinMaxScaler(feature_range=(-1, 1))

df[colunas_fatores_risco] = scaler_fatores.fit_transform(df[colunas_fatores_risco])
df[colunas_condicionais] = scaler_condicoes.fit_transform(df[colunas_condicionais])

print("\nDados normalizados para o intervalo [-1, 1].")

# Converter para tensores do PyTorch
fatores_tensor = torch.FloatTensor(df[colunas_fatores_risco].values)
condicoes_tensor = torch.FloatTensor(df[colunas_condicionais].values)

# Criar DataLoader para carregar os dados em lotes (batches)
batch_size = 64
dataset = TensorDataset(fatores_tensor, condicoes_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 2. Definição da Arquitetura da cGAN ---

# Dimensões
fatores_dim = len(colunas_fatores_risco) # e.g., 8+8+2 = 18
cond_dim = len(colunas_condicionais)     # e.g., 4
latent_dim = 32 # Dimensão do vetor de ruído (pode ser ajustado)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # A entrada é a concatenação do ruído e das condições
            nn.Linear(latent_dim + cond_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, fatores_dim),
            # Tanh garante que a saída esteja no mesmo intervalo dos dados normalizados [-1, 1]
            nn.Tanh()
        )

    def forward(self, z, cond):
        # Concatena o vetor de ruído (z) com o vetor de condições (cond)
        input_vec = torch.cat((z, cond), dim=1)
        return self.model(input_vec)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # A entrada é a concatenação dos fatores de risco e das condições
            nn.Linear(fatores_dim + cond_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            
            # Camada de saída com 1 neurônio para classificação (real vs. falso)
            nn.Linear(256, 1)
            # Não usamos Sigmoid aqui, pois a função de perda BCEWithLogitsLoss é mais estável
        )

    def forward(self, fatores, cond):
        # Concatena o vetor de fatores de risco com o vetor de condições
        input_vec = torch.cat((fatores, cond), dim=1)
        return self.model(input_vec)

# --- 3. Loop de Treinamento ---

# Instanciar modelos
generator = Generator()
discriminator = Discriminator()

# Otimizadores (Adam é uma boa escolha padrão)
lr = 0.0002
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Função de Perda
adversarial_loss = nn.BCEWithLogitsLoss()

# Mover modelos para a GPU se disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsando dispositivo: {device}")
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)


# Iniciar o treinamento
num_epochs = 200 # Aumente para um treinamento real (e.g., 500, 1000 ou mais)

print("\n--- Iniciando Treinamento ---")
for epoch in range(num_epochs):
    for i, (fatores_reais, condicoes_reais) in enumerate(dataloader):
        
        # Mover dados para o dispositivo
        fatores_reais = fatores_reais.to(device)
        condicoes_reais = condicoes_reais.to(device)
        
        # Labels para dados reais (1) e falsos (0)
        real_labels = torch.ones(fatores_reais.size(0), 1, device=device)
        fake_labels = torch.zeros(fatores_reais.size(0), 1, device=device)

        # --- Treinamento do Discriminador ---
        optimizer_d.zero_grad()

        # Perda com dados reais
        output_real = discriminator(fatores_reais, condicoes_reais)
        loss_d_real = adversarial_loss(output_real, real_labels)

        # Gerar dados falsos
        z = torch.randn(fatores_reais.size(0), latent_dim, device=device)
        fatores_falsos = generator(z, condicoes_reais)
        
        # Perda com dados falsos (usamos .detach() para não calcular gradientes para o gerador aqui)
        output_fake = discriminator(fatores_falsos.detach(), condicoes_reais)
        loss_d_fake = adversarial_loss(output_fake, fake_labels)

        # Perda total do discriminador e atualização
        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()
        
        # --- Treinamento do Gerador ---
        optimizer_g.zero_grad()
        
        # O objetivo do gerador é enganar o discriminador.
        # Portanto, calculamos a perda dos dados falsos, mas com labels de "reais".
        output_g = discriminator(fatores_falsos, condicoes_reais)
        loss_g = adversarial_loss(output_g, real_labels)

        # Atualização do gerador
        loss_g.backward()
        optimizer_g.step()

    # Imprimir progresso
    if (epoch + 1) % 20 == 0:
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"[D loss: {loss_d.item():.4f}] "
            f"[G loss: {loss_g.item():.4f}]"
        )
        
print("--- Treinamento Concluído ---")


# --- 4. Geração de Cenários de Estresse ---

def gerar_cenario_estresse(generator_model, cenario_economico, n_amostras=1):
    """
    Gera um ou mais cenários de fatores de risco dado um conjunto de condições econômicas.
    
    Args:
        generator_model: O modelo gerador treinado.
        cenario_economico (dict): Um dicionário com os valores das variáveis econômicas.
        n_amostras (int): Quantos cenários diferentes gerar para a mesma condição.
    
    Returns:
        pd.DataFrame: Um DataFrame com os cenários gerados em sua escala original.
    """
    generator_model.eval() # Coloca o modelo em modo de avaliação
    
    # 1. Preparar o vetor de condição
    # Certifique-se de que a ordem das colunas é a mesma do treinamento
    cond_df = pd.DataFrame([cenario_economico], columns=colunas_condicionais)
    
    # 2. Normalizar a condição usando o MESMO scaler do treinamento
    cond_norm = scaler_condicoes.transform(cond_df)
    
    # 3. Converter para tensor e replicar se necessário
    cond_tensor = torch.FloatTensor(cond_norm).to(device)
    if n_amostras > 1:
        cond_tensor = cond_tensor.repeat(n_amostras, 1)

    # 4. Gerar o ruído latente
    z = torch.randn(n_amostras, latent_dim, device=device)
    
    # 5. Gerar o cenário
    with torch.no_grad(): # Não precisamos calcular gradientes aqui
        cenario_gerado_norm = generator_model(z, cond_tensor)
    
    # 6. Mover para CPU e converter para numpy
    cenario_gerado_norm = cenario_gerado_norm.cpu().numpy()
    
    # 7. Desnormalizar o resultado para a escala original
    cenario_gerado_original = scaler_fatores.inverse_transform(cenario_gerado_norm)
    
    # 8. Retornar como um DataFrame para fácil interpretação
    df_resultado = pd.DataFrame(cenario_gerado_original, columns=colunas_fatores_risco)
    
    return df_resultado

# Exemplo de uso:
print("\n--- Gerando Cenário de Estresse ---")

# Defina um cenário macroeconômico de estresse
# (valores hipotéticos altos para inflação e juros externos)
cenario_estresse = {
    'inflacao_br': 15.0,
    'juros_eua': 5.5,
    'ibov': 85000,
    'balanca_comercial': -10.0
}

# Gerar 5 cenários "diferentes" para esta mesma condição macroeconômica
cenarios_gerados = gerar_cenario_estresse(generator, cenario_estresse, n_amostras=5)

print(f"\nCenário Macroeconômico de Estresse Fornecido:\n {cenario_estresse}")
print("\nFatores de Risco Gerados pela GAN (5 amostras):\n")
print(cenarios_gerados)
