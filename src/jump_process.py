import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

# Incluindo a função de simulação da resposta anterior para gerar dados de teste
def simular_processo_salto_reversao_media(
    x0: float, mu: float, theta: float, sigma: float,
    lambda_salto: float, mu_salto: float, sigma_salto: float,
    T: int, N: int
) -> tuple[np.ndarray, np.ndarray]:
    dt = T / N
    t = np.linspace(0, T, N + 1)
    x = np.zeros(N + 1)
    x[0] = x0
    Z_difusao = np.random.standard_normal(N)
    num_saltos = np.random.poisson(lambda_salto * dt, N)
    for i in range(1, N + 1):
        drift = theta * (mu - x[i-1]) * dt
        diffusion = sigma * np.sqrt(dt) * Z_difusao[i-1]
        salto_total = 0
        if num_saltos[i-1] > 0:
            tamanhos_salto = np.random.normal(mu_salto, sigma_salto, num_saltos[i-1])
            salto_total = np.sum(tamanhos_salto)
        x[i] = x[i-1] + drift + diffusion + salto_total
    return t, x


def identificar_saltos(retornos: np.ndarray, limiar_std: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Separa os retornos em componentes de salto e difusão com base em um limiar.
    """
    std_retornos = np.std(retornos)
    limiar_absoluto = limiar_std * std_retornos
    
    mascara_salto = np.abs(retornos) > limiar_absoluto
    
    retornos_salto = retornos[mascara_salto]
    retornos_difusao = retornos[~mascara_salto]
    
    return retornos_salto, retornos_difusao, mascara_salto


def calibrar_parametros_salto(retornos_salto: np.ndarray, T: float) -> Dict[str, float]:
    """
    Calibra os parâmetros da componente de salto.
    """
    num_saltos = len(retornos_salto)
    if num_saltos == 0:
        return {'lambda_estimado': 0, 'mu_J_estimado': 0, 'sigma_J_estimado': 0}
        
    lambda_estimado = num_saltos / T
    mu_J_estimado = np.mean(retornos_salto)
    sigma_J_estimado = np.std(retornos_salto)
    
    return {
        'lambda_estimado': lambda_estimado,
        'mu_J_estimado': mu_J_estimado,
        'sigma_J_estimado': sigma_J_estimado
    }

def calibrar_parametros_ou(precos: np.ndarray, dt: float) -> Dict[str, float]:
    """
    Calibra os parâmetros do processo de Ornstein-Uhlenbeck via regressão linear.
    """
    n = len(precos)
    # Regressão de X_t contra X_{t-1}
    X_t = precos[1:]
    X_t_menos_1 = precos[:-1]
    
    # Usando polyfit para uma regressão linear simples (y = b*x + a)
    # Nota: polyfit retorna [b, a] para grau 1
    b, a = np.polyfit(X_t_menos_1, X_t, 1)

    # Calcular resíduos
    residuos = X_t - (b * X_t_menos_1 + a)
    
    # Recuperar parâmetros a partir dos coeficientes da regressão
    theta_estimado = (1 - b) / dt
    mu_estimado = a / (theta_estimado * dt)
    sigma_estimado = np.std(residuos) / np.sqrt(dt)
    
    return {
        'mu_estimado': mu_estimado,
        'theta_estimado': theta_estimado,
        'sigma_estimado': sigma_estimado
    }

def calibrar_processo_completo(
    precos: np.ndarray, 
    T: float, 
    limiar_std_salto: float = 2.0
) -> Dict[str, float]:
    """
    Orquestra o processo completo de calibração.
    
    Args:
        precos (np.ndarray): Série histórica de preços.
        T (float): Período total em anos que os dados representam.
        limiar_std_salto (float): Número de desvios padrão para identificar um salto.
    """
    N = len(precos) - 1
    dt = T / N
    
    # Usamos log-retornos pois são aditivos e têm propriedades estatísticas melhores
    log_retornos = np.log(precos[1:] / precos[:-1])
    
    # Passo 1: Identificar saltos
    retornos_salto, retornos_difusao, mascara_salto = identificar_saltos(log_retornos, limiar_std_salto)
    
    # Passo 2: Calibrar parâmetros de salto
    params_salto = calibrar_parametros_salto(retornos_salto, T)
    
    # Passo 3: Calibrar parâmetros de OU
    # Criamos uma máscara booleana para os preços que NÃO antecedem um salto
    mascara_precos_difusao = np.insert(~mascara_salto, 0, True)
    precos_difusao_apenas = precos[mascara_precos_difusao]

    # A calibração de OU requer pares (Xt, Xt-1) que não sejam interrompidos por um salto.
    # Filtramos para manter apenas as transições que não foram classificadas como salto.
    X_t = precos[1:][~mascara_salto]
    X_t_menos_1 = precos[:-1][~mascara_salto]
    
    # A regressão precisa ser feita com os pares contínuos
    b, a = np.polyfit(X_t_menos_1, X_t, 1)
    residuos = X_t - (b * X_t_menos_1 + a)
    theta_estimado = (1 - b) / dt
    mu_estimado = a / (theta_estimado * dt) if theta_estimado != 0 else np.mean(precos)
    sigma_estimado = np.std(residuos) / np.sqrt(dt)

    params_ou = {
        'mu_estimado': mu_estimado,
        'theta_estimado': theta_estimado,
        'sigma_estimado': sigma_estimado
    }
    
    # Combinar todos os parâmetros
    parametros_finais = {**params_ou, **params_salto}
    
    return parametros_finais


# --- Exemplo de Uso ---
if __name__ == "__main__":
    # 1. GERAR DADOS SINTÉTICOS COM PARÂMETROS CONHECIDOS
    parametros_verdadeiros = {
        'x0': 150.0, 'mu': 155.0, 'theta': 2.0, 'sigma': 20.0,
        'lambda_salto': 3.0, 'mu_salto': -0.05, 'sigma_salto': 0.03,
        'T': 5, 'N': 252 * 5
    }
    
    # Nota: mu_salto e sigma_salto são aplicados aos log-retornos,
    # por isso são valores pequenos.
    
    # Gerando a série de preços (vamos converter saltos para log-retornos)
    dt_sim = parametros_verdadeiros['T'] / parametros_verdadeiros['N']
    _, precos_simulados = simular_processo_salto_reversao_media(
        x0=parametros_verdadeiros['x0'],
        mu=parametros_verdadeiros['mu'],
        theta=parametros_verdadeiros['theta'],
        sigma=parametros_verdadeiros['sigma'],
        # Convertendo lambda para taxa de salto em log-preços
        lambda_salto=parametros_verdadeiros['lambda_salto'],
        # Usando a aproximação de que o salto no preço é exp(salto_log)
        # S_{t} = S_{t-1} * exp(J) => log(S_t) - log(S_{t-1}) = J
        # Para simular, precisamos do salto no preço, não no log-preço.
        # Um salto aditivo no log-preço é um salto multiplicativo no preço.
        # A simulação original é aditiva. Para ser consistente com a calibração, 
        # a calibração deve usar retornos aritméticos, ou a simulação deve ser multiplicativa.
        # Vamos ajustar a calibração para retornos aritméticos para manter a simplicidade.
        mu_salto = parametros_verdadeiros['mu'] * parametros_verdadeiros['mu_salto'],
        sigma_salto = parametros_verdadeiros['mu'] * parametros_verdadeiros['sigma_salto'],
        T=parametros_verdadeiros['T'],
        N=parametros_verdadeiros['N']
    )
    
    print("--- Calibração do Processo ---")
    
    # 2. CALIBRAR O PROCESSO USANDO OS DADOS GERADOS
    parametros_calibrados = calibrar_processo_completo(
        precos=precos_simulados,
        T=parametros_verdadeiros['T'],
        limiar_std_salto=2.5 # Um salto é um evento que está a 2.5 desvios padrão da média
    )
    
    # 3. COMPARAR OS RESULTADOS
    print("\nParâmetros Verdadeiros:")
    print(f"  mu: {parametros_verdadeiros['mu']:.4f}")
    print(f"  theta: {parametros_verdadeiros['theta']:.4f}")
    print(f"  sigma: {parametros_verdadeiros['sigma']:.4f}")
    print(f"  lambda: {parametros_verdadeiros['lambda_salto']:.4f}")
    # Ajustando os parâmetros de salto verdadeiros para retornos aritméticos para comparação
    print(f"  mu_J: {parametros_verdadeiros['mu'] * parametros_verdadeiros['mu_salto']:.4f}")
    print(f"  sigma_J: {parametros_verdadeiros['mu'] * parametros_verdadeiros['sigma_salto']:.4f}")

    print("\nParâmetros Calibrados:")
    print(f"  mu: {parametros_calibrados['mu_estimado']:.4f}")
    print(f"  theta: {parametros_calibrados['theta_estimado']:.4f}")
    print(f"  sigma: {parametros_calibrados['sigma_estimado']:.4f}")
    print(f"  lambda: {parametros_calibrados['lambda_estimado']:.4f}")
    # A calibração foi feita com log-retornos, que são uma boa aproximação de retornos simples para valores pequenos
    print(f"  mu_J (log-retorno): {parametros_calibrados['mu_J_estimado']:.4f}") 
    print(f"  sigma_J (log-retorno): {parametros_calibrados['sigma_J_estimado']:.4f}")
