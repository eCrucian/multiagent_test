# Ferramentas necessárias (conceituais)
from sentence_transformers import SentenceTransformer
from minha_vector_db import VectorDB # Ex: pinecone, chromadb
from minha_llm_api import LLM # Ex: google.generativeai, openai

# --- FASE 1: SETUP (FEITO UMA VEZ) ---

# 1. Carregar base de dados com pares (texto, dados_numericos)
base_de_conhecimento = carregar_base_historica() # [{'texto': '...', 'dados': [...]}, ...]

# 2. Inicializar modelo de embedding e DB vetorial
embedding_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
vector_db = VectorDB()

# 3. Indexar os dados
textos_historicos = [item['texto'] for item in base_de_conhecimento]
embeddings_historicos = embedding_model.encode(textos_historicos)
vector_db.index(embeddings=embeddings_historicos, metadata=base_de_conhecimento)


# --- FASE 2: GERAÇÃO DE CENÁRIO ---

def gerar_cenario_com_llm(descricao_cenario: str, k: int = 3):
    """Gera um cenário de risco usando a arquitetura RAG."""
    
    # 1. Criar embedding para a descrição do usuário
    embedding_consulta = embedding_model.encode([descricao_cenario])
    
    # 2. Buscar cenários históricos similares no DB vetorial
    resultados_similares = vector_db.search(query_embedding=embedding_consulta, top_k=k)
    
    # 3. Construir o prompt aumentado
    prompt = f"""
    Você é um especialista em risco de mercado quantitativo. Sua tarefa é gerar um conjunto de parâmetros de fatores de risco (curvas, superfícies de volatilidade, preços) que seja consistente com um cenário descrito pelo usuário.

    **Cenário do Usuário:**
    "{descricao_cenario}"

    ---

    Para te ajudar, encontrei {len(resultados_similares)} cenários históricos que são semanticamente similares. Use-os como referência principal para garantir que os números gerados sejam plausíveis e consistentes.

    **Exemplos Históricos Relevantes:**

    """
    
    for i, res in enumerate(resultados_similares):
        prompt += f"**Exemplo {i+1}:**\n"
        prompt += f"**Descrição Histórica:** {res['texto']}\n"
        prompt += f"**Dados Observados:** {res['dados_numericos']}\n\n"
        
    prompt += """
    ---
    
    **Sua Tarefa:**
    Com base no cenário do usuário e nos exemplos históricos, gere um novo conjunto de dados numéricos. O resultado deve refletir a severidade descrita pelo usuário, mas ser ancorado na realidade dos dados históricos.
    Os parâmetros são: 8 para curva, 8 para superfície de vol, e 2 para preços de ativos.
    
    Retorne a resposta EXCLUSIVAMENTE no formato JSON, como no exemplo abaixo:
    {
      "parametros_curva": [p1, p2, p3, p4, p5, p6, p7, p8],
      "parametros_vol": [v1, v2, v3, v4, v5, v6, v7, v8],
      "precos": [preco_a, preco_b]
    }
    """
    
    # 4. Chamar a LLM e obter a resposta
    llm = LLM()
    resposta_llm = llm.generate(prompt)
    
    # 5. Parsear a resposta JSON para obter os dados
    cenario_gerado = json.loads(resposta_llm)
    
    return cenario_gerado

# --- Exemplo de uso ---
meu_cenario_texto = "Cenário de estresse extremo com crise de crédito global similar à de 2008, mas com o complicador de uma inflação global persistentemente alta. O Banco Central brasileiro é forçado a sinalizar uma alta de juros agressiva, mesmo com a atividade econômica em queda."

cenario_final = gerar_cenario_com_llm(meu_cenario_texto)
print(cenario_final)
