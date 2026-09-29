"""Configuração central do projeto.

Concentra caminhos, semente aleatória, definição das variáveis e paleta que
precisam ser idênticos entre notebooks, scripts de treino e aplicação
Streamlit. Nenhum outro módulo deve montar caminho na mão.
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parent.parent

DATA = RAIZ / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

MODELS = RAIZ / "models"

REPORTS = RAIZ / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

# --------------------------------------------------------------------------- #
# Fonte dos dados
# --------------------------------------------------------------------------- #

#: Base curada herdada do projeto CoracaoDadoTema6 (notebook 02-ajustes-dados):
#: 68.551 pacientes, sem duplicatas nem registros fisiologicamente implausíveis.
BASE_CURADA = DATA_PROCESSED / "cardio_curado.csv"

#: Alvo supervisionado — presença de doença cardiovascular (0 = não, 1 = sim).
#: No projeto de agrupamento ele ficava fora da modelagem; aqui é o que a rede
#: aprende a prever.
ALVO = "cardio"

#: Variáveis contínuas — entram padronizadas (média 0, desvio 1).
CONTINUAS = ["idade_anos", "height", "weight", "imc", "ap_hi", "ap_lo"]

#: Variáveis ordinais de 3 níveis — entram em one-hot (a rede não precisa
#: supor que o salto 1→2 vale o mesmo que 2→3).
ORDINAIS = ["cholesterol", "gluc"]

#: Variáveis binárias — entram como 0/1, sem transformação. `sexo_masculino`
#: é derivada de `gender` (1 = mulher, 2 = homem na base original).
BINARIAS = ["sexo_masculino", "smoke", "alco", "active"]

ENTRADAS = CONTINUAS + ORDINAIS + BINARIAS

#: Nomes legíveis para gráficos, explicações e aplicação.
ROTULOS = {
    "idade_anos": "Idade (anos)",
    "height": "Altura (cm)",
    "weight": "Peso (kg)",
    "imc": "IMC",
    "ap_hi": "Pressão sistólica",
    "ap_lo": "Pressão diastólica",
    "cholesterol": "Colesterol",
    "gluc": "Glicose",
    "sexo_masculino": "Sexo masculino",
    "smoke": "Fumante",
    "alco": "Consome álcool",
    "active": "Fisicamente ativo",
}

# --------------------------------------------------------------------------- #
# Reprodutibilidade e protocolo de avaliação
# --------------------------------------------------------------------------- #

SEMENTE = 42
DPI = 150

#: Divisão estratificada 60% treino / 20% validação / 20% teste — o mesmo
#: protocolo dos exercícios da disciplina. O teste só é tocado na avaliação final.
PROPORCAO_TESTE = 0.20
PROPORCAO_VALIDACAO = 0.25  # 25% dos 80% restantes = 20% do total

#: Sensibilidade mínima exigida ao escolher o limiar de decisão. Em triagem,
#: deixar passar um doente (falso negativo) custa mais que um alarme falso.
RECALL_MINIMO = 0.80

# --------------------------------------------------------------------------- #
# Identidade visual (a mesma do projeto CoracaoDadoTema6)
# --------------------------------------------------------------------------- #

AZUL_PROFUNDO = "#1B4965"
AZUL_COBALTO = "#2A6F97"
AZUL_CLARO = "#62B6CB"
VERDE = "#74C69D"
AMARELO = "#F4D35E"
CORAL = "#EE6C4D"
ROSA = "#F2A6A6"
LARANJA = "#F49D6E"
ROXO = "#8E7DBE"

#: Sequência categórica para gráficos com múltiplos grupos.
PALETA = [AZUL_COBALTO, CORAL, VERDE, AMARELO, ROXO, LARANJA, ROSA, AZUL_CLARO]


def garantir_diretorios() -> None:
    """Cria os diretórios de saída caso ainda não existam."""
    for diretorio in (DATA_RAW, DATA_PROCESSED, MODELS, FIGURES, TABLES):
        diretorio.mkdir(parents=True, exist_ok=True)
