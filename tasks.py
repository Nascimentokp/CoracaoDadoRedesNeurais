"""Tarefas do projeto executadas com o invoke.

Liste as tarefas disponíveis com `uv run invoke --list`.
"""

from invoke import task

NOTEBOOKS = [
    "01-mlp-do-zero-numpy",
    "02-mlp-keras",
    "03-mlp-pytorch",
    "04-otimizadores-arquiteturas",
    "05-comparacao-modelo-final",
    "06-explicabilidade",
    "07-transformer-tabular",
    "08-autoencoder",
    "09-agente",
]


@task
def lab(c):
    """Abre o JupyterLab."""
    c.run("jupyter lab")


@task
def treinar(c):
    """Treina a rede final e grava modelo, pré-processador e metadados em models/."""
    from src.model.treino import treinar_modelo_final

    metadados = treinar_modelo_final(verbose=1)
    teste = metadados["teste_limiar_escolhido"]
    print(
        f"AUC teste {teste['auc_roc']:.4f} · limiar {metadados['limiar']:.3f} · "
        f"sensibilidade {teste['sensibilidade']:.1%} · especificidade {teste['especificidade']:.1%}"
    )


@task
def notebooks(c):
    """Executa todos os notebooks em ordem, regravando figuras e tabelas em reports/."""
    for nome in NOTEBOOKS:
        c.run(
            f"jupyter nbconvert --to notebook --execute --inplace "
            f"--ExecutePreprocessor.timeout=1800 notebooks/{nome}.ipynb"
        )


@task
def app(c):
    """Executa a aplicação Streamlit definida em src/deployment/app.py."""
    c.run("streamlit run src/deployment/app.py")


@task
def lint(c):
    """Verifica o código com o ruff (lint e formatação)."""
    c.run("ruff check src tests tasks.py")
    c.run("ruff format --check src tests tasks.py")


@task
def fmt(c):
    """Formata o código e aplica correções automáticas do ruff."""
    c.run("ruff format src tests tasks.py")
    c.run("ruff check --fix src tests tasks.py")


@task
def test(c):
    """Executa a suíte de testes com o pytest."""
    c.run("pytest")


@task
def agente(c):
    """Conversa no terminal com o agente de triagem (requer OPENAI_API_KEY no .env)."""
    c.run("python -m src.agente.agente", pty=True)
