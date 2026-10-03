# Especificação: triagem cardiovascular com rede neural

## Objetivo e escopo
Apoiar a **priorização** de pacientes para investigação cardiovascular a partir de 11 dados
clínicos simples, com uma rede neural treinada na base curada Coração de Dados. Não faz
diagnóstico, não prescreve e não substitui a avaliação clínica.

## Critérios de aceitação

### Dados e modelo
- **DADOS-01**: a base é dividida 60/20/20 de forma estratificada; o pré-processador é
  ajustado somente no treino. (`tests/test_dados.py`)
- **MOD-01**: as redes (NumPy, Keras, PyTorch) aprendem um problema não linear (XOR) e o
  gradiente da rede em NumPy confere com o numérico. (`tests/test_modelos.py`)
- **MOD-02**: o limiar de triagem é o **maior** que garante sensibilidade ≥ 80 % na
  validação; o teste nunca é usado para escolhê-lo. (`tests/test_modelos.py`)
- **MOD-03**: comparações de AUC entre modelos usam bootstrap **pareado**.
  (`tests/test_modelos.py`)

### Validação de entrada (Pydantic)
- **VAL-01**: o esquema `Paciente` rejeita valores fisiologicamente impossíveis, pressão
  diastólica ≥ sistólica e campos extras. Nenhuma previsão é feita com dado inválido.
- **VAL-02**: o esquema normaliza formatos humanos: "1,72 m" → 172 cm, "15 por 9,5" →
  150/95 mmHg, "homem" → masculino, "acima do normal" → 2, "normais" → 1.
- **VAL-03**: valores plausíveis mas fora da faixa do treino (ex.: 78 anos) geram **aviso**
  de extrapolação, não erro.

### Fluxo LangGraph
- **FLX-01**: o fluxo segue interpretar → validar → (perguntar | corrigir | avaliar →
  explicar); perguntas sobre o modelo vão para responder_pergunta. Cada nó registra uma
  linha no rastreio.
- **FLX-02**: se a mensagem trouxe dados de paciente, a intenção é triagem — regra em
  código, não decisão do LLM.
- **FLX-03**: a pressão é convertida em código a partir do trecho escrito, nunca pelo LLM.
- **FLX-04**: dados se acumulam entre mensagens; "e se…" altera só o valor citado e
  recalcula.
- **FLX-05**: faltando dados, o fluxo lista o que falta e **não** chama a rede.

### Agente e LLM
- **AGT-01**: todo número de risco vem da rede neural, por ferramenta ou nó do fluxo;
  o LLM só extrai dados e redige.
- **AGT-02**: a explicação cita apenas os fatores devolvidos pela rede, com o sentido
  devolvido. A nota sobre fumo/álcool só aparece quando eles constam reduzindo o risco.
- **AGT-03**: sem chave de API, o assistente funciona em **modo demonstração** (regras,
  sem IA, sem custo), identificado na resposta. Com chave, usa OpenAI ou DeepSeek.
  As regras não confundem: "ativo" sem contexto de exercício, idade de parente ou duração
  ("fumou por 10 anos"), hábito de parente ("o marido fuma"), data ("12/08") e negação
  antes ou depois do nome ("não está com colesterol alto", "colesterol não está alto",
  "nega tabagismo"). Um valor que o esquema recusa é descartado, nunca derruba a extração.
- **AGT-04**: a saída estruturada do LLM é validada pelo Pydantic; se for rejeitada, o erro
  volta ao LLM para correção (até 2 tentativas). A ferramenta `avaliar_paciente` do agente
  ReAct tem o próprio `Paciente` como esquema.

### Aplicação
- **APP-01**: o app abre sem erro, calcula o risco do formulário e mostra os 9 gráficos.
  (`tests/test_app.py`)
- **APP-02**: o controle do limiar atualiza as métricas no teste.
- **APP-03**: o assistente mostra o rastreio do fluxo; falha do provedor gera mensagem
  amigável, sem fingir resposta; mensagens limitadas a 1000 caracteres.
- **APP-04**: números no padrão brasileiro (vírgula decimal) no app, nos gráficos e nas
  respostas do assistente; a conversão é feita no número formatado, nunca na frase pronta.

## Desenvolvimento orientado a especificações
1. Alterar ou acrescentar um critério aqui antes de mudar o comportamento.
2. Escrever o cenário em `tests/`, citando o ID na docstring.
3. Implementar a menor mudança necessária.
4. Rodar `uv run invoke test` — o GitHub Actions roda o mesmo a cada push.

## Fora do escopo
Diagnóstico, prescrição, dados reais de pacientes, ECG ou exames de imagem, login e
armazenamento de conversas. Use apenas dados fictícios nas demonstrações.
