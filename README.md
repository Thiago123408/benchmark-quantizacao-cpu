# Benchmark de Quantização em CPU (ResNet-18) — Estudo de Caso para o Python Norte

Este repositório contém o estudo de caso empírico e os códigos utilizados para analisar o impacto da **Quantização Dinâmica** aplicada isoladamente na camada inicial (`conv1`) de uma rede neural ResNet-18, utilizando o ecossistema PyTorch.

O objetivo desta análise é demonstrar os desafios de otimização de modelos de Deep Learning em arquiteturas de CPU de uso geral e validar empiricamente o conceito de *overhead* de conversão de tipos de dados (*on-the-fly*).

---

## 📊 Resultados do Benchmark (Latência Média)

Os testes foram executados em ambiente de CPU de alto desempenho (Google Colab), calculando a média de tempo de resposta após estabilização do hardware (*warm-up* de 10 iterações e benchmark sobre 100 inferências):

| Configuração do Modelo | Camada Modificada | Tipo de Dado | Latência Média (ms) | Impacto em Performance |
| :--- | :---: | :---: | :---: | :---: |
| **Modelo Original** | Nenhuma | Float32 | **110.68 ms** | Referência (0.0%) |
| **Modelo Modificado** | `conv1` | Int8 (Dynamic) | **109.40 ms** | **+1.2% mais rápido** |

---

## 🧠 Análise Técnica: O Fenômeno do *Overhead*

O ganho sutil de apenas **1.2%** obtido no teste prático traz uma lição valiosa de engenharia de software para IA, que quebra o mito de que *"qualquer quantização gera velocidade imediata"*:

1. **Gargalo de Conversão:** Como apenas a primeiríssima camada (`conv1`) foi convertida para `int8`, a CPU precisa receber a imagem em `float32`, convertê-la para `int8` antes da operação, processar a convolução de forma leve e, logo em seguida, reverter o resultado para `float32` para que o restante da ResNet-18 consiga continuar o fluxo de processamento.
2. **Custo Computacional:** Esse processo constante de "leva-e-traz" e transformação de formatos de dados em tempo de execução consome ciclos valiosos do processador. O ganho de velocidade pura na execução da camada isolada acabou sendo quase que inteiramente engolido pelo custo de conversão.

### 💡 Conclusão para Produção
Este experimento prova empiricamente que para extrair o máximo desempenho de hardware de uso geral (CPUs), a abordagem isolada ou puramente dinâmica em poucas camadas é insuficiente. Para eliminar o gargalo do *overhead*, a estratégia recomendada para ambientes de borda (*edge*) é a **Quantização Estática Total (PTQ)**, onde o grafo completo do modelo é mapeado e convertido previamente, extinguindo as transformações de tipo em tempo de execução.

---

## 🚀 Como Executar o Experimento

O código está estruturado em um Notebook Jupyter pronto para rodar no Google Colab.

1. Abra o arquivo [`benchmark_quantizacao.ipynb`](./benchmark_quantizacao.ipynb) aqui no GitHub.
2. Clique no botão **"Open in Colab"** no topo do arquivo.
3. Execute a célula para reproduzir as métricas de latência em tempo real.
