# Retropropagação (Backpropagation)
A retropropagação (ou backpropagation, em inglês) é um algoritmo fundamental no treinamento de redes neurais artificiais. Ela permite que a rede aprenda ajustando seus pesos internos de forma eficiente, minimizando o erro entre as previsões da rede e os valores reais esperados.

## Como funciona?
O processo de retropropagação ocorre em duas etapas principais:

### 1. Propagação direta (Forward Pass):

- Os dados de entrada passam pela rede neural, camada por camada, até gerar uma saída.
- A saída é comparada com o valor esperado, e calcula-se o erro (função de custo).


### 2. Propagação reversa (Backward Pass):

- O erro calculado é propagado de volta pela rede, do final para o início.
- Durante esse processo, o algoritmo calcula o gradiente do erro em relação a cada peso da rede.
- Os pesos são então ajustados (usando métodos como o gradiente descendente) para reduzir o erro.

## Por que é importante?

A retropropagação tornou possível o treinamento eficiente de redes neurais profundas, permitindo avanços significativos em áreas como reconhecimento de imagem, processamento de linguagem natural e jogos.

### Exemplo simples
Imagine que você tem uma rede neural tentando prever o preço de uma casa. Se a previsão estiver errada, a retropropagação ajuda a ajustar os parâmetros internos da rede para que, na próxima vez, a previsão seja mais precisa.

## Referências
Wikipedia: Backpropagation
Deep Learning Book - Ian Goodfellow

Este repositório contém exemplos e implementações do algoritmo de retropropagação para fins de estudo e experimentação.