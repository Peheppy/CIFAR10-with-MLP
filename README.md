# Experimentos com Redes MLP no CIFAR-10

Este repositório apresenta experimentos conduzidos com redes **Multilayer Perceptron (MLP)** aplicadas ao conjunto de dados **CIFAR-10**, utilizando a biblioteca **PyTorch**. O objetivo principal é comparar diferentes arquiteturas e estratégias de otimização em um problema supervisionado de classificação de imagens.

## Estrutura do Repositório

├── results/
│ ├── model_comparison.csv # Tabela comparativa dos modelos avaliados
│ ├── baseline_model.pth # Pesos do modelo "Baseline"
│ ├── baseline_history.json # Histórico de treinamento do "Baseline"
│ ├── smallnet_model.pth # Pesos do modelo "SmallNet"
│ ├── smallnet_history.json # Histórico de treinamento do "SmallNet"
│ ├── deepnet_model.pth # Pesos do modelo "DeepNet"
│ ├── deepnet_history.json # Histórico de treinamento do "DeepNet"
├── mlp_cifar10_experiments.ipynb # Notebook principal com os experimentos


## Modelos Avaliados

Três configurações de MLP foram definidas, variando em profundidade, funções de ativação, otimizadores e taxa de dropout. Abaixo, um resumo das principais características:

| Modelo     | Arquitetura           | Otimizador | Dropout | LR Inicial |
|------------|------------------------|------------|---------|------------|
| Baseline   | [256, 512, 256]        | AdamW      | 0.3     | 4e-3       |
| SmallNet   | [128, 256, 128]        | Adam       | 0.2     | 1e-3       |
| DeepNet    | [256, 512, 512, 256]   | RMSprop    | 0.4     | 3e-4       |

Os modelos foram treinados utilizando **early stopping** e **scheduler** (`ReduceLROnPlateau`) para controle dinâmico da taxa de aprendizado, além de validação estratificada com divisão 85/15 do conjunto de treinamento.

## Resultados

Cada experimento gera os seguintes artefatos:
- Curvas de **acurácia e função de perda** para treino e validação
- Evolução do **learning rate** ao longo das épocas
- Métricas de desempenho no **conjunto de teste**
- Registro de resultados consolidados no formato `.csv` e `.json`

## Requisitos

O projeto foi desenvolvido para ser executado no ambiente **Google Colab**, com os seguintes pacotes:

- `torch`, `torchvision`
- `scikit-learn`
- `matplotlib`, `seaborn`, `pandas`, `numpy`
- `tqdm`, `torchinfo`

## Execução

1. Abra o notebook `mlp_cifar10_experiments.ipynb` no Google Colab.
2. Execute todas as células sequencialmente.
3. Os resultados serão salvos automaticamente na pasta `results/`.

## Contato

Para dúvidas, sugestões ou colaborações, entre em contato via [GitHub](https://github.com/peheppy)
