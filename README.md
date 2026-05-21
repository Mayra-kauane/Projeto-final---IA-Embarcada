# Projeto Final - IA Embarcada e Modelos Compactos

Reconhecimento de atividade humana usando dados de acelerômetro e giroscópio, com treinamento em Python e inferência embarcada em um ESP32-S3 simulado no Wokwi.

## Objetivo

Classificar atividades humanas a partir de sinais inerciais:

- `WALKING`
- `WALKING_UPSTAIRS`
- `WALKING_DOWNSTAIRS`
- `SITTING`
- `STANDING`
- `LAYING`

O fluxo completo do projeto é:

```text
sensor -> janela com 128 leituras -> extração de características -> MLP compacta -> inferência no ESP32-S3
```

## Dataset

Foi usado o dataset público **UCI HAR - Human Activity Recognition Using Smartphones**, citado no enunciado do projeto.

O dataset contém leituras de acelerômetro e giroscópio coletadas por smartphone. Cada amostra é uma janela de 128 leituras.

Sinais usados:

```text
total_acc_x
total_acc_y
total_acc_z
body_gyro_x
body_gyro_y
body_gyro_z
```

Para cada sinal foram calculadas 5 características:

```text
média, desvio padrão, mínimo, máximo e RMS
```

Como são 6 sinais e 5 características por sinal, cada janela vira uma entrada com 30 características.

## Modelo

O modelo final embarcado foi uma **MLP compacta** (`MLPClassifier`), ou seja, uma pequena rede neural perceptron multicamadas.

Arquitetura usada:

```text
entrada: 30 características
camada oculta: 16 neurônios com ReLU
saída: 6 classes de atividade
```

Configuração principal em Python:

```python
make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=42,
        learning_rate_init=0.001,
    ),
)
```

Resultado no conjunto de teste:

```text
MLP compacta: 85,48% de acurácia
Quantidade de parâmetros: 598
```

Também foi treinada uma árvore de decisão compacta para comparação:

```text
Árvore compacta: 78,42% de acurácia
Árvore compacta: 61 nós
Árvore completa de comparação: 593 nós
```

Por isso, a MLP foi escolhida como modelo final: ela obteve melhor acurácia e continuou pequena o suficiente para embarcar no ESP32-S3.

Não foi feita quantização int8. A compactação foi feita pela escolha de uma arquitetura pequena, com poucos neurônios e poucos parâmetros.

## Embarque do modelo

O modelo treinado em Python foi convertido para arrays C/C++ no arquivo:

```text
firmware/wokwi/model_data.h
```

O arquivo exportado contém:

```text
SCALER_MEAN
SCALER_SCALE
MLP_W1
MLP_B1
MLP_W2
MLP_B2
```

No firmware, o ESP32-S3 executa a inferência manualmente:

```text
normalização das 30 características
-> camada oculta com ReLU
-> camada de saída
-> escolha da classe com maior pontuação
```

Assim, o dispositivo não precisa rodar Python nem `scikit-learn`.

## Simulação no Wokwi

Hardware simulado:

- ESP32-S3
- MPU6050

![Simulação no Wokwi](img/simulacao.png)

O MPU6050 representa o sensor inercial usado no dispositivo. Ele possui:

```text
acelerômetro x/y/z
giroscópio x/y/z
```

Comandos no terminal do Wokwi:

```text
n = sorteia uma janela real do dataset UCI HAR para comparação
l = coleta 128 leituras do MPU6050 e faz uma inferência live
r = mostra uma leitura instantânea do MPU6050
m = liga/desliga o monitor live do MPU6050
```

O monitor live do MPU6050 inicia ligado por padrão. A simulação fica lendo os sliders do sensor, formando uma janela de 128 leituras e executando a MLP compacta no ESP32-S3. O dataset não fica rodando sozinho; ele é usado apenas quando o comando `n` é enviado.

O comando `l` executa a pipeline embarcada completa:

```text
MPU6050 -> 128 leituras -> 30 características -> MLP compacta -> classe prevista
```

## Como executar

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Baixe o dataset:

```bash
python scripts/download_dataset.py
```

Treine e exporte o modelo:

```bash
python scripts/train_and_export.py
```

Compile o firmware:

```bash
python -m platformio run
```

Para simular no VS Code:

1. Instale as extensões PlatformIO IDE e Wokwi Simulator.
2. Abra a pasta do projeto.
3. Execute `PlatformIO: Build`.
4. Execute `Wokwi: Start Simulator`.
5. Abra o terminal `Wokwi Term...`.
6. Use os comandos `n`, `l`, `r` e `a`.

## Estrutura principal

```text
scripts/download_dataset.py       # baixa o dataset UCI HAR
scripts/train_and_export.py       # treina, compara e exporta o modelo
notebooks/                        # análise do dataset e treinamento
firmware/wokwi/sketch.ino         # firmware do ESP32-S3
firmware/wokwi/model_data.h       # MLP embarcada em C/C++
firmware/wokwi/demo_windows.h     # janelas reais do dataset para demonstração
platformio.ini                    # configuração PlatformIO
wokwi.toml                        # configuração Wokwi
diagram.json                      # circuito da simulação
```

## Observações

No Wokwi, o sensor simulado fica praticamente parado quando não há alteração manual dos valores. Por isso, o modo `l` tende a prever atividades estáticas, como `STANDING` ou `LAYING`.

Para demonstrar diferentes classes, o modo `n` usa janelas reais do dataset UCI HAR.
