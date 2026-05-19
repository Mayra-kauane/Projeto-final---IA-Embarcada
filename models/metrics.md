# Metricas do Modelo

Acuracia no conjunto de teste: 0.7842
Nos da arvore compacta: 61
Nos da arvore completa de comparacao: 593

A compressao usada aqui foi limitar a profundidade da arvore.
Isso reduz a quantidade de regras que precisa ir para o ESP32-S3.

## Relatorio por classe

```text
precision    recall  f1-score   support

           WALKING       0.73      0.62      0.67       496
  WALKING_UPSTAIRS       0.56      0.80      0.66       471
WALKING_DOWNSTAIRS       0.83      0.57      0.68       420
           SITTING       0.89      0.73      0.80       491
          STANDING       0.79      0.92      0.85       532
            LAYING       1.00      1.00      1.00       537

          accuracy                           0.78      2947
         macro avg       0.80      0.77      0.78      2947
      weighted avg       0.80      0.78      0.78      2947
```
