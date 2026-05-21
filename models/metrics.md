# Metricas do Modelo

Modelo final embarcado: MLP compacta
Acuracia da MLP no conjunto de teste: 0.8548
Parametros da MLP compacta: 598

Modelo anterior de comparacao: arvore de decisao compacta
Acuracia da arvore compacta: 0.7842
Nos da arvore compacta: 61
Nos da arvore completa de comparacao: 593

A compactacao final foi feita escolhendo uma MLP pequena, com apenas uma camada oculta de 16 neuronios.
Isso melhora a metrica em relacao a arvore compacta e ainda permite embarcar o modelo como arrays C/C++.

## Relatorio por classe

```text
precision    recall  f1-score   support

           WALKING       0.73      0.84      0.78       496
  WALKING_UPSTAIRS       0.82      0.74      0.78       471
WALKING_DOWNSTAIRS       0.88      0.83      0.86       420
           SITTING       0.89      0.78      0.83       491
          STANDING       0.82      0.90      0.86       532
            LAYING       1.00      1.00      1.00       537

          accuracy                           0.85      2947
         macro avg       0.86      0.85      0.85      2947
      weighted avg       0.86      0.85      0.85      2947
```
