# Clasificación de texto

## MERCADOLIBRE DATA CHALLENGE 2019

Un modelo de clasificación de texto en Python que hice *from scratch* para la competencia de Mercado Libre.

La idea principal es contar k-uplas de palabras (k=1,2,3) pensando en optimizar la métrica balanced accuracy.

Ambos notebooks (`test.ipynb` y `val.ipynb`) se pueden correr en una instancia `r5.12xlarge` en aproximadamente 2hs.

Utilicé esta instancia porque el modelo requiere bastante memoria (257GB al entrenar en todo el dataset) pero no está paralelizado (utiliza un solo thread).

### Puntaje de validación (`val.ipynb`): 0.8844

### Puntaje público (`test.ipynb`): 0.89207 (puesto  22)

### Puntaje privado  (`test.ipynb`): ????? (puesto ?)

