# Predicción de abandono de clientes bancarios (Customer Churn)

Estudio comparativo de modelos de *machine learning* para predecir qué clientes de un banco van a abandonarlo. Parte de un modelo base y va mejorándolo paso a paso (ingeniería de variables, tratamiento del desbalanceo, ajuste de hiperparámetros y ensamblado), comparando cada variante con las mismas métricas.

Trabajo de Fin de Máster del **Máster de FP en Inteligencia Artificial y Big Data**.

📄 [Memoria completa del TFM](docs/TFM_Neco_Ruiz_Crisostomo.pdf)

## Objetivo

Comparar cómo se comportan distintos modelos de clasificación ante un conjunto de datos desbalanceado (solo ~20 % de los clientes abandona) y determinar qué combinación de preprocesado y modelo identifica mejor a los clientes que se van.

## Dataset

- **Fichero:** `data/Customer-Churn-Records.csv`
- **Tamaño:** 10.000 clientes y 18 columnas, sin valores nulos ni duplicados.
- **Variable objetivo:** `Exited` (1 = el cliente abandona el banco). El 20,4 % de los clientes (2.038) abandona.
- **Variables:** puntuación de crédito, país (Francia, Alemania, España), género, edad, antigüedad, saldo, número de productos, tarjeta de crédito, cliente activo, salario estimado, puntuación de satisfacción, tipo de tarjeta y puntos acumulados.

## Metodología

1. **Análisis exploratorio:** estructura, nulos, duplicados, distribución de la variable objetivo y matriz de correlación.
2. **Limpieza:** se eliminan los identificadores (`RowNumber`, `CustomerId`, `Surname`) y `Complain`, que aporta prácticamente la misma información que `Exited` y haría que el modelo "hiciera trampa".
3. **División de datos:** 80 % entrenamiento y 20 % test, estratificada por la variable objetivo.
4. **Pipeline A (básico):** imputación (media / valor más frecuente) y codificación One-Hot de las variables categóricas.
5. **Pipeline B (mejorado):** añade variables nuevas (`Age_Balance`, `Credit_Products`, `Tenure_Salary`), transformación logarítmica de saldo y salario para suavizar valores extremos, escalado con `StandardScaler` y `class_weight="balanced"` para compensar el desbalanceo.
6. **Modelos:** Regresión Logística, Random Forest, XGBoost y un `VotingClassifier` (votación suave) que combina los tres.
7. **Ajuste de hiperparámetros:** `RandomizedSearchCV` sobre XGBoost (10 combinaciones, validación cruzada de 3 pliegues, optimizando F1).
8. **Evaluación:** *classification report*, matriz de confusión, curva ROC, importancia de variables y curvas de aprendizaje.

Todo el preprocesado va dentro de `Pipeline` y `ColumnTransformer` de Scikit-learn, de modo que las transformaciones se ajustan solo con los datos de entrenamiento.

## Resultados

Métricas sobre el conjunto de test (2.000 clientes), tomando la clase "abandona" como positiva:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|:--------:|:---------:|:------:|:--:|:-------:|
| Regresión Logística (A) | 0,79 | 0,41 | 0,03 | 0,06 | 0,67 |
| Random Forest (A) | 0,87 | 0,81 | 0,47 | 0,60 | 0,86 |
| Regresión Logística (B) | 0,71 | 0,39 | 0,72 | 0,51 | 0,78 |
| Random Forest (B) | 0,86 | 0,79 | 0,44 | 0,56 | 0,86 |
| XGBoost (B) | 0,85 | 0,68 | 0,51 | 0,58 | 0,85 |
| XGBoost tuneado (B) | 0,86 | 0,74 | 0,51 | 0,60 | 0,85 |
| VotingClassifier (B) | 0,86 | 0,70 | 0,54 | 0,61 | 0,86 |
| **VotingClassifier tuneado (B)** | 0,86 | 0,71 | **0,56** | **0,62** | 0,86 |

### Conclusiones

- **Accuracy engaña en este problema.** La Regresión Logística básica acierta el 79 % de las veces y aun así solo detecta al 3 % de los clientes que se van, porque casi siempre predice "se queda".
- **Los modelos basados en árboles superan claramente a la regresión logística** (ROC-AUC 0,85-0,86 frente a 0,67-0,78).
- **El balanceo de clases cambia el equilibrio entre precisión y recall.** La Regresión Logística con `class_weight="balanced"` pasa de detectar el 3 % al 72 % de los abandonos, a costa de más falsas alarmas (precisión 0,39).
- **El mejor resultado global (F1 = 0,62) lo da el ensamblado `VotingClassifier` con XGBoost tuneado**, que también logra el mayor recall (0,56) entre los modelos con buena precisión.

### Limitaciones y trabajo futuro

- Los modelos se comparan sobre una única división train/test, y las diferencias entre los mejores (F1 de 0,60 a 0,62) son pequeñas, por lo que no son concluyentes. Una validación cruzada repetida daría una comparación más sólida.
- Se usa el umbral de decisión por defecto (0,5). Ajustarlo según el coste real de perder un cliente frente al de contactarlo en vano mejoraría el recall.
- El ajuste de hiperparámetros se limita a 10 combinaciones sobre XGBoost.

## Estructura del repositorio

```
.
├── notebooks/
│   └── tfm.ipynb                    # Análisis completo, de principio a fin
├── data/
│   └── Customer-Churn-Records.csv   # Dataset
├── docs/
│   └── memoria_tfm.pdf              # Memoria del TFM
├── requirements.txt
└── README.md
```

## Tecnologías

Python 3.12 · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Jupyter

## Cómo ejecutarlo

1. Clona el repositorio:
   ```bash
   git clone https://github.com/NecoRuiz/NOMBRE-DE-TU-REPOSITORIO.git
   cd NOMBRE-DE-TU-REPOSITORIO
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Abre el notebook desde la carpeta `notebooks/` (la ruta del dataset es relativa) y ejecuta las celdas en orden:
   ```bash
   cd notebooks
   jupyter notebook tfm.ipynb
   ```

## Autor

**Neco Ruiz Crisóstomo**
GitHub: [NecoRuiz](https://github.com/NecoRuiz)
