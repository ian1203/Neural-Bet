# ⚽ Corner Prediction with LSTM — AI Sports Analytics

Este proyecto utiliza redes neuronales LSTM para predecir el número total de córners en partidos de fútbol, con énfasis en la Premier League. Su objetivo es servir como herramienta de análisis para decisiones deportivas, estadísticas tácticas y mercados de apuestas over/under.

---

## 📌 Características principales

- Predicción por separado de córners local y visitante
- Más de 30 variables predictoras: forma, odds, fuerza ofensiva, historial, derbis
- Red LSTM con arquitectura personalizada (last step + mean pooling)
- Función de pérdida con penalización cuadrática en partidos UNDER
- Exportación de predicciones y análisis de métricas como MAE, RMSE y P(>9.5)
- Preparado para integración futura vía API y frontend web

---

## 🗂️ Estructura del repositorio

```plaintext
.
├── data_loader.py              # Carga múltiples temporadas en un solo DataFrame
├── feature_engineering.py      # Genera estadísticas de forma, odds, h2h, etc.
├── match_dataset.py            # Dataset personalizado para secuencias LSTM
├── model.py                    # Arquitectura CornerLSTM
├── prepare_dataset.py          # Pipeline de limpieza y exportación del dataset
├── train.py                    # Entrenamiento del modelo, validación y test
├── model/
│   └── corner_lstm.pth         # Modelo entrenado
├── data/
│   └── processed/              # Dataset listo para modelar
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación general

```

## 📦 Requisitos

Instala las dependencias con:

``` plaintext
pip install -r requirements.txt
```

Dependencias principales:

- Python 3.10+

- PyTorch

- Pandas / NumPy / scikit-learn

- Matplotlib / Seaborn (opcional para visualizaciones)

## 🔄 Preparar el dataset

Ejecuta el script para cargar y procesar los datos desde múltiples temporadas:

```bash
python prepare_dataset.py
```

Esto generará un archivo 2000-2022_with_form.csv en la carpeta /processed con todas las características calculadas mediante ingeniería de datos avanzada (forma, odds, enfrentamientos directos, etc.).

## 🧠 Entrenar el modelo

Ejecuta el script de entrenamiento:

```bash
python train.py
```

Este script entrenará la red LSTM con los siguientes parámetros:

✅ 2 capas LSTM  
✅ 128 unidades ocultas (`hidden_size`)  
✅ Dropout de 0.2  
✅ Batch size de 24  
✅ 40 épocas (con `early stopping` si no mejora en 10 épocas)

Durante el entrenamiento se evalúa el modelo con las siguientes métricas:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **P(over 9.5):** probabilidad estimada de que el partido tenga más de 9.5 córners
- **Hit Rate:** precisión en clasificar correctamente OVER/UNDER

El modelo con mejor validación se guarda automáticamente como:

```plaintext
/model/corner_lstm.pth
```

---

## 📊 Resultados y Métricas

Después del entrenamiento y evaluación final, se obtienen resultados aproximados como:

- 📉 **MAE:** ~2.3  
- 📉 **RMSE:** ~2.8  
- 📈 **P(over 9.5):** ~0.90  
- ✅ **Precisión clasificada:** Buena separación entre partidos UNDER y OVER

También se genera un archivo `.csv` con las predicciones del conjunto de prueba, que incluye:
- Córners predichos (local y visitante)
- Etiquetas reales
- Error absoluto
- Clasificación OVER/UNDER


