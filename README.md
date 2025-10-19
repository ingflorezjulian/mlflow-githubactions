# MLflow CI/CD Pipeline - Taller MLFlow Github Actions

**Autor:** Julian David Florez Sanchez - EAN 2025-2

## Descripción del Proyecto

Este proyecto implementa un pipeline completo de MLOps que:

1. Entrena un modelo de clasificación (Random Forest)
2. Registra experimentos y artefactos con MLflow
3. Valida el desempeño del modelo
4. Automatiza el proceso con GitHub Actions
5. Usa un dataset externo (NO sklearn.datasets)

## Estructura del Proyecto
```
mlflow-githubactions/
├── train.py              # Script de entrenamiento
├── validate.py           # Script de validación
├── requirements.txt      # Dependencias Python
├── Makefile             # Comandos de automatización
├── mlruns/              # Almacenamiento MLflow (generado)
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml  # Pipeline CI/CD
└── README.md            # Este archivo
```

## Dataset Utilizado

### Heart Disease UCI Dataset

- **Fuente:** UCI Machine Learning Repository
- **URL:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Descripción:** Dataset médico para predecir enfermedad cardíaca
- **Features:** 14 atributos (edad, sexo, presión arterial, colesterol, etc.)
- **Target:** Presencia/ausencia de enfermedad cardíaca (clasificación binaria)
- **Tamaño:** ~297 muestras después de limpieza

### Justificación

Se eligió este dataset porque:

1. Es un problema real de clasificación médica
2. No proviene de sklearn.datasets (requisito del taller)
3. Tiene suficientes muestras para entrenamiento y validación
4. Es público y reproducible
