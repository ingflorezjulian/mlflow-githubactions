"""
Script de validación del modelo
Carga el modelo desde MLflow (NO desde joblib)
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# Configuración
EXPERIMENT_NAME = "Heart-Disease-Classification"
THRESHOLD_ACCURACY = 0.70  # Umbral mínimo
mlruns_dir = os.path.join(os.getcwd(), "mlruns")
tracking_uri = f"file://{os.path.abspath(mlruns_dir)}"

mlflow.set_tracking_uri(tracking_uri)


def load_validation_data():
    """Carga datos para validación"""
    print("📂 Cargando datos de validación...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        df = df.dropna()
        df['target'] = (df['target'] > 0).astype(int)
        print(f"✅ Datos cargados: {df.shape[0]} muestras")
        return df
    except Exception as e:
        print(f"⚠️  Error: {e}. Usando datos sintéticos...")
        return create_synthetic_data()


def create_synthetic_data():
    """Datos sintéticos de respaldo"""
    np.random.seed(123)
    n_samples = 100
    
    data = {
        'age': np.random.randint(29, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)


def get_latest_model():
    """
    Obtiene el modelo más reciente desde MLflow
    NO usa joblib (requisito del taller)
    """
    print(f"\n🔍 Buscando modelo en experimento '{EXPERIMENT_NAME}'...")
    
    try:
        # Obtener experimento
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if not experiment:
            raise ValueError(f" Experimento '{EXPERIMENT_NAME}' no encontrado. Ejecuta 'make train' primero.")
        
        # Buscar último run
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise ValueError(" No hay runs en el experimento. Ejecuta 'make train' primero.")
        
        latest_run_id = runs.iloc[0]['run_id']
        print(f"✅ Run encontrado: {latest_run_id}")
        
        # Cargar modelo desde MLflow (NO desde .pkl)
        model_uri = f"runs:/{latest_run_id}/model"
        print(f"📦 Cargando modelo desde MLflow...")
        
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Modelo cargado exitosamente\n")
        
        return model, latest_run_id
        
    except Exception as e:
        print(f" Error: {e}")
        raise


def validate_model():
    """Valida el desempeño del modelo"""
    print("=" * 70)
    print("🧪 VALIDACIÓN DEL MODELO")
    print("=" * 70)
    
    try:
        # 1. Cargar modelo desde MLflow
        model, run_id = get_latest_model()
        
        # 2. Cargar datos de validación
        df = load_validation_data()
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Usar split diferente para validación
        _, X_val, _, y_val = train_test_split(
            X, y, test_size=0.3, random_state=999, stratify=y
        )
        
        print(f"📊 Datos de validación: {X_val.shape[0]} muestras\n")
        
        # 3. Predicciones
        print(f"🔮 Realizando predicciones...")
        y_pred = model.predict(X_val)
        
        # 4. Métricas
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_val, y_pred)
        
        # 5. Mostrar resultados
        print("\n" + "=" * 70)
        print("📊 RESULTADOS DE VALIDACIÓN")
        print("=" * 70)
        print(f"Run ID: {run_id}\n")
        print(f"Métricas:")
        print(f"  • Accuracy:  {accuracy:.4f}")
        print(f"  • Precision: {precision:.4f}")
        print(f"  • Recall:    {recall:.4f}")
        print(f"  • F1-Score:  {f1:.4f}\n")
        
        print(f"Matriz de Confusión:")
        print(cm)
        print(f"\nReporte Detallado:")
        print(classification_report(y_val, y_pred, 
                                   target_names=['Sin Enfermedad', 'Con Enfermedad'],
                                   zero_division=0))
        
        # 6. Validar contra umbral
        print("=" * 70)
        print("✅ VALIDACIÓN DE UMBRAL")
        print("=" * 70)
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Umbral mínimo:   {THRESHOLD_ACCURACY:.4f}")
        
        if accuracy >= THRESHOLD_ACCURACY:
            diferencia = accuracy - THRESHOLD_ACCURACY
            print(f"\n ¡ÉXITO! El modelo supera el umbral")
            print(f"   Margen: +{diferencia:.4f}")
            print("=" * 70)
            return 0
        else:
            diferencia = THRESHOLD_ACCURACY - accuracy
            print(f"\n FALLO: El modelo NO alcanza el umbral")
            print(f"   Falta: {diferencia:.4f}")
            print("=" * 70)
            return 1
        
    except Exception as e:
        print(f"\n Error durante validación: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = validate_model()
    exit(exit_code)