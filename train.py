"""
Script de entrenamiento del modelo con MLflow
Dataset: Heart Disease UCI (NO sklearn.datasets)
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models import infer_signature

# Configuración
EXPERIMENT_NAME = "Heart-Disease-Classification"
mlruns_dir = os.path.join(os.getcwd(), "mlruns")
tracking_uri = f"file://{(mlruns_dir)}"

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(tracking_uri)


def load_data():
    """
    Carga dataset externo: Heart Disease UCI
    NO usa sklearn.datasets (requisito del taller)
    """
    print("📂 Cargando dataset externo (UCI Heart Disease)...")
    
    # URL del dataset de UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    try:
        df = pd.read_csv(url, names=column_names, na_values='?')
        df = df.dropna()
        df['target'] = (df['target'] > 0).astype(int)
        
        print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
        
    except Exception as e:
        print(f"Error descargando dataset: {e}")
        print("💡 Generando dataset sintético...")
        return create_synthetic_data()


def create_synthetic_data():
    """Dataset sintético como alternativa (sin conexión a internet)"""
    np.random.seed(42)
    n_samples = 300
    
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


def train_model():
    """Entrena el modelo y registra en MLflow"""
    print("=" * 70)
    print("ENTRENAMIENTO DEL MODELO")
    print("=" * 70)
    
    # Configurar experimento
    try:
        mlflow.create_experiment(
            name=EXPERIMENT_NAME,
            artifact_location=f"file://{(mlruns_dir)}"
        )
    except mlflow.exceptions.MlflowException:
        pass
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="RandomForest-Training") as run:
        
        # 1. Cargar datos
        df = load_data()
        X = df.drop('target', axis=1)
        y = df['target']
        
        # 2. Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n División de datos:")
        print(f"   Training: {X_train.shape[0]} muestras")
        print(f"   Testing:  {X_test.shape[0]} muestras")
        
        # 3. Hiperparámetros
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 4. Entrenar modelo
        print(f"\n🔧 Entrenando Random Forest...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # 5. Predicciones
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 6. Métricas
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'test_precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'test_recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'test_f1': f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
        }
        
        # 7. Registrar parámetros
        mlflow.log_params(params)
        
        # 8. Registrar métricas
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # 9. IMPORTANTE: Signature e Input Example (REQUISITO DEL TALLER)
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.iloc[:5]
        
        # 10. Registrar modelo con signature e input_example
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,         
            input_example=input_example  
        )
        
        # 11. Resultados
        print("\n" + "=" * 70)
        print("📈 RESULTADOS DEL ENTRENAMIENTO")
        print("=" * 70)
        print(f"Run ID: {run.info.run_id}")
        print(f"\nMétricas:")
        for metric_name, metric_value in metrics.items():
            print(f"  • {metric_name}: {metric_value:.4f}")
        print(f"\nModelo guardado en: mlruns/")
        print("=" * 70)
        
        return run.info.run_id


if __name__ == "__main__":
    try:
        run_id = train_model()
        print(f"\n Entrenamiento completado exitosamente")
        print(f"🔑 Run ID: {run_id}\n")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)