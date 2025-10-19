.PHONY: install train validate clean ui help all

PYTHON := python3
PIP := $(PYTHON) -m pip

help:
	@echo "═══════════════════════════════════════════"
	@echo "  MLflow CI/CD Pipeline - Comandos"
	@echo "═══════════════════════════════════════════"
	@echo "  make install   → Instalar dependencias"
	@echo "  make train     → Entrenar modelo"
	@echo "  make validate  → Validar modelo"
	@echo "  make ui        → Abrir MLflow UI"
	@echo "  make clean     → Limpiar artefactos"
	@echo "  make all       → Pipeline completo"
	@echo "═══════════════════════════════════════════"

install:
	@echo "Instalando dependencias..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Dependencias instaladas correctamente"

train:
	@echo "Iniciando entrenamiento..."
	@$(PYTHON) train.py

validate:
	@echo "Iniciando validación..."
	@$(PYTHON) validate.py

ui:
	@echo "🌐 Abriendo MLflow UI en http://localhost:5000"
	@echo "   Presiona Ctrl+C para detener"
	@mlflow ui --backend-store-uri file://$(PWD)/mlruns

clean:
	@echo "🧹 Limpiando artefactos..."
	@rm -rf mlruns/
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pkl" -delete 2>/dev/null || true
	@echo "Limpieza completada"

all: install train validate
	@echo ""
	@echo "═══════════════════════════════════════════"
	@echo "Pipeline completo ejecutado exitosamente"
	@echo "═══════════════════════════════════════════"