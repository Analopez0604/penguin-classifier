# 🐧 Penguin Species Classifier

Aplicación web para clasificación de especies de pingüinos desarrollada como proyecto final del curso de Machine Learning.

## Descripción

Esta app predice la especie de un pingüino (Adelie, Chinstrap o Gentoo) a partir de sus características morfológicas usando un modelo de Árbol de Decisión entrenado con el dataset Palmer Penguins.

## Modelo

- **Algoritmo:** Árbol de Decisión (max_depth=4)
- **Accuracy:** 100% en conjunto de prueba
- **F1-Score macro:** 1.0000
- **Dataset:** Palmer Penguins via Seaborn (344 registros, 7 variables)
- **Metodología:** CRISP-DM

## Variables de entrada

| Variable | Descripción |
|---|---|
| bill_length_mm | Longitud del pico (mm) |
| bill_depth_mm | Profundidad del pico (mm) |
| flipper_length_mm | Longitud de la aleta (mm) |
| body_mass_g | Masa corporal (g) |
| island | Isla de observación |
| sex | Sexo del pingüino |

## Ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Institución

Institución Universitaria Pascual Bravo · Curso Machine Learning · Grupo 4 · 2026
