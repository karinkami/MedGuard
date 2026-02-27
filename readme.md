# 🏥 MedGuard AI

**Интеллектуальная система диагностики заболеваний**  
*Раннее выявление сердечно-сосудистых заболеваний и диабета с помощью ансамбля ML-моделей*

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-orange?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.0-orange?logo=xgboost&logoColor=white)](https://xgboost.ai)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-blue?logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2-yellow?logo=catboost&logoColor=white)](https://catboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/karinkami/MedGuard?style=social)](https://github.com/karinkami/MedGuard/stargazers)

---

## 📋 Оглавление
- [О проекте](#-о-проекте)
- [Возможности](#-возможности)
- [Модели](#-модели)
- [Точность](#-точность)
- [Установка](#-установка)

---

## 🏥 О проекте

**MedGuard AI** — это комплексная система для ранней диагностики заболеваний, использующая **ансамбль из 6 современных ML-алгоритмов**. Проект позволяет с высокой точностью оценивать риски развития сердечно-сосудистых заболеваний и диабета.

### 🎯 Цель
Создать доступный инструмент для первичной диагностики, который могут использовать как медицинские работники, так и обычные пользователи.

---

## ✨ Возможности

### ❤️ Кардиодиагностика
| Параметр | Значение |
|----------|----------|
| **Моделей в ансамбле** | 6 |
| **Точность** | >73% |
| **Анализируемые факторы** | Давление, холестерин, глюкоза, ИМТ, возраст, вредные привычки |

### 🍭 Диагностика диабета
| Параметр | Значение |
|----------|----------|
| **Моделей в ансамбле** | 6 |
| **Точность** | >97% |
| **Анализируемые факторы** | HbA1c, глюкоза, ИМТ, возраст, сопутствующие заболевания |

### 📊 Уникальные особенности
- **Оценка уверенности прогноза** — анализ согласия между моделями
- **Визуализация результатов** — интерактивные графики Plotly
- **Медицинская интерпретация** — понятное объяснение факторов риска
- **Интуитивный интерфейс** — дашборд на Streamlit

---

## 🤖 Модели

В проекте используется ансамбль из 6 алгоритмов машинного обучения:

| Модель | Описание | Преимущества |
|--------|----------|--------------|
| **Random Forest** | Случайный лес | Устойчивость к переобучению |
| **Gradient Boosting** | Градиентный бустинг | Высокая точность |
| **Extra Trees** | Экстремально случайные деревья | Быстрота обучения |
| **XGBoost** | Экстремальный градиентный бустинг | Эффективность на больших данных |
| **LightGBM** | Легкий градиентный бустинг | Скорость и память |
| **CatBoost** | Бустинг с категориальными признаками | Работа с категориями |

> **Принцип работы:** Каждая модель голосует за диагноз, финальное решение принимается на основе консенсуса с учетом уверенности каждого алгоритма.

---

## 📈 Точность

| Заболевание | Точность | AUC-ROC | F1-Score |
|-------------|----------|---------|----------|
| ❤️ Сердечно-сосудистые | **>73%** | 0.78 | 0.72 |
| 🍭 Диабет | **>97%** | 0.98 | 0.96 |

**Факторы, влияющие на точность:**
- Качество обучающих данных (датасеты с Kaggle)
- Ансамблевый подход (снижает ошибки на 15-20%)
- Оценка уверенности (фильтрация сомнительных случаев)

---

## 🚀 Установка

### 1. Клонирование
```bash
git clone https://github.com/karinkami/MedGuard.git
cd MedGuard
```
### 2. Создайте и активируйте виртуальное окружение
```bash
python -m venv venv
venv\Scripts\activate
```
### 3. Установите зависимости
```bash
pip install -r reguirements.txt
```
### 4. Скачайте датасеты
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/data - для сердца
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset - для диабета

### 5. Создание файлов
Создайте папку data. В ней создайте файлы diabetes_dataset.csv - для диабета и heart_dataset.csv - для сердца

### 6. Обучите модели
```bash
python train_diabetes.py
python train_heart.py
```
### 7. Запустите проект
```bash
streamlit run app.py
```
