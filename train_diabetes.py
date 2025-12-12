import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings('ignore')


def load_diabetes_dataset():
    """Загрузка датасета диабета"""
    print("Загрузка датасета диабета...")

    try:
        # Конкретный путь к вашему файлу
        file_path = 'data/diabetes_dataset.csv'

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"Датасет найден: {file_path}")
            print(f"Размер: {df.shape}")
            print(f"Колонки: {df.columns.tolist()}")

            # Покажем первые строки для проверки
            print("\nПервые 3 строки данных:")
            print(df.head(3))

            return df
        else:
            print(f"Файл {file_path} не найден!")
            print("💡 Убедитесь что файл diabetes_dataset.csv находится в папке data/")
            return None

    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return None


def explore_dataset(df):
    """EDA - Анализ данных"""
    print("\nАНАЛИЗ ДАТАСЕТА:")

    print(f"Информация о данных:")
    print(f"   Размер: {df.shape}")
    print(f"   Колонки: {df.columns.tolist()}")

    # Проверяем целевую переменную 'diabetes'
    if 'diabetes' not in df.columns:
        print(" Целевая переменная 'diabetes' не найдена!")
        print(f"   Доступные колонки: {df.columns.tolist()}")
        return False

    print(f"\nЦелевая переменная 'diabetes':")
    target_counts = df['diabetes'].value_counts()
    for value, count in target_counts.items():
        status = "Диабет" if value == 1 else "Нет диабета"
        percentage = count / len(df) * 100
        print(f"   {status} ({value}): {count} пациентов ({percentage:.1f}%)")

    # Анализ категориальных признаков
    categorical_cols = ['gender', 'smoking_history']
    print(f"\nКАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ:")
    for col in categorical_cols:
        if col in df.columns:
            print(f"   {col}: {df[col].value_counts().to_dict()}")

    # Проверяем пропущенные значения
    print(f"\n Пропущенные значения:")
    missing_data = df.isnull().sum()
    has_missing = False
    for col, count in missing_data.items():
        if count > 0:
            print(f"   {col}: {count} пропусков ({count / len(df) * 100:.1f}%)")
            has_missing = True

    if not has_missing:
        print("Пропущенных значений нет")

    return True


def preprocess_diabetes_data(df):
    """Предобработка данных"""
    print("\nПРЕДОБРАБОТКА ДАННЫХ:")
    print("=" * 50)

    df_clean = df.copy()
    initial_size = len(df_clean)

    # 1. Обработка пропущенных значений
    missing_data = df_clean.isnull().sum()
    if missing_data.sum() > 0:
        print("Удаление строк с пропущенными значениями...")
        before = len(df_clean)
        df_clean = df_clean.dropna()
        after = len(df_clean)
        print(f"   Удалено строк: {before - after}")

    # 2. Кодирование категориальных признаков
    print("\nКОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ:")

    # Gender
    if 'gender' in df_clean.columns:
        gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
        df_clean['gender_encoded'] = df_clean['gender'].map(gender_mapping)
        print(f"   gender: {gender_mapping}")

    # Smoking History
    if 'smoking_history' in df_clean.columns:
        smoking_mapping = {
            'never': 0,
            'No Info': 1,
            'current': 2,
            'former': 3,
            'ever': 4,
            'not current': 5
        }
        df_clean['smoking_history_encoded'] = df_clean['smoking_history'].map(smoking_mapping)
        print(f"   smoking_history: {smoking_mapping}")

    # 3. Удаляем дубликаты
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    if before - after > 0:
        print(f"   Удалено дубликатов: {before - after}")

    print(f"Размер после очистки: {df_clean.shape}")
    print(f"   Сохранено: {len(df_clean) / initial_size * 100:.1f}% данных")

    return df_clean


def prepare_features(df):
    """Подготовка признаков для обучения"""
    print("\nПОДГОТОВКА ПРИЗНАКОВ:")

    # Определяем числовые и категориальные признаки
    numeric_features = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    categorical_features = ['gender_encoded', 'smoking_history_encoded']

    # Выбираем только существующие признаки
    available_numeric = [col for col in numeric_features if col in df.columns]
    available_categorical = [col for col in categorical_features if col in df.columns]

    feature_columns = available_numeric + available_categorical

    print(f"   Числовые признаки: {available_numeric}")
    print(f"   Категориальные признакм: {available_categorical}")
    print(f"   Всего признаков: {len(feature_columns)}")

    X = df[feature_columns]
    y = df['diabetes']

    return X, y, feature_columns


def create_optimized_ensemble():
    """Создание оптимизированного ансамбля из 6 лучших моделей"""
    print("\nСОЗДАНИЕ ОПТИМИЗИРОВАННОГО АНСАМБЛЯ...")

    # 1. Random Forest - лучшая для tabular data
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # 2. Gradient Boosting - отличная производительность
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )

    # 3. Extra Trees - разнообразие для ансамбля
    et_model = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )

    # 4. XGBoost - высокая точность
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # 5. LightGBM - скорость и эффективность
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    # 6. CatBoost - работа с категориальными признаками
    catboost_model = CatBoostClassifier(
        iterations=200,
        depth=8,
        learning_rate=0.1,
        random_state=42,
        verbose=False
    )

    models = [
        ('random_forest', rf_model),
        ('gradient_boosting', gb_model),
        ('extra_trees', et_model),
        ('xgboost', xgb_model),
        ('lightgbm', lgbm_model),
        ('catboost', catboost_model)
    ]

    # Ансамбль с мягким голосованием
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )

    print("Ансамбль создан из 6 моделей:")
    print("   • Random Forest")
    print("   • Gradient Boosting")
    print("   • Extra Trees")
    print("   • XGBoost")
    print("   • LightGBM")
    print("   • CatBoost")

    return ensemble, dict(models)


def evaluate_model(model, X_test, y_test, model_name):
    """Оценка модели"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"   {model_name:25} | Accuracy: {accuracy:.4f} | ROC-AUC: {roc_auc:.4f}")

    return accuracy, roc_auc


def train_diabetes_ensemble():
    """Обучение ансамбля на датасете диабета"""
    print("ЗАПУСК ОБУЧЕНИЯ НА ДАТАСЕТЕ ДИАБЕТА")
    print("=" * 70)

    # Загрузка данных
    df = load_diabetes_dataset()
    if df is None:
        return

    # EDA
    if not explore_dataset(df):
        return

    # Предобработка
    df_clean = preprocess_diabetes_data(df)

    if len(df_clean) == 0:
        print("Нет данных после очистки!")
        return

    # Подготовка признаков
    X, y, feature_names = prepare_features(df_clean)

    print(f"\n📊 ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
    print(f"   Признаки ({len(feature_names)}): {feature_names}")
    print(f"   Примеры: {X.shape[0]}")
    print(f"   Распределение целевой переменной:")
    target_counts = y.value_counts()
    for value, count in target_counts.items():
        status = "Диабет" if value == 1 else "Нет диабета"
        percentage = count / len(y) * 100
        print(f"     {status} ({value}): {count} пациентов ({percentage:.1f}%)")

    # Масштабирование только числовых признаков
    print("\n⚖️  МАСШТАБИРОВАНИЕ ПРИЗНАКОВ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nРАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   Обучающая выборка: {X_train.shape[0]}")
    print(f"   Тестовая выборка: {X_test.shape[0]}")

    # Создание ансамбля
    ensemble, individual_models = create_optimized_ensemble()

    # Обучение и оценка индивидуальных моделей
    print("\nОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ:")
    print("=" * 60)

    individual_results = {}

    for name, model in individual_models.items():
        print(f"Обучение {name}...")
        model.fit(X_train, y_train)
        accuracy, roc_auc = evaluate_model(model, X_test, y_test, name)
        individual_results[name] = {'accuracy': accuracy, 'roc_auc': roc_auc}

    # Обучение ансамбля
    print(f"\nОбучение ансамбля...")
    ensemble.fit(X_train, y_train)

    # Оценка ансамбля
    ensemble_accuracy, ensemble_auc = evaluate_model(ensemble, X_test, y_test, "ENSEMBLE")

    # Детальная оценка ансамбля
    y_pred_ensemble = ensemble.predict(X_test)
    y_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]

    print(f"\nДЕТАЛЬНАЯ ОЦЕНКА АНСАМБЛЯ:")
    print(classification_report(y_test, y_pred_ensemble, target_names=['No Diabetes', 'Diabetes']))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred_ensemble)
    print(f"Матрица ошибок:")
    print(f"   True Negative:  {cm[0, 0]} (правильно предсказано 'нет диабета')")
    print(f"   False Positive: {cm[0, 1]} (ложная тревога)")
    print(f"   False Negative: {cm[1, 0]} (пропущен диабет)")
    print(f"   True Positive:  {cm[1, 1]} (правильно предсказано 'диабет')")

    # Сравнение с лучшей индивидуальной моделью
    best_individual_acc = max([result['accuracy'] for result in individual_results.values()])
    improvement = ensemble_accuracy - best_individual_acc

    print(f"\nСРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print(f"   Лучшая индивидуальная модель: {best_individual_acc:.4f}")
    print(f"   Ансамбль: {ensemble_accuracy:.4f}")
    print(f"   Улучшение: +{improvement:.4f}")

    # Кросс-валидация
    print(f"\nКРОСС-ВАЛИДАЦИЯ (5-fold)...")
    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='accuracy')
    print(f"   CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
    print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Сохранение моделей
    print("\n СОХРАНЕНИЕ МОДЕЛЕЙ...")
    os.makedirs('models', exist_ok=True)

    # Сохраняем ансамбль
    with open('models/diabetes_ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)

    # Сохраняем индивидуальные модели
    for name, model in individual_models.items():
        with open(f'models/diabetes_{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Сохраняем scaler
    with open('models/diabetes_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Сохраняем информацию о кодировании
    encoding_info = {
        'gender_mapping': {'Female': 0, 'Male': 1, 'Other': 2},
        'smoking_mapping': {
            'never': 0, 'No Info': 1, 'current': 2,
            'former': 3, 'ever': 4, 'not current': 5
        }
    }

    # Сохраняем метаданные
    metadata = {
        'feature_names': feature_names,
        'target_column': 'diabetes',
        'encoding_info': encoding_info,
        'performance': {
            'ensemble_accuracy': f"{ensemble_accuracy:.4f}",
            'ensemble_auc': f"{ensemble_auc:.4f}",
            'random_forest': f"{individual_results['random_forest']['accuracy']:.4f}",
            'gradient_boosting': f"{individual_results['gradient_boosting']['accuracy']:.4f}",
            'extra_trees': f"{individual_results['extra_trees']['accuracy']:.4f}",
            'xgboost': f"{individual_results['xgboost']['accuracy']:.4f}",
            'lightgbm': f"{individual_results['lightgbm']['accuracy']:.4f}",
            'catboost': f"{individual_results['catboost']['accuracy']:.4f}",
            'improvement': f"{improvement:.4f}",
            'cv_mean': f"{cv_scores.mean():.4f}"
        },
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'source': 'Diabetes Dataset (diabetes_dataset.csv)',
            'samples': len(X),
            'features': len(feature_names),
            'diabetes_rate': f"{y.mean():.1%}",
            'test_accuracy': f"{ensemble_accuracy:.1%}",
            'test_auc': f"{ensemble_auc:.3f}"
        },
        'model_info': {
            'ensemble_type': 'VotingClassifier',
            'voting': 'soft',
            'models': ['random_forest', 'gradient_boosting', 'extra_trees', 'xgboost', 'lightgbm', 'catboost']
        }
    }

    with open('models/diabetes_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nАНСАМБЛЬ ДЛЯ ДИАБЕТА УСПЕШНО ОБУЧЕН!")
    print(f"Final Ensemble Accuracy: {ensemble_accuracy:.1%}")
    print(f" Final Ensemble ROC-AUC: {ensemble_auc:.3f}")
    print(f"Improvement over best individual: +{improvement:.3f}")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.1%}")
    print(f"\nСохраненные файлы:")
    for name in individual_models.keys():
        print(f"   • models/diabetes_{name}_model.pkl")
    print(f"   • models/diabetes_ensemble_model.pkl")
    print(f"   • models/diabetes_scaler.pkl")
    print(f"   • models/diabetes_metadata.pkl")


if __name__ == "__main__":
    train_diabetes_ensemble()