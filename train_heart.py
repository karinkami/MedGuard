import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import pickle
import os


def load_heart_data():
    """Загрузка сердечных данных из файла"""
    try:
        # Загружаем данные с правильным разделителем
        df = pd.read_csv('data/heart_dataset.csv', sep=';')
        print(f"✅ Данные загружены из файла 'heart_dataset.csv': {df.shape}")
        print(f"📊 Столбцы: {list(df.columns)}")

        # Проверяем наличие целевой переменной
        if 'cardio' not in df.columns:
            print("❌ Ошибка: В датасете отсутствует целевая переменная 'cardio'")
            return None

        print(f"🎯 Целевая переменная: 'cardio' (0 - здоров, 1 - болен)")
        print(f"📊 Распределение:")
        target_counts = df['cardio'].value_counts()
        print(f"   Здоровы (cardio=0): {target_counts.get(0, 0)} пациентов")
        print(f"   Больны (cardio=1): {target_counts.get(1, 0)} пациентов")
        print(f"🏥 Процент заболеваний: {df['cardio'].mean():.1%}")

        # Базовая информация о данных
        print(f"\n📈 Информация о данных:")
        print(f"   Возраст: {df['age'].min() / 365:.1f}-{df['age'].max() / 365:.1f} лет")
        print(f"   Рост: {df['height'].min()}-{df['height'].max()} см")
        print(f"   Вес: {df['weight'].min()}-{df['weight'].max()} кг")
        print(f"   Давление: АД сист. {df['ap_hi'].min()}-{df['ap_hi'].max()} мм рт.ст.")
        print(f"   Давление: АД диаст. {df['ap_lo'].min()}-{df['ap_lo'].max()} мм рт.ст.")

        return df

    except Exception as e:
        print(f"❌ Ошибка загрузки данных: {e}")
        return None


def clean_missing_values(df):
    """Очистка пропущенных значений с детальным анализом"""
    print("\n🧹 ОЧИСТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ...")

    df_clean = df.copy()
    initial_size = len(df_clean)

    # Анализ пропущенных значений до очистки
    missing_before = df_clean.isnull().sum()
    total_missing_before = missing_before.sum()

    if total_missing_before == 0:
        print("✅ Пропущенных значений не обнаружено")
        return df_clean

    print("📊 Обнаружены пропущенные значения:")
    for col, count in missing_before.items():
        if count > 0:
            percentage = (count / len(df_clean)) * 100
            print(f"   {col}: {count} пропусков ({percentage:.1f}%)")

    # Удаляем строки с пропущенными значениями
    df_clean = df_clean.dropna()

    # Анализ после очистки
    missing_after = df_clean.isnull().sum().sum()
    removed_count = initial_size - len(df_clean)

    print(f"\n🗑️  Удалено {removed_count} строк с пропущенными значениями")
    print(f"📊 После очистки: {len(df_clean)}/{initial_size} записей ({len(df_clean) / initial_size * 100:.1f}%)")

    if missing_after == 0:
        print("✅ Все пропущенные значения успешно удалены")
    else:
        print(f"⚠️  Осталось пропусков: {missing_after}")

    return df_clean


def preprocess_heart_data(df):
    """Предобработка данных сердечных заболеваний"""
    print("\n🔧 ПРЕДОБРАБОТКА ДАННЫХ...")

    # Начинаем с очистки пропущенных значений
    df_clean = clean_missing_values(df)

    if len(df_clean) == 0:
        print("❌ После очистки не осталось данных!")
        return None, None, None

    # Преобразуем возраст из дней в годы
    df_clean['age_years'] = df_clean['age'] / 365
    print(
        f"📅 Возраст преобразован из дней в годы: {df_clean['age_years'].min():.1f}-{df_clean['age_years'].max():.1f} лет")

    # Рассчитаем BMI (индекс массы тела)
    df_clean['bmi'] = df_clean['weight'] / ((df_clean['height'] / 100) ** 2)
    print(f"⚖️  Рассчитан BMI: {df_clean['bmi'].min():.1f}-{df_clean['bmi'].max():.1f}")

    # Обработка аномальных значений давления
    initial_count = len(df_clean)
    mask = (df_clean['ap_hi'] >= 80) & (df_clean['ap_hi'] <= 250) & \
           (df_clean['ap_lo'] >= 40) & (df_clean['ap_lo'] <= 150) & \
           (df_clean['ap_hi'] >= df_clean['ap_lo'])

    df_clean = df_clean[mask]
    removed_count = initial_count - len(df_clean)
    print(f"📉 Удалено {removed_count} записей с аномальным давлением")

    # Обработка аномальных значений роста и веса
    initial_count = len(df_clean)
    mask = (df_clean['height'] >= 100) & (df_clean['height'] <= 220) & \
           (df_clean['weight'] >= 30) & (df_clean['weight'] <= 200)

    df_clean = df_clean[mask]
    removed_count = initial_count - len(df_clean)
    print(f"📉 Удалено {removed_count} записей с аномальным ростом/весом")

    # Создаем дополнительные медицинские признаки
    df_clean['pulse_pressure'] = df_clean['ap_hi'] - df_clean['ap_lo']  # Пульсовое давление
    df_clean['map'] = df_clean['ap_lo'] + (df_clean['pulse_pressure'] / 3)  # Среднее артериальное давление

    # Категоризация BMI
    df_clean['bmi_category'] = pd.cut(df_clean['bmi'],
                                      bins=[0, 18.5, 25, 30, 100],
                                      labels=['underweight', 'normal', 'overweight', 'obese'])

    # One-hot encoding для категориальных признаков
    df_clean = pd.get_dummies(df_clean, columns=['bmi_category', 'cholesterol', 'gluc'], drop_first=True)

    # Выбираем финальные признаки для модели
    feature_columns = [
        'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'pulse_pressure', 'map', 'bmi',
        'smoke', 'alco', 'active',
        'cholesterol_2', 'cholesterol_3',
        'gluc_2', 'gluc_3',
        'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese'
    ]

    # Убедимся, что все колонки существуют
    available_columns = [col for col in feature_columns if col in df_clean.columns]
    missing_columns = [col for col in feature_columns if col not in df_clean.columns]

    if missing_columns:
        print(f"⚠️  Отсутствующие колонки: {missing_columns}")

    # Целевая переменная - 'cardio'
    target_column = 'cardio'

    print(f"🎯 Используем целевую переменную: '{target_column}'")
    print(f"   Здоровы (0): {len(df_clean[df_clean[target_column] == 0])} пациентов")
    print(f"   Больны (1): {len(df_clean[df_clean[target_column] == 1])} пациентов")

    # Финальная проверка на пропуски в выбранных признаках
    initial_count = len(df_clean)
    df_clean = df_clean[available_columns + [target_column]].dropna()
    removed_count = initial_count - len(df_clean)

    if removed_count > 0:
        print(f"📉 Удалено {removed_count} записей с пропущенными значениями в финальных признаках")

    print(f"✅ Данные предобработаны: {df_clean.shape}")
    print(f"📊 Распределение после очистки:")
    target_counts = df_clean[target_column].value_counts()
    print(f"   Здоровы (cardio=0): {target_counts.get(0, 0)} пациентов")
    print(f"   Больны (cardio=1): {target_counts.get(1, 0)} пациентов")
    print(f"🏥 Процент заболеваний: {df_clean[target_column].mean():.1%}")

    return df_clean[available_columns], df_clean[target_column], available_columns


def create_ensemble():
    print("\n🤖 СОЗДАНИЕ ОПТИМИЗИРОВАННОГО АНСАМБЛЯ...")


    models = [
        ('random_forest', RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )),
        ('gradient_boosting', GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )),
        ('extra_trees', ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('xgboost', XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('lightgbm', LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )),
        ('catboost', CatBoostClassifier(
            iterations=200,
            depth=8,
            learning_rate=0.1,
            random_state=42,
            verbose=False
        ))
    ]

    print("Ансамбль создан из 6 моделей:")
    for name, _ in models:
        print(f"   • {name}")

    # Ансамбль с мягким голосованием
    ensemble = VotingClassifier(
        estimators=models,
        voting='soft',
        n_jobs=-1
    )

    return ensemble, dict(models)


def train_ensemble():
    """Обучение оптимизированного ансамбля на данных сердечных заболеваний"""
    print("🚀 Запуск обучения на данных сердечных заболеваний")
    print("=" * 60)

    # Загружаем данные
    df = load_heart_data()
    if df is None:
        print("❌ Не удалось загрузить данные. Обучение прервано.")
        return

    # Предобработка данных (включая очистку пропусков)
    X, y, feature_names = preprocess_heart_data(df)
    if X is None or len(X) == 0:
        print("❌ Ошибка предобработки данных. Обучение прервано.")
        return

    print(f"\n📊 ДАННЫЕ ДЛЯ ОБУЧЕНИЯ:")
    print(f"   Пациентов: {X.shape[0]}")
    print(f"   Медицинских признаков: {X.shape[1]}")
    print(f"   Признаки: {', '.join(feature_names)}")
    print(f"🎯 Целевая переменная: 'cardio'")
    print(f"   Здоровы (0): {len(y[y == 0])} пациентов")
    print(f"   Больны (1): {len(y[y == 1])} пациентов")
    print(f"🏥 Процент заболеваний: {y.mean():.1%}")

    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📁 РАЗДЕЛЕНИЕ ДАННЫХ:")
    print(f"   Обучающая выборка: {X_train.shape[0]} пациентов")
    print(f"   Тестовая выборка: {X_test.shape[0]} пациентов")

    # Создаем ансамбль
    print("\n🤖 СОЗДАНИЕ ОПТИМИЗИРОВАННОГО АНСАМБЛЯ...")
    ensemble, individual_models = create_ensemble()

    # Обучаем индивидуальные модели и оцениваем
    print("\n🔍 ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ:")
    print("=" * 50)

    individual_results = {}

    for name, model in individual_models.items():
        print(f"🔄 Обучение {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        individual_results[name] = {
            'accuracy': accuracy,
            'roc_auc': roc_auc
        }

        print(f"   {name:20} | Accuracy: {accuracy:.3f} | ROC-AUC: {roc_auc:.3f}")

    # Обучаем ансамбль
    print("\n🔄 ОБУЧЕНИЕ АНСАМБЛЯ...")
    ensemble.fit(X_train, y_train)

    # Оценка ансамбля
    ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    ensemble_pred = (ensemble_proba > 0.5).astype(int)
    ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_proba)

    print(f"\n🏆 РЕЗУЛЬТАТЫ АНСАМБЛЯ:")
    print(f"   Accuracy: {ensemble_accuracy:.3f}")
    print(f"   ROC-AUC: {ensemble_auc:.3f}")

    # Детальная оценка
    print(f"\n📋 ДЕТАЛЬНАЯ ОЦЕНКА:")
    print(classification_report(y_test, ensemble_pred, target_names=['Здоров (cardio=0)', 'Болен (cardio=1)']))

    # Матрица ошибок
    cm = confusion_matrix(y_test, ensemble_pred)
    print(f"📊 Матрица ошибок для 'cardio':")
    print(f"   True Negative (корректно здоров):  {cm[0, 0]}")
    print(f"   False Positive (ложно больны):     {cm[0, 1]}")
    print(f"   False Negative (ложно здоровы):    {cm[1, 0]}")
    print(f"   True Positive (корректно больны):  {cm[1, 1]}")

    # Сравнение с индивидуальными моделями
    best_individual = max([result['accuracy'] for result in individual_results.values()])
    improvement = ensemble_accuracy - best_individual
    print(f"\n📈 УЛУЧШЕНИЕ АНСАМБЛЯ:")
    print(f"   Лучшая индивидуальная модель: {best_individual:.3f}")
    print(f"   Улучшение ансамбля: +{improvement:.3f}")

    # Сохраняем модели
    print("\n💾 СОХРАНЕНИЕ МОДЕЛЕЙ...")
    os.makedirs('models', exist_ok=True)

    # Сохраняем ансамбль
    with open('models/ensemble_model.pkl', 'wb') as f:
        pickle.dump(ensemble, f)

    # Сохраняем индивидуальные модели
    for name, model in individual_models.items():
        with open(f'models/{name}_model.pkl', 'wb') as f:
            pickle.dump(model, f)

    # Сохраняем scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Сохраняем метаданные
    metadata = {
        'feature_names': feature_names,
        'target_variable': 'cardio',
        'target_description': '0 - здоров, 1 - сердечно-сосудистые заболевания',
        'performance': {
            'ensemble': f"{ensemble_accuracy:.3f}",
            'random_forest': f"{individual_results['random_forest']['accuracy']:.3f}",
            'gradient_boosting': f"{individual_results['gradient_boosting']['accuracy']:.3f}",
            'extra_trees': f"{individual_results['extra_trees']['accuracy']:.3f}",
            'xgboost': f"{individual_results['xgboost']['accuracy']:.3f}",
            'lightgbm': f"{individual_results['lightgbm']['accuracy']:.3f}",
            'catboost': f"{individual_results['catboost']['accuracy']:.3f}"
        },
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'source': 'heart_dataset.csv',
            'dataset_name': 'heart_dataset.csv',
            'samples': len(X),
            'features': len(feature_names),
            'disease_rate': f"{y.mean():.1%}",
            'test_accuracy': f"{ensemble_accuracy:.1%}"
        },
        'data_cleaning_info': {
            'missing_values_handling': 'complete_removal',
            'anomaly_detection': 'pressure, height, weight',
            'final_data_quality': 'high'
        },
        'ensemble_size': len(individual_models),
        'improvement_over_best': f"{improvement:.3f}"
    }

    with open('models/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\n✅ АНСАМБЛЬ ОБУЧЕН!")
    print(f"🎯 Итоговая точность: {ensemble_accuracy:.1%}")
    print(f"📊 Данные: {len(X)} пациентов из heart_dataset.csv")
    print(f"🎯 Целевая переменная: 'cardio' (0 - здоров, 1 - болен)")
    print(f"🏥 Процент заболеваний в данных: {y.mean():.1%}")
    print(f"📁 Сохраненные модели:")
    for name in individual_models.keys():
        print(f"   • models/{name}_model.pkl")
    print(f"   • models/ensemble_model.pkl")
    print(f"   • models/scaler.pkl")
    print(f"   • models/metadata.pkl")


if __name__ == "__main__":
    train_ensemble()