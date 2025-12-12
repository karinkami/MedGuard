import pickle
import pandas as pd


def show_diabetes_info():
    print("=" * 50)
    print("🩸 ИНФОРМАЦИЯ О МОДЕЛИ ДИАБЕТА")
    print("=" * 50)

    try:
        # Загрузка метаданных
        with open('models/diabetes_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Загрузка датасета
        df = pd.read_csv('data/diabetes_dataset.csv')

        print("📊 ДАННЫЕ:")
        print(f"   Размер датасета: {len(df)} пациентов")
        print(f"   Диабет: {len(df[df['diabetes'] == 1])} пациентов ({df['diabetes'].mean() * 100:.1f}%)")

        print("\n🎯 ТОЧНОСТЬ АЛГОРИТМОВ (% верных предсказаний):")
        performance = metadata.get('performance', {})

        # Основные алгоритмы
        print(f"      Random Forest:      {float(performance.get('random_forest', 0)) * 100:.1f}%")
        print(f"      Gradient Boosting:  {float(performance.get('gradient_boosting', 0)) * 100:.1f}%")
        print(f"      Extra Trees:        {float(performance.get('extra_trees', 0)) * 100:.1f}%")
        print(f"      XGBoost:            {float(performance.get('xgboost', 0)) * 100:.1f}%")
        print(f"      LightGBM:           {float(performance.get('lightgbm', 0)) * 100:.1f}%")
        print(f"      CatBoost:           {float(performance.get('catboost', 0)) * 100:.1f}%")
        print(f"      Ensemble:        {float(performance.get('ensemble_accuracy', 0)) * 100:.1f}%")

        # Улучшение ансамбля
        improvement = float(performance.get('improvement', 0)) * 100
        if improvement > 0:
            print(f"      Улучшение:       +{improvement:.2f}%")

        print(f"\nДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ:")
        print(f"   ROC-AUC: {performance.get('ensemble_auc', 'N/A')}")
        print(f"   Кросс-валидация: {float(performance.get('cv_mean', 0)) * 100:.1f}%")

        print(f"\nДата обучения: {metadata.get('training_date', 'N/A')}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def show_heart_info():
    """Показать информацию о модели сердца"""
    print("\n" + "=" * 50)
    print("❤️ ИНФОРМАЦИЯ О МОДЕЛИ СЕРДЦА")
    print("=" * 50)

    try:
        # Загрузка метаданных
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Загрузка датасета
        df = pd.read_csv('data/heart_dataset.csv', sep=';')

        print("📊 ДАННЫЕ:")
        print(f"   Размер датасета: {len(df)} пациентов")
        print(f"   Заболевания: {len(df[df['cardio'] == 1])} пациентов ({df['cardio'].mean() * 100:.1f}%)")

        print("\nТОЧНОСТЬ АЛГОРИТМОВ (% верных предсказаний):")
        performance = metadata.get('performance', {})

        print(f"      Random Forest:      {float(performance.get('random_forest', 0)) * 100:.1f}%")
        print(f"      Gradient Boosting:  {float(performance.get('gradient_boosting', 0)) * 100:.1f}%")
        print(f"      Extra Trees:        {float(performance.get('extra_trees', 0)) * 100:.1f}%")
        print(f"      XGBoost:            {float(performance.get('xgboost', 0)) * 100:.1f}%")
        print(f"      LightGBM:           {float(performance.get('lightgbm', 0)) * 100:.1f}%")
        print(f"      CatBoost:           {float(performance.get('catboost', 0)) * 100:.1f}%")

        print(f"      Ensemble:        {float(performance.get('ensemble', 0)) * 100:.1f}%")

        # Улучшение ансамбля
        improvement = float(metadata.get('improvement_over_best', 0)) * 100
        if improvement > 0:
            print(f"Улучшение:       +{improvement:.2f}%")

        print(f"\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        print(f"   Размер ансамбля: {metadata.get('ensemble_size', 'N/A')} моделей")
        print(f"   Качество данных: {metadata.get('data_cleaning_info', {}).get('final_data_quality', 'N/A')}")

        print(f"\n📅 Дата обучения: {metadata.get('training_date', 'N/A')}")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def show_model_comparison():
    """Показать сравнение алгоритмов между моделями"""
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ АЛГОРИТМОВ: ДИАБЕТ vs СЕРДЦЕ")
    print("=" * 70)

    try:
        # Загрузка метаданных
        with open('models/diabetes_metadata.pkl', 'rb') as f:
            diabetes_meta = pickle.load(f)

        with open('models/metadata.pkl', 'rb') as f:
            heart_meta = pickle.load(f)

        diabetes_perf = diabetes_meta.get('performance', {})
        heart_perf = heart_meta.get('performance', {})

        print("АЛГОРИТМ           |   ДИАБЕТ   |   СЕРДЦЕ   |  РАЗНИЦА")
        print("-" * 55)

        algorithms = [
            ('random_forest', 'Random Forest'),
            ('gradient_boosting', 'Gradient Boosting'),
            ('extra_trees', 'Extra Trees'),
            ('xgboost', 'XGBoost'),
            ('lightgbm', 'LightGBM'),
            ('catboost', 'CatBoost')
        ]

        for key, name in algorithms:
            diabetes_acc = float(diabetes_perf.get(key, 0)) * 100
            heart_acc = float(heart_perf.get(key, 0)) * 100
            difference = diabetes_acc - heart_acc

            diff_symbol = "↑" if difference > 0 else "↓" if difference < 0 else "="
            print(
                f"{name:17} |   {diabetes_acc:5.1f}%   |   {heart_acc:5.1f}%   |  {diff_symbol} {abs(difference):.1f}%")

        print("-" * 55)

        # Ансамбли
        diabetes_ensemble = float(diabetes_perf.get('ensemble_accuracy', 0)) * 100
        heart_ensemble = float(heart_perf.get('ensemble', 0)) * 100
        ensemble_diff = diabetes_ensemble - heart_ensemble
        diff_symbol = "↑" if ensemble_diff > 0 else "↓" if ensemble_diff < 0 else "="

        print(
            f"{'ENSEMBLE':17} |   {diabetes_ensemble:5.1f}%   |   {heart_ensemble:5.1f}%   |  {diff_symbol} {abs(ensemble_diff):.1f}%")

        print(f"\nИНТЕРПРЕТАЦИЯ:")
        print(
            f"   • Лучшая модель для диабета: {max([(float(diabetes_perf.get(k, 0)) * 100, k) for k in [a[0] for a in algorithms]])[1]}")
        print(
            f"   • Лучшая модель для сердца: {max([(float(heart_perf.get(k, 0)) * 100, k) for k in [a[0] for a in algorithms]])[1]}")
        print(f"   • Общая точность ансамблей: {(diabetes_ensemble + heart_ensemble) / 2:.1f}%")

    except Exception as e:
        print(f"Ошибка при сравнении моделей: {e}")


def show_best_algorithms():
    """Показать лучшие алгоритмы для каждой задачи"""
    print("\n" + "=" * 60)
    print("ЛУЧШИЕ АЛГОРИТМЫ ДЛЯ КАЖДОЙ ЗАДАЧИ")
    print("=" * 60)

    try:
        with open('models/diabetes_metadata.pkl', 'rb') as f:
            diabetes_meta = pickle.load(f)

        with open('models/metadata.pkl', 'rb') as f:
            heart_meta = pickle.load(f)

        diabetes_perf = diabetes_meta.get('performance', {})
        heart_perf = heart_meta.get('performance', {})

        # Лучшие алгоритмы для диабета
        diabetes_algorithms = []
        for algo in ['random_forest', 'gradient_boosting', 'extra_trees', 'xgboost', 'lightgbm', 'catboost']:
            accuracy = float(diabetes_perf.get(algo, 0)) * 100
            diabetes_algorithms.append((accuracy, algo))

        diabetes_algorithms.sort(reverse=True)

        print("ДИАБЕТ - ТОП 3 АЛГОРИТМА:")
        for i, (acc, algo) in enumerate(diabetes_algorithms[:3], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {medal} {algo.replace('_', ' ').title():15} {acc:.1f}%")

        # Лучшие алгоритмы для сердца
        heart_algorithms = []
        for algo in ['random_forest', 'gradient_boosting', 'extra_trees', 'xgboost', 'lightgbm', 'catboost']:
            accuracy = float(heart_perf.get(algo, 0)) * 100
            heart_algorithms.append((accuracy, algo))

        heart_algorithms.sort(reverse=True)

        print("\nСЕРДЦЕ - ТОП 3 АЛГОРИТМА:")
        for i, (acc, algo) in enumerate(heart_algorithms[:3], 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {medal} {algo.replace('_', ' ').title():15} {acc:.1f}%")

        # Общие выводы
        print(f"\nВЫВОДЫ:")
        best_diabetes_algo = diabetes_algorithms[0][1]
        best_heart_algo = heart_algorithms[0][1]
        print(f"   • Лучший для диабета: {best_diabetes_algo.replace('_', ' ').title()}")
        print(f"   • Лучший для сердца: {best_heart_algo.replace('_', ' ').title()}")

        if best_diabetes_algo == best_heart_algo:
            print(f"    {best_diabetes_algo.replace('_', ' ').title()} показывает лучшие результаты для обеих задач!")
        else:
            print(f"   Разные алгоритмы лучше подходят для разных медицинских задач")

    except Exception as e:
        print(f"❌ Ошибка при анализе лучших алгоритмов: {e}")


if __name__ == "__main__":
    show_diabetes_info()
    show_heart_info()
    show_model_comparison()
    show_best_algorithms()