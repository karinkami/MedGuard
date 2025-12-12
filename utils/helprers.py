import pickle
import numpy as np


def load_ensemble_model():
    """Загрузка ансамбля моделей сердца"""
    try:
        with open('models/ensemble_model.pkl', 'rb') as f:
            ensemble = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Загрузка индивидуальных моделей
        individual_models = {}
        model_files = {
            'random_forest': 'random_forest_model.pkl',
            'gradient_boosting': 'gradient_boosting_model.pkl',
            'extra_trees': 'extra_trees_model.pkl',
            'xgboost': 'xgboost_model.pkl',
            'lightgbm': 'lightgbm_model.pkl',
            'catboost': 'catboost_model.pkl'
        }

        for name, filename in model_files.items():
            try:
                with open(f'models/{filename}', 'rb') as f:
                    individual_models[name] = pickle.load(f)
            except FileNotFoundError:
                print(f"⚠️ Модель {name} не найдена")

        return ensemble, scaler, metadata, individual_models

    except Exception as e:
        print(f"❌ Ошибка загрузки моделей сердца: {e}")
        return None, None, None, None


def load_diabetes_ensemble():
    """Загрузка ансамбля моделей диабета"""
    try:
        with open('models/diabetes_ensemble_model.pkl', 'rb') as f:
            ensemble = pickle.load(f)
        with open('models/diabetes_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/diabetes_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Загрузка индивидуальных моделей
        individual_models = {}
        model_files = {
            'random_forest': 'diabetes_random_forest_model.pkl',
            'gradient_boosting': 'diabetes_gradient_boosting_model.pkl',
            'extra_trees': 'diabetes_extra_trees_model.pkl',
            'xgboost': 'diabetes_xgboost_model.pkl',
            'lightgbm': 'diabetes_lightgbm_model.pkl',
            'catboost': 'diabetes_catboost_model.pkl'
        }

        for name, filename in model_files.items():
            try:
                with open(f'models/{filename}', 'rb') as f:
                    individual_models[name] = pickle.load(f)
            except FileNotFoundError:
                print(f"⚠️ Модель диабета {name} не найдена")

        return ensemble, scaler, metadata, individual_models

    except Exception as e:
        print(f"❌ Ошибка загрузки моделей диабета: {e}")
        return None, None, None, None


def smart_predict(patient_data, ensemble, individual_models, scaler):
    """Умное предсказание с оценкой уверенности для сердца"""
    try:
        # Масштабирование
        patient_scaled = scaler.transform([patient_data])

        # Предсказание ансамбля
        ensemble_proba = ensemble.predict_proba(patient_scaled)[0][1]

        # Предсказания индивидуальных моделей
        individual_probas = {}
        for name, model in individual_models.items():
            individual_probas[name] = model.predict_proba(patient_scaled)[0][1]

        # Оценка согласия моделей (чем меньше std, тем выше уверенность)
        agreement = np.std(list(individual_probas.values()))
        confidence = max(1 - agreement, 0.1)  # Минимальная уверенность 10%

        result = {
            'ensemble_probability': ensemble_proba,
            'individual_probabilities': individual_probas,
            'confidence': min(confidence, 1.0),
            'agreement': agreement
        }

        return result

    except Exception as e:
        print(f"❌ Ошибка предсказания: {e}")
        return None


def smart_predict_diabetes(patient_data, ensemble, scaler, individual_models):
    """Умное предсказание диабета с оценкой уверенности"""
    try:
        # Масштабирование
        patient_scaled = scaler.transform([patient_data])

        # Предсказание ансамбля
        ensemble_proba = ensemble.predict_proba(patient_scaled)[0][1]

        # Предсказания индивидуальных моделей
        individual_probas = {}
        for name, model in individual_models.items():
            individual_probas[name] = model.predict_proba(patient_scaled)[0][1]

        # Оценка согласия моделей
        agreement = np.std(list(individual_probas.values()))
        confidence = max(1 - agreement, 0.1)  # Минимальная уверенность 10%

        result = {
            'ensemble_probability': ensemble_proba,
            'individual_probabilities': individual_probas,
            'confidence': min(confidence, 1.0),
            'agreement': agreement
        }

        return result

    except Exception as e:
        print(f"❌ Ошибка предсказания диабета: {e}")
        return None


def get_risk_level(risk_prob, confidence=1.0):
    """Определение уровня риска с учетом уверенности для сердца"""
    # Корректируем риск на основе уверенности
    if confidence < 0.7:
        # При низкой уверенности сдвигаем к среднему риску
        adjusted_prob = risk_prob * 0.7 + 0.3 * 0.5
    else:
        adjusted_prob = risk_prob

    if adjusted_prob >= 0.7:
        return "high", "🔴 ВЫСОКИЙ РИСК", "#e74c3c"
    elif adjusted_prob >= 0.4:
        return "medium", "🟡 СРЕДНИЙ РИСК", "#f39c12"
    else:
        return "low", "🟢 НИЗКИЙ РИСК", "#27ae60"


def get_diabetes_risk_level(risk_prob, confidence=1.0):
    """Определение уровня риска диабета с учетом уверенности"""
    # Корректируем риск на основе уверенности
    if confidence < 0.7:
        adjusted_prob = risk_prob * 0.7 + 0.3 * 0.5
    else:
        adjusted_prob = risk_prob

    if adjusted_prob >= 0.7:
        return "high", "🔴 ВЫСОКИЙ РИСК ДИАБЕТА", "#e74c3c"
    elif adjusted_prob >= 0.4:
        return "medium", "🟡 СРЕДНИЙ РИСК ДИАБЕТА", "#f39c12"
    else:
        return "low", "🟢 НИЗКИЙ РИСК ДИАБЕТА", "#27ae60"


def get_recommendations(risk_level, confidence, individual_probas):
    """Получение рекомендаций по уровню риска для сердца"""
    base_recommendations = {
        "high": [
            "🚨 Немедленная консультация кардиолога",
            "📊 ЭКГ и эхокардиография",
            "💊 Суточный мониторинг давления",
            "🥗 Липидограмма и биохимический анализ крови",
            "📋 Полное кардиологическое обследование"
        ],
        "medium": [
            "📅 Плановый осмотр у терапевта",
            "📋 Контроль давления 2 раза в день",
            "🏃‍♂️ Диета с низким содержанием соли и жиров",
            "💪 Регулярная физическая активность",
            "🚭 Отказ от курения и снижение стресса"
        ],
        "low": [
            "💚 Ежегодный профилактический осмотр",
            "🌿 Поддержание здорового образа жизни",
            "🥦 Сбалансированное питание",
            "🚶‍♂️ Регулярные физические нагрузки",
            "😊 Контроль веса и управление стрессом"
        ]
    }

    recommendations = base_recommendations[risk_level]

    # Добавляем рекомендации на основе уверенности
    if confidence < 0.7:
        recommendations.insert(0, "🔍 Рекомендуется дополнительная диагностика для уточнения")

    return recommendations


def get_diabetes_recommendations(risk_level, confidence, individual_probas):
    """Рекомендации при диабете с учетом уверенности"""
    base_recommendations = {
        "high": [
            "🚨 Срочная консультация эндокринолога",
            "📊 Анализ на гликированный гемоглобин (HbA1c)",
            "💊 Регулярный контроль уровня глюкозы",
            "🥗 Строгая диета с низким гликемическим индексом",
            "🏃‍♂️ Ежедневная физическая активность 30+ минут",
            "📋 Полное эндокринологическое обследование"
        ],
        "medium": [
            "📅 Консультация терапевта в ближайшее время",
            "📋 Контроль глюкозы 2 раза в неделю",
            "🥦 Сбалансированное питание с ограничением сахара",
            "🚶‍♂️ Регулярные прогулки и физическая активность",
            "⚖️ Контроль веса и ИМТ",
            "🌿 Коррекция образа жизни"
        ],
        "low": [
            "💚 Ежегодный профилактический осмотр",
            "🌿 Поддержание здорового образа жизни",
            "🥗 Сбалансированное питание",
            "🚶‍♂️ Регулярная физическая активность",
            "😊 Контроль веса и управление стрессом"
        ]
    }

    recommendations = base_recommendations[risk_level]

    # Добавляем рекомендации на основе уверенности
    if confidence < 0.7:
        recommendations.insert(0, "🔍 Рекомендуется дополнительная диагностика для уточнения")

    return recommendations


def get_model_performance_info():
    """Информация о производительности моделей сердца"""
    try:
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return metadata.get('performance', {})
    except:
        return {
            'random_forest': 73.7,
            'gradient_boosting': 72.9,
            'extra_trees': 73.3,
            'xgboost': 73.0,
            'lightgbm': 73.5,
            'catboost': 73.5,
            'ensemble': 73.6
        }


def get_diabetes_performance_info():
    """Информация о производительности моделей диабета"""
    try:
        print("🔍 Пытаюсь загрузить diabetes_metadata.pkl...")

        # Проверим существует ли файл
        import os
        if not os.path.exists('models/diabetes_metadata.pkl'):
            print("❌ Файл models/diabetes_metadata.pkl не существует!")
            raise FileNotFoundError("Файл не найден")

        print("✅ Файл существует, загружаем...")

        with open('models/diabetes_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        print("✅ Метаданные загружены успешно!")
        performance = metadata.get('performance', {})
        print(f"📊 Загруженные данные: {performance}")

        return performance

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в get_diabetes_performance_info(): {e}")
        print(f"🔍 Тип ошибки: {type(e)}")
        return {
            "ensemble_accuracy": "0.971",      # 97.1%
            "random_forest": "0.972",          # 97.2%
            "gradient_boosting": "0.970",      # 97.0%
            "extra_trees": "0.972",            # 97.2%
            "xgboost": "0.970",                # 97.0%
            "lightgbm": "0.971",               # 97.1%
            "catboost": "0.971",               # 97.1%
        }


def get_ensemble_info():
    """Информация об ансамбле сердца"""
    try:
        with open('models/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except:
        return {
            'ensemble_size': 6,
            'improvement_over_best': '0.015'
        }


def get_diabetes_ensemble_info():
    """Информация об ансамбле диабета"""
    try:
        with open('models/diabetes_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return metadata
    except:
        return {
            'ensemble_size': 6,
            'improvement_over_best': '0.010'
        }


def prepare_heart_patient_data(age, gender, height, weight, ap_hi, ap_lo,
                               cholesterol, gluc, smoke, alco, active):
    """ПОДГОТОВКА ДАННЫХ ПАЦИЕНТА ДЛЯ НОВОГО ДАТАСЕТА СЕРДЦА"""

    # Рассчитываем дополнительные признаки как при обучении
    age_years = age  # возраст уже в годах от пользователя
    bmi = weight / ((height / 100) ** 2)
    pulse_pressure = ap_hi - ap_lo
    map_pressure = ap_lo + (pulse_pressure / 3)

    # Кодируем категориальные признаки
    cholesterol_2 = 1 if cholesterol == 2 else 0
    cholesterol_3 = 1 if cholesterol == 3 else 0
    gluc_2 = 1 if gluc == 2 else 0
    gluc_3 = 1 if gluc == 3 else 0

    # Категоризация BMI (как при обучении)
    bmi_category_normal = 1 if 18.5 <= bmi < 25 else 0
    bmi_category_overweight = 1 if 25 <= bmi < 30 else 0
    bmi_category_obese = 1 if bmi >= 30 else 0

    # Порядок признаков должен соответствовать порядку при обучении
    return [
        age_years, gender, height, weight, ap_hi, ap_lo,
        pulse_pressure, map_pressure, bmi,
        smoke, alco, active,
        cholesterol_2, cholesterol_3,
        gluc_2, gluc_3,
        bmi_category_normal, bmi_category_overweight, bmi_category_obese
    ]


def prepare_diabetes_input(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c, blood_glucose):
    """Подготовка входных данных для модели диабета"""
    # Кодирование категориальных признаков
    gender_mapping = {
        'Женский': 0,
        'Female': 0,
        'Мужской': 1,
        'Male': 1,
        'Другой': 2,
        'Other': 2
    }

    smoking_mapping = {
        'Никогда не курил(а)': 0,
        'never': 0,
        'Нет информации': 1,
        'No Info': 1,
        'Курит в настоящее время': 2,
        'current': 2,
        'Бросил(а) курить': 3,
        'former': 3,
        'Курил(а) в прошлом': 4,
        'ever': 4,
        'Не курит в настоящее время': 5,
        'not current': 5
    }

    gender_encoded = gender_mapping.get(gender, 1)  # По умолчанию Мужской
    smoking_encoded = smoking_mapping.get(smoking_history, 1)  # По умолчанию Нет информации

    # Порядок признаков должен соответствовать порядку при обучении
    return [age, hypertension, heart_disease, bmi, hba1c, blood_glucose, gender_encoded, smoking_encoded]

def get_heart_feature_descriptions():
    """Описание признаков для нового датасета сердца"""
    return {
        'age_years': 'Возраст в годах',
        'gender': 'Пол (1 - женщина, 2 - мужчина)',
        'height': 'Рост в см',
        'weight': 'Вес в кг',
        'ap_hi': 'Систолическое давление (верхнее)',
        'ap_lo': 'Диастолическое давление (нижнее)',
        'pulse_pressure': 'Пульсовое давление',
        'map': 'Среднее артериальное давление',
        'bmi': 'Индекс массы тела',
        'smoke': 'Курение (0 - нет, 1 - да)',
        'alco': 'Алкоголь (0 - нет, 1 - да)',
        'active': 'Физическая активность (0 - нет, 1 - да)',
        'cholesterol_2': 'Холестерин уровень 2',
        'cholesterol_3': 'Холестерин уровень 3',
        'gluc_2': 'Глюкоза уровень 2',
        'gluc_3': 'Глюкоза уровень 3',
        'bmi_category_normal': 'Нормальный вес',
        'bmi_category_overweight': 'Избыточный вес',
        'bmi_category_obese': 'Ожирение'
    }


def get_diabetes_feature_descriptions():
    """Описание признаков диабета для нового датасета"""
    return {
        'age': 'Возраст пациента',
        'hypertension': 'Гипертония (0 - нет, 1 - да)',
        'heart_disease': 'Заболевания сердца (0 - нет, 1 - да)',
        'bmi': 'Индекс массы тела',
        'HbA1c_level': 'Уровень гликированного гемоглобина',
        'blood_glucose_level': 'Уровень глюкозы в крови',
        'gender_encoded': 'Пол (0 - женский, 1 - мужской, 2 - другой)',
        'smoking_history_encoded': 'История курения'
    }


def get_heart_medical_thresholds():
    """Медицинские пороговые значения для сердца"""
    return {
        'ap_hi_high': 140,  # Гипертония систолическая
        'ap_lo_high': 90,  # Гипертония диастолическая
        'bmi_obesity': 30,  # Ожирение
        'bmi_overweight': 25,  # Избыточный вес
        'bmi_normal': 18.5,  # Нормальный вес
        'cholesterol_high': 2,  # Высокий холестерин (уровень 2 или 3)
        'gluc_high': 2,  # Высокая глюкоза (уровень 2 или 3)
        'age_risk': 50  # Повышенный риск после 50 лет
    }


def get_diabetes_medical_thresholds():
    """Медицинские пороговые значения для диабета (новый датасет)"""
    return {
        'HbA1c_diabetes': 6.5,  # >6.5% - диагностика диабета
        'HbA1c_prediabetes': 5.7,  # 5.7-6.4% - преддиабет
        'blood_glucose_diabetes': 126,  # >126 мг/дл - диабет (натощак)
        'blood_glucose_prediabetes': 100,  # 100-125 мг/дл - преддиабет
        'BMI_obesity': 30,  # >30 - ожирение
        'BMI_overweight': 25,  # 25-30 - избыточный вес
        'BMI_normal': 18.5,  # 18.5-25 - нормальный вес
        'age_risk': 45,  # >45 лет - повышенный риск
        'hypertension_risk': 1,  # Наличие гипертонии
        'heart_disease_risk': 1  # Наличие болезней сердца
    }


def analyze_heart_factors(patient_data, metadata):
    """Показываем ТОЛЬКО факторы с повышенным риском"""
    thresholds = get_heart_medical_thresholds()
    feature_descriptions = get_heart_feature_descriptions()

    feature_names = metadata.get('feature_names', [])
    factors = []
    feature_dict = dict(zip(feature_names, patient_data))

    # ✅ ТОЛЬКО проверяем риск-факторы, не показываем нормальные
    risk_factors = []

    # Проверяем возраст
    age = feature_dict.get('age_years', 0)
    if age >= thresholds.get('age_risk', 50):
        risk_factors.append({
            'Показатель': 'Возраст',
            'Значение': age,
            'Статус': "🟡 Повышенный риск",
            'Объяснение': f"Возраст {age} ≥ {thresholds.get('age_risk', 50)} лет"
        })

    # Проверяем давление
    ap_hi = feature_dict.get('ap_hi', 0)
    if ap_hi >= thresholds.get('ap_hi_high', 140):
        risk_factors.append({
            'Показатель': 'Систолическое давление',
            'Значение': ap_hi,
            'Статус': "🔴 Высокий риск",
            'Объяснение': f"Давление {ap_hi} ≥ {thresholds.get('ap_hi_high', 140)} (гипертония)"
        })
    elif ap_hi >= 130:
        risk_factors.append({
            'Показатель': 'Систолическое давление',
            'Значение': ap_hi,
            'Статус': "🟡 Повышенный риск",
            'Объяснение': f"Давление {ap_hi} (высокое нормальное)"
        })

    ap_lo = feature_dict.get('ap_lo', 0)
    if ap_lo >= thresholds.get('ap_lo_high', 90):
        risk_factors.append({
            'Показатель': 'Диастолическое давление',
            'Значение': ap_lo,
            'Статус': "🔴 Высокий риск",
            'Объяснение': f"Давление {ap_lo} ≥ {thresholds.get('ap_lo_high', 90)} (гипертония)"
        })

    # Проверяем BMI
    bmi = feature_dict.get('bmi', 0)
    if bmi >= thresholds.get('bmi_obesity', 30):
        risk_factors.append({
            'Показатель': 'Индекс массы тела',
            'Значение': f"{bmi:.1f}",
            'Статус': "🔴 Ожирение",
            'Объяснение': f"ИМТ {bmi:.1f} ≥ {thresholds.get('bmi_obesity', 30)}"
        })
    elif bmi >= thresholds.get('bmi_overweight', 25):
        risk_factors.append({
            'Показатель': 'Индекс массы тела',
            'Значение': f"{bmi:.1f}",
            'Статус': "🟡 Избыточный вес",
            'Объяснение': f"ИМТ {bmi:.1f} ≥ {thresholds.get('bmi_overweight', 25)}"
        })

    # Вредные привычки
    if feature_dict.get('smoke') == 1:
        risk_factors.append({
            'Показатель': 'Курение',
            'Значение': 'Да',
            'Статус': "🔴 Фактор риска",
            'Объяснение': "Курение значительно повышает риск сердечных заболеваний"
        })

    if feature_dict.get('alco') == 1:
        risk_factors.append({
            'Показатель': 'Алкоголь',
            'Значение': 'Да',
            'Статус': "🟡 Фактор риска",
            'Объяснение': "Употребление алкоголя может повышать риск"
        })

    if feature_dict.get('active') == 0:
        risk_factors.append({
            'Показатель': 'Физическая активность',
            'Значение': 'Низкая',
            'Статус': "🟡 Фактор риска",
            'Объяснение': "Низкая физическая активность"
        })

    # Холестерин и глюкоза
    if feature_dict.get('cholesterol_3') == 1:
        risk_factors.append({
            'Показатель': 'Холестерин',
            'Значение': 'Высокий',
            'Статус': "🔴 Высокий риск",
            'Объяснение': "Высокий уровень холестерина"
        })
    elif feature_dict.get('cholesterol_2') == 1:
        risk_factors.append({
            'Показатель': 'Холестерин',
            'Значение': 'Повышенный',
            'Статус': "🟡 Повышенный риск",
            'Объяснение': "Уровень холестерина выше нормы"
        })

    if feature_dict.get('gluc_3') == 1:
        risk_factors.append({
            'Показатель': 'Глюкоза',
            'Значение': 'Высокая',
            'Статус': "🔴 Высокий риск",
            'Объяснение': "Высокий уровень глюкозы"
        })
    elif feature_dict.get('gluc_2') == 1:
        risk_factors.append({
            'Показатель': 'Глюкоза',
            'Значение': 'Повышенная',
            'Статус': "🟡 Повышенный риск",
            'Объяснение': "Уровень глюкозы выше нормы"
        })

    # Если нет факторов риска - показываем позитивное сообщение
    if not risk_factors:
        risk_factors.append({
            'Показатель': 'Общее состояние',
            'Значение': 'Отличное',
            'Статус': "🟢 Низкий риск",
            'Объяснение': "Все основные показатели в норме!"
        })

    return risk_factors


def analyze_diabetes_factors(patient_data, metadata):
    """Анализ факторов риска диабета - ТОЛЬКО проблемные показатели"""
    thresholds = get_diabetes_medical_thresholds()
    feature_descriptions = get_diabetes_feature_descriptions()

    feature_names = metadata.get('feature_names', [
        'age', 'hypertension', 'heart_disease', 'bmi',
        'HbA1c_level', 'blood_glucose_level', 'gender_encoded', 'smoking_history_encoded'
    ])

    factors = []
    feature_dict = dict(zip(feature_names, patient_data))

    # ✅ ТОЛЬКО проверяем риск-факторы, не показываем нормальные
    risk_factors = []

    # Проверяем HbA1c (гликированный гемоглобин)
    hba1c = feature_dict.get('HbA1c_level', 0)
    if hba1c >= thresholds.get('HbA1c_diabetes', 6.5):
        risk_factors.append({
            'Показатель': 'Гликированный гемоглобин (HbA1c)',
            'Значение': f"{hba1c}%",
            'Статус': "🔴 Высокий риск",
            'Объяснение': f"HbA1c {hba1c}% ≥ {thresholds.get('HbA1c_diabetes', 6.5)}% (диабет)"
        })
    elif hba1c >= thresholds.get('HbA1c_prediabetes', 5.7):
        risk_factors.append({
            'Показатель': 'Гликированный гемоглобин (HbA1c)',
            'Значение': f"{hba1c}%",
            'Статус': "🟡 Повышенный риск",
            'Объяснение': f"HbA1c {hba1c}% ≥ {thresholds.get('HbA1c_prediabetes', 5.7)}% (преддиабет)"
        })

    # Проверяем уровень глюкозы в крови
    glucose = feature_dict.get('blood_glucose_level', 0)
    if glucose >= thresholds.get('blood_glucose_diabetes', 126):
        risk_factors.append({
            'Показатель': 'Уровень глюкозы в крови',
            'Значение': f"{glucose} мг/дл",
            'Статус': "🔴 Высокий риск",
            'Объяснение': f"Глюкоза {glucose} ≥ {thresholds.get('blood_glucose_diabetes', 126)} мг/дл (диабет)"
        })
    elif glucose >= thresholds.get('blood_glucose_prediabetes', 100):
        risk_factors.append({
            'Показатель': 'Уровень глюкозы в крови',
            'Значение': f"{glucose} мг/дл",
            'Статус': "🟡 Повышенный риск",
            'Объяснение': f"Глюкоза {glucose} ≥ {thresholds.get('blood_glucose_prediabetes', 100)} мг/дл (преддиабет)"
        })

    # Проверяем BMI
    bmi = feature_dict.get('bmi', 0)
    if bmi >= thresholds.get('BMI_obesity', 30):
        risk_factors.append({
            'Показатель': 'Индекс массы тела',
            'Значение': f"{bmi:.1f}",
            'Статус': "🔴 Ожирение",
            'Объяснение': f"ИМТ {bmi:.1f} ≥ {thresholds.get('BMI_obesity', 30)}"
        })
    elif bmi >= thresholds.get('BMI_overweight', 25):
        risk_factors.append({
            'Показатель': 'Индекс массы тела',
            'Значение': f"{bmi:.1f}",
            'Статус': "🟡 Избыточный вес",
            'Объяснение': f"ИМТ {bmi:.1f} ≥ {thresholds.get('BMI_overweight', 25)}"
        })

    # Проверяем возраст
    age = feature_dict.get('age', 0)
    if age >= thresholds.get('age_risk', 45):
        risk_factors.append({
            'Показатель': 'Возраст',
            'Значение': f"{age} лет",
            'Статус': "🟡 Повышенный риск",
            'Объяснение': f"Возраст {age} ≥ {thresholds.get('age_risk', 45)} лет"
        })

    # Проверяем сопутствующие заболевания
    if feature_dict.get('hypertension') == 1:
        risk_factors.append({
            'Показатель': 'Гипертония',
            'Значение': 'Да',
            'Статус': "🟡 Фактор риска",
            'Объяснение': "Наличие гипертонии повышает риск диабета"
        })

    if feature_dict.get('heart_disease') == 1:
        risk_factors.append({
            'Показатель': 'Заболевания сердца',
            'Значение': 'Да',
            'Статус': "🟡 Фактор риска",
            'Объяснение': "Заболевания сердца повышают риск диабета"
        })

    # Проверяем курение (только активное курение)
    smoking = feature_dict.get('smoking_history_encoded', 1)
    if smoking == 2:  # Курит сейчас
        risk_factors.append({
            'Показатель': 'Курение',
            'Значение': 'Активное',
            'Статус': "🟡 Фактор риска",
            'Объяснение': "Курение повышает риск развития диабета"
        })

    # Если нет факторов риска - показываем позитивное сообщение
    if not risk_factors:
        risk_factors.append({
            'Показатель': 'Общее состояние',
            'Значение': 'Отличное',
            'Статус': "🟢 Низкий риск",
            'Объяснение': "Все основные показатели в норме! Продолжайте вести здоровый образ жизни"
        })

    return risk_factors


def get_smoking_history_options():
    """Опции для истории курения на русском"""
    return [
        "Никогда не курил(а)",
        "Нет информации",
        "Курит в настоящее время",
        "Бросил(а) курить",
        "Курил(а) в прошлом",
        "Не курит в настоящее время"
    ]


def get_gender_options():
    """Опции для пола на русском"""
    return ["Женский", "Мужской", "Другой"]