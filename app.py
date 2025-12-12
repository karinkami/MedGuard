import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils.helpers import (load_ensemble_model, smart_predict,
                           get_risk_level, get_recommendations, get_model_performance_info,
                           get_ensemble_info, prepare_heart_patient_data, analyze_heart_factors,
                           load_diabetes_ensemble, smart_predict_diabetes,
                           get_diabetes_risk_level, get_diabetes_recommendations,
                           get_diabetes_performance_info, get_diabetes_ensemble_info,
                           get_diabetes_feature_descriptions, get_diabetes_medical_thresholds,
                           analyze_diabetes_factors, prepare_diabetes_input,
                           get_gender_options, get_smoking_history_options)

# Настройка страницы
st.set_page_config(
    page_title="MedGuard AI - Умная диагностика заболеваний",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Кастомные стили CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }

    .sub-header {
        font-size: 1.2rem;
        color: #6c757d;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }

    .risk-high { 
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
        color: white !important;
    }
    .risk-medium { 
        background: linear-gradient(135deg, #feca57 0%, #ff9ff3 100%);
        box-shadow: 0 10px 20px rgba(254, 202, 87, 0.3);
        color: #2d3436 !important;
    }
    .risk-low { 
        background: linear-gradient(135deg, #1dd1a1 0%, #00d2d3 100%);
        box-shadow: 0 10px 20px rgba(29, 209, 161, 0.3);
        color: #2d3436 !important;
    }

    .risk-card {
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        transition: transform 0.3s ease;
        font-weight: 600;
    }

    .risk-card:hover {
        transform: translateY(-5px);
    }

    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .feature-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }

    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }

    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        text-align: center;
        border-top: 4px solid #667eea;
    }

    .highlight {
        background: linear-gradient(120deg, #a8edea 0%, #fed6e3 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }

    .confidence-high { background-color: #d4edda; border-left: 4px solid #28a745; padding: 1rem; border-radius: 8px; }
    .confidence-medium { background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 1rem; border-radius: 8px; }
    .confidence-low { background-color: #f8d7da; border-left: 4px solid #dc3545; padding: 1rem; border-radius: 8px; }

    /* ИСПРАВЛЕННЫЕ СТИЛИ ДЛЯ КАРТОЧЕК НА ГЛАВНОЙ */
    .heart-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        height: 200px; /* Фиксированная высота */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        transition: all 0.3s ease;
    }

    .heart-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }

    .heart-card h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .heart-card p {
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }

    .diabetes-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        height: 200px; /* Такая же высота */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        transition: all 0.3s ease;
    }

    .diabetes-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(255, 154, 158, 0.3);
    }

    .diabetes-card h3 {
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .diabetes-card p {
        font-size: 1rem;
        line-height: 1.5;
        margin: 0;
    }

    .factor-risk-high { border-left: 4px solid #e74c3c !important; }
    .factor-risk-medium { border-left: 4px solid #f39c12 !important; }
    .factor-risk-low { border-left: 4px solid #27ae60 !important; }
    .factor-risk-normal { border-left: 4px solid #667eea !important; }

    /* Дополнительные стили для выравнивания */
    .card-container {
        display: flex;
        flex-direction: column;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Заголовок приложения
    st.markdown('<h1 class="main-header">🏥 MedGuard AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Умная система диагностики сердечно-сосудистых заболеваний и диабета на основе ансамбля ML моделей</p>',
        unsafe_allow_html=True)

    # Загрузка моделей
    heart_ensemble, heart_scaler, heart_metadata, heart_individual_models = load_ensemble_model()
    diabetes_ensemble, diabetes_scaler, diabetes_metadata, diabetes_individual_models = load_diabetes_ensemble()

    # Навигация в сайдбаре
    with st.sidebar:
        st.markdown("## 🧭 Навигация")
        page = st.radio("Навигация", ["🏠 Главная", "❤️ Диагностика сердца", "🍭 Диагностика диабета", "📊 Анализ & ML",
                                      "ℹ️ О проекте"])


        try:
            # Статистика сердца
            heart_df = pd.read_csv('data/heart_dataset.csv', sep=';')
            heart_total = len(heart_df)
            heart_healthy = len(heart_df[heart_df['cardio'] == 0])
            heart_sick = len(heart_df[heart_df['cardio'] == 1])

            # Статистика диабета
            diabetes_df = pd.read_csv('data/diabetes_dataset.csv')
            diabetes_total = len(diabetes_df)
            diabetes_healthy = len(diabetes_df[diabetes_df['diabetes'] == 0])
            diabetes_sick = len(diabetes_df[diabetes_df['diabetes'] == 1])

        except Exception as e:
            st.info("📁 Данные не загружены")

    if page == "🏠 Главная":
        show_homepage()
    elif page == "❤️ Диагностика сердца":
        if heart_ensemble is None:
            show_model_error("сердца")
        else:
            show_heart_diagnosis(heart_ensemble, heart_scaler, heart_individual_models, heart_metadata)
    elif page == "🍭 Диагностика диабета":
        if diabetes_ensemble is None:
            show_model_error("диабета")
        else:
            show_diabetes_diagnosis(diabetes_ensemble, diabetes_scaler, diabetes_individual_models, diabetes_metadata)
    elif page == "📊 Анализ & ML":
        show_analysis_and_ml(heart_metadata, heart_individual_models, diabetes_metadata, diabetes_individual_models)
    elif page == "ℹ️ О проекте":
        show_about()


def show_model_error(model_type):
    """Показать ошибку загрузки модели"""
    st.error(f"""
    ## ❌ Модели {model_type} не найдены!

    Перед использованием приложения выполните:
    ```bash
    python train_{'heart' if model_type == 'сердца' else 'diabetes'}.py
    ```
    """)


def show_homepage():
    st.markdown("""
    <div class="highlight">
        <h2 style="text-align: center; margin-bottom: 1rem;">🚀 Добро пожаловать в MedGuard AI</h2>
        <p style="text-align: center; font-size: 1.1rem;">
            Комплексная система для ранней диагностики сердечно-сосудистых заболеваний и диабета 
            с использованием машинного обучения
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Карточки функций - ВЫРОВНЕННЫЕ
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="heart-card">
            <div class="card-container">
                <h3>❤️ Диагностика сердца</h3>
                <p>Ансамбль ML моделей для оценки риска сердечно-сосудистых заболеваний на основе медицинских показателей</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="heart-card">
            <div class="card-container">
                <h3>🍭 Диагностика диабета</h3>
                <p>Специализированные модели для выявления риска диабета на основе медицинских показателей и анализов</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="heart-card">
            <div class="card-container">
                <h3>📊 Умная аналитика</h3>
                <p>Интерактивные графики и детальный анализ факторов риска для обоих заболеваний с визуализацией</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Как это работает
    st.markdown("## 🎯 Как это работает")

    steps_col1, steps_col2, steps_col3, steps_col4 = st.columns(4)

    with steps_col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">1️⃣</div>
            <h4 style="margin-bottom: 0.5rem; color: #333;">Выбор диагностики</h4>
            <p style="margin: 0; color: #666; line-height: 1.4;">Выберите тип диагностики - сердце или диабет</p>
        </div>
        """, unsafe_allow_html=True)

    with steps_col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">2️⃣</div>
            <h4 style="margin-bottom: 0.5rem; color: #333;">Ввод данных</h4>
            <p style="margin: 0; color: #666; line-height: 1.4;">Введите медицинские показатели пациента</p>
        </div>
        """, unsafe_allow_html=True)

    with steps_col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">3️⃣</div>
            <h4 style="margin-bottom: 0.5rem; color: #333;">Анализ ML</h4>
            <p style="margin: 0; color: #666; line-height: 1.4;">Ансамбль моделей анализирует факторы риска</p>
        </div>
        """, unsafe_allow_html=True)

    with steps_col4:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; height: 180px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">4️⃣</div>
            <h4 style="margin-bottom: 0.5rem; color: #333;">Результаты</h4>
            <p style="margin: 0; color: #666; line-height: 1.4;">Получите оценку риска и рекомендации</p>
        </div>
        """, unsafe_allow_html=True)

    # Дополнительная информация
    st.markdown("---")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 220px;">
            <h4 style="margin-top: 0; color: #333;">📈 Высокая точность</h4>
            <p style="margin-bottom: 0.5rem; color: #666;">Наши ансамбли ML моделей обеспечивают точность предсказаний:</p>
            <ul style="margin-bottom: 0; color: #666;">
                <li>❤️ Заболевания сердца: <strong style="color: #333;">>73%</strong></li>
                <li>🍭 Сахарный диабет: <strong style="color: #333;">>97%</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with info_col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; height: 220px;">
            <h4 style="margin-top: 0; color: #333;">🛡️ Надежность</h4>
            <p style="margin-bottom: 0.5rem; color: #666;">Система использует несколько моделей для повышения надежности:</p>
            <ul style="margin-bottom: 0; color: #666;">
                <li>Ансамбль из 6 моделей для каждого заболевания</li>
                <li>Оценка уверенности предсказаний</li>
                <li>Анализ согласия между моделями</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def show_heart_diagnosis(ensemble, scaler, individual_models, metadata):
    st.header("❤️ Умная диагностика сердечных заболеваний")
    st.markdown("Введите данные пациента для анализа риска сердечно-сосудистых заболеваний")

    # Форма ввода данных для нового датасета
    with st.container():
        st.subheader("👤 Демографическая информация")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("**Возраст** (лет)", 20, 100, 50, help="Возраст пациента в годах")
            gender = st.radio("**Пол**", [1, 2], format_func=lambda x: "👩 Женский" if x == 1 else "👨 Мужской",
                              help="Пол пациента")

        with col2:
            height = st.slider("**Рост** (см)", 100, 220, 170, help="Рост пациента в сантиметрах")
            weight = st.slider("**Вес** (кг)", 30, 150, 70, help="Вес пациента в килограммах")

        with col3:
            ap_hi = st.slider("**Систолическое давление** (мм рт.ст.)", 80, 250, 120,
                              help="Верхнее артериальное давление")
            ap_lo = st.slider("**Диастолическое давление** (мм рт.ст.)", 40, 150, 80,
                              help="Нижнее артериальное давление")

    with st.container():
        st.subheader("📊 Медицинские показатели")
        col1, col2, col3 = st.columns(3)

        with col1:
            cholesterol = st.selectbox("**Уровень холестерина**",
                                       [1, 2, 3],
                                       format_func=lambda x: {1: "Нормальный", 2: "Выше нормы", 3: "Высокий"}[x],
                                       help="Уровень холестерина в крови")
            gluc = st.selectbox("**Уровень глюкозы**",
                                [1, 2, 3],
                                format_func=lambda x: {1: "Нормальный", 2: "Выше нормы", 3: "Высокий"}[x],
                                help="Уровень глюкозы в крови")

        with col2:
            smoke = st.radio("**Курение**", [0, 1], format_func=lambda x: "✅ Да" if x == 1 else "❌ Нет",
                             help="Курение пациента")
            alco = st.radio("**Употребление алкоголя**", [0, 1], format_func=lambda x: "✅ Да" if x == 1 else "❌ Нет",
                            help="Употребление алкоголя")

        with col3:
            active = st.radio("**Физическая активность**", [0, 1], format_func=lambda x: "✅ Да" if x == 1 else "❌ Нет",
                              help="Регулярная физическая активность")

    # Кнопка анализа
    if st.button("🔍 Проанализировать риск сердечных заболеваний", type="primary", use_container_width=True):
        with st.spinner("🔄 Анализируем данные с помощью ансамбля моделей..."):
            # Преобразование данных для нового датасета
            patient_data = prepare_heart_patient_data(
                age=age, gender=gender, height=height, weight=weight,
                ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc,
                smoke=smoke, alco=alco, active=active
            )

            # Умное предсказание
            prediction_result = smart_predict(patient_data, ensemble, individual_models, scaler)

            if prediction_result is not None:
                display_heart_results(prediction_result, patient_data, metadata)


def show_diabetes_diagnosis(ensemble, scaler, individual_models, metadata):
    st.header("🩺 Умная диагностика диабета")
    st.markdown("Введите данные пациента для анализа риска развития диабета")

    # Форма ввода данных
    with st.container():
        st.subheader("👤 Основная информация")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("**Пол**", get_gender_options(), help="Выберите пол пациента")
            age = st.slider("**Возраст**", 1, 100, 45, help="Возраст пациента в годах")

        with col2:
            hypertension = st.radio("**Гипертония**", [0, 1],
                                    format_func=lambda x: "✅ Есть" if x == 1 else "❌ Нет",
                                    help="Наличие повышенного давления")
            heart_disease = st.radio("**Болезни сердца**", [0, 1],
                                     format_func=lambda x: "✅ Есть" if x == 1 else "❌ Нет",
                                     help="Наличие сердечно-сосудистых заболеваний")

        with col3:
            smoking_history = st.selectbox("**Курение**", get_smoking_history_options(),
                                           help="Выберите статус курения пациента")

    with st.container():
        st.subheader("📊 Медицинские показатели")
        col1, col2, col3 = st.columns(3)

        with col1:
            bmi = st.slider("**Индекс массы тела (ИМТ)**", 10.0, 60.0, 25.0, 0.1,
                            help="Рассчитывается как вес (кг) / рост (м)²")
            hba1c_level = st.slider("**Уровень гликированного гемоглобина** (%)", 3.0, 15.0, 5.5, 0.1,
                                    help="Показатель среднего уровня сахара в крови за 3 месяца")

        with col2:
            blood_glucose_level = st.slider("**Уровень глюкозы в крови** (мг/дл)", 50, 300, 100,
                                            help="Текущий уровень сахара в крови")

    # Кнопка анализа
    if st.button("🔍 Проанализировать риск диабета", type="primary", use_container_width=True):
        with st.spinner("🔄 Анализируем данные с помощью алгоритмов искусственного интеллекта..."):
            # Подготовка данных
            patient_data = prepare_diabetes_input(
                gender=gender,
                age=age,
                hypertension=hypertension,
                heart_disease=heart_disease,
                smoking_history=smoking_history,
                bmi=bmi,
                hba1c=hba1c_level,
                blood_glucose=blood_glucose_level
            )

            # Умное предсказание
            prediction_result = smart_predict_diabetes(patient_data, ensemble, scaler, individual_models)

            if prediction_result is not None:
                display_diabetes_results(prediction_result, patient_data, metadata, ensemble, individual_models)


def display_heart_results(prediction_result, patient_data, metadata):
    """Отображение результатов диагностики сердца"""
    st.markdown("---")
    st.header("🎯 Результаты умной диагностики сердца")

    risk_prob = prediction_result['ensemble_probability']
    confidence = prediction_result['confidence']
    individual_probas = prediction_result['individual_probabilities']

    risk_level, risk_text, color = get_risk_level(risk_prob, confidence)

    # Карточка риска
    st.markdown(f"""
    <div class="risk-card {risk_level}">
        <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">{risk_text}</h2>
        <h3 style="font-size: 1.5rem; margin: 0;">Вероятность: {risk_prob:.1%}</h3>
        <p style="font-size: 1rem; margin-top: 0.5rem;">Уверенность прогноза: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Визуализация
    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart риска
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Уровень риска сердца", 'font': {'size': 20}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#1dd1a1'},
                    {'range': [30, 70], 'color': '#feca57'},
                    {'range': [70, 100], 'color': '#ff6b6b'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Согласие моделей
        models = list(individual_probas.keys())
        probas = [individual_probas[model] * 100 for model in models]

        fig = go.Figure(data=[go.Bar(
            x=probas,
            y=models,
            orientation='h',
            marker_color=['#667eea', '#764ba2', '#f093fb', '#4ecdc4', '#45b7d1', '#96ceb4'],
            text=[f'{p:.1f}%' for p in probas],
            textposition='auto',
        )])
        fig.update_layout(
            title="📊 Вероятности от каждой модели",
            height=400,
            xaxis_title="Вероятность заболевания (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    # Анализ факторов риска
    st.subheader("🔍 Анализ факторов риска")
    factors = analyze_heart_factors(patient_data, metadata)

    for factor in factors:
        risk_class = "factor-risk-high" if "🔴" in factor['Статус'] else "factor-risk-medium" if "🟡" in factor[
            'Статус'] else "factor-risk-low" if "🟢" in factor['Статус'] else "factor-risk-normal"

        st.markdown(f"""
        <div class="feature-card {risk_class}">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <strong>{factor['Показатель']}</strong>
                    <br>
                    <span style="color: #666;">{factor['Значение']}</span>
                </div>
                <div style="text-align: right;">
                    <strong>{factor['Статус']}</strong>
                    <br>
                    <span style="color: #666; font-size: 0.9rem;">{factor['Объяснение']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Детали согласия моделей
    st.subheader("🤖 Согласие моделей")

    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"

    st.markdown(f"""
    <div class="{confidence_class}">
        <h4>📈 Уверенность прогноза: {confidence:.1%}</h4>
        <p>Стандартное отклонение вероятностей: {prediction_result['agreement']:.3f}</p>
        <p><strong>Интерпретация:</strong> {'Высокое согласие' if confidence > 0.8 else 'Умеренное согласие' if confidence > 0.6 else 'Низкое согласие'} между моделями</p>
    </div>
    """, unsafe_allow_html=True)

    # Рекомендации
    st.subheader("💡 Умные рекомендации")
    recommendations = get_recommendations(risk_level, confidence, individual_probas)

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
            <strong>{i}. {rec}</strong>
        </div>
        """, unsafe_allow_html=True)


def display_diabetes_results(prediction_result, patient_data, metadata, ensemble, individual_models):
    """Отображение результатов диагностики диабета для нового датасета"""
    st.markdown("---")
    st.header("🎯 Результаты умной диагностики диабета")

    risk_prob = prediction_result['ensemble_probability']
    confidence = prediction_result['confidence']
    individual_probas = prediction_result['individual_probabilities']

    risk_level, risk_text, color = get_diabetes_risk_level(risk_prob, confidence)

    # Карточка риска
    st.markdown(f"""
    <div class="risk-card {risk_level}">
        <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">{risk_text}</h2>
        <h3 style="font-size: 1.5rem; margin: 0;">Вероятность: {risk_prob:.1%}</h3>
        <p style="font-size: 1rem; margin-top: 0.5rem;">Уверенность прогноза: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    # Визуализация
    col1, col2 = st.columns(2)

    with col1:
        # Gauge chart риска диабета
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Уровень риска диабета", 'font': {'size': 20}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#1dd1a1'},
                    {'range': [30, 70], 'color': '#feca57'},
                    {'range': [70, 100], 'color': '#ff6b6b'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        fig.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Согласие моделей диабета
        models = list(individual_probas.keys())
        probas = [individual_probas[model] * 100 for model in models]

        fig = go.Figure(data=[go.Bar(
            x=probas,
            y=models,
            orientation='h',
            marker_color=['#ff9a9e', '#fecfef', '#fccb94', '#a6c1ee', '#d4a5a5', '#c7ceea'],
            text=[f'{p:.1f}%' for p in probas],
            textposition='auto',
        )])
        fig.update_layout(
            title="📊 Вероятности от каждой модели",
            height=400,
            xaxis_title="Вероятность диабета (%)",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

    # Анализ факторов риска
    st.subheader("🔍 Анализ факторов риска")
    factors = analyze_diabetes_factors(patient_data, metadata)

    for factor in factors:
        risk_class = "factor-risk-high" if "🔴" in factor['Статус'] else "factor-risk-medium" if "🟡" in factor[
            'Статус'] else "factor-risk-low" if "🟢" in factor['Статус'] else "factor-risk-normal"

        st.markdown(f"""
        <div class="feature-card {risk_class}">
            <div style="display: flex; justify-content: between; align-items: center;">
                <div style="flex: 1;">
                    <strong>{factor['Показатель']}</strong>
                    <br>
                    <span style="color: #666;">{factor['Значение']}</span>
                </div>
                <div style="text-align: right;">
                    <strong>{factor['Статус']}</strong>
                    <br>
                    <span style="color: #666; font-size: 0.9rem;">{factor['Объяснение']}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Детали согласия моделей
    st.subheader("🤖 Согласие моделей")

    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"

    st.markdown(f"""
    <div class="{confidence_class}">
        <h4>📈 Уверенность прогноза: {confidence:.1%}</h4>
        <p>Стандартное отклонение вероятностей: {prediction_result['agreement']:.3f}</p>
        <p><strong>Интерпретация:</strong> {'Высокое согласие' if confidence > 0.8 else 'Умеренное согласие' if confidence > 0.6 else 'Низкое согласие'} между моделями</p>
    </div>
    """, unsafe_allow_html=True)

    # Рекомендации по диабету
    st.subheader("💡 Рекомендации по профилактике диабета")
    recommendations = get_diabetes_recommendations(risk_level, confidence, individual_probas)

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ff9a9e;">
            <strong>{i}. {rec}</strong>
        </div>
        """, unsafe_allow_html=True)


def show_analysis_and_ml(heart_metadata, heart_individual_models, diabetes_metadata, diabetes_individual_models):
    """Объединенная страница анализа данных и ML моделей"""
    st.header("📊 Анализ данных & ML Модели")

    # Создаем табы для разделения контента
    tab1, tab2, tab3 = st.tabs(["📈 Анализ данных", "❤️ Модели сердца", "🍭 Модели диабета"])

    with tab1:
        show_data_analysis()

    with tab2:
        show_heart_ml_models_info(heart_metadata, heart_individual_models)

    with tab3:
        show_diabetes_ml_models_info(diabetes_metadata, diabetes_individual_models)


def show_data_analysis():
    """Компонент анализа данных"""
    st.subheader("📈 Анализ медицинских данных")

    try:
        # Создаем табы для разных наборов данных
        heart_tab, diabetes_tab = st.tabs(["❤️ Данные сердца", "🍭 Данные диабета"])

        with heart_tab:
            st.subheader("Данные сердечных заболеваний")
            df_heart = pd.read_csv('data/heart_dataset.csv', sep=';')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего пациентов", len(df_heart))
            with col2:
                st.metric("С заболеванием", len(df_heart[df_heart['cardio'] == 1]))
            with col3:
                st.metric("Здоровы", len(df_heart[df_heart['cardio'] == 0]))
            with col4:
                prevalence = len(df_heart[df_heart['cardio'] == 1]) / len(df_heart) * 100
                st.metric("Распространенность", f"{prevalence:.1f}%")

            # Визуализации для сердца
            col1, col2 = st.columns(2)
            with col1:
                # Преобразуем возраст в годы для визуализации
                df_heart['age_years'] = df_heart['age'] / 365
                fig = px.histogram(df_heart, x='age_years', color='cardio',
                                   title="Распределение возраста по заболеваниям сердца",
                                   color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(df_heart, x='cardio', y='ap_hi',
                             title="Систолическое давление по группам",
                             color='cardio',
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

            # Дополнительные визуализации
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df_heart, x='ap_hi', y='ap_lo',
                                 color='cardio', title="Систолическое vs Диастолическое давление",
                                 color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Распределение по полу и заболеваниям
                gender_heart = df_heart.groupby(['gender', 'cardio']).size().reset_index(name='count')
                gender_heart['gender'] = gender_heart['gender'].map({1: 'Женщины', 2: 'Мужчины'})
                fig = px.bar(gender_heart, x='gender', y='count', color='cardio',
                             title="Распределение заболеваний по полу",
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

        with diabetes_tab:
            st.subheader("Данные диабета")
            df_diabetes = pd.read_csv('data/diabetes_dataset.csv')

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Всего пациентов", len(df_diabetes))
            with col2:
                st.metric("С диабетом", len(df_diabetes[df_diabetes['diabetes'] == 1]))
            with col3:
                st.metric("Здоровы", len(df_diabetes[df_diabetes['diabetes'] == 0]))
            with col4:
                prevalence = len(df_diabetes[df_diabetes['diabetes'] == 1]) / len(df_diabetes) * 100
                st.metric("Распространенность", f"{prevalence:.1f}%")

            # Визуализации для нового датасета диабета
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(df_diabetes, x='age', color='diabetes',
                                   title="Распределение возраста по наличию диабета",
                                   color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.box(df_diabetes, x='diabetes', y='bmi',
                             title="Индекс массы тела по группам",
                             color='diabetes',
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

            # Дополнительные визуализации для нового датасета
            col1, col2 = st.columns(2)
            with col1:
                fig = px.scatter(df_diabetes, x='blood_glucose_level', y='HbA1c_level',
                                 color='diabetes', title="Глюкоза vs HbA1c",
                                 color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Распределение по полу и диабету
                gender_diabetes = df_diabetes.groupby(['gender', 'diabetes']).size().reset_index(name='count')
                fig = px.bar(gender_diabetes, x='gender', y='count', color='diabetes',
                             title="Распределение диабета по полу",
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")
        st.info("Убедитесь, что файлы данных находятся в папке 'data/'")


def show_heart_ml_models_info(heart_metadata, heart_individual_models):
    """Компонент информации о ML моделях сердца"""
    st.subheader("🤖 Машинное обучение для диагностики сердца")

    # Производительность моделей
    performance = get_model_performance_info()
    ensemble_info = get_ensemble_info()

    # Создаем 7 колонок для всех моделей + ансамбль
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        ensemble_acc = float(performance.get('ensemble', '0.736')) * 100
        st.metric("Ансамбль", f"{ensemble_acc:.1f}%")
    with col2:
        rf_acc = float(performance.get('random_forest', '0.737')) * 100
        st.metric("Random Forest", f"{rf_acc:.1f}%")
    with col3:
        gb_acc = float(performance.get('gradient_boosting', '0.729')) * 100
        st.metric("Gradient Boosting", f"{gb_acc:.1f}%")
    with col4:
        et_acc = float(performance.get('extra_trees', '0.733')) * 100
        st.metric("Extra Trees", f"{et_acc:.1f}%")
    with col5:
        xgb_acc = float(performance.get('xgboost', '0.730')) * 100
        st.metric("XGBoost", f"{xgb_acc:.1f}%")
    with col6:
        lgbm_acc = float(performance.get('lightgbm', '0.735')) * 100
        st.metric("LightGBM", f"{lgbm_acc:.1f}%")
    with col7:
        cb_acc = float(performance.get('catboost', '0.735')) * 100
        st.metric("CatBoost", f"{cb_acc:.1f}%")

    # Визуализация сравнения моделей
    models = ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'XGBoost', 'LightGBM', 'CatBoost', 'Ансамбль']
    accuracy = [rf_acc, gb_acc, et_acc, xgb_acc, lgbm_acc,
                float(performance.get('catboost', '0.735')) * 100, ensemble_acc]

    fig = go.Figure(data=[go.Bar(
        x=models,
        y=accuracy,
        marker_color=['#667eea', '#764ba2', '#f093fb', '#4ecdc4', '#45b7d1', '#96ceb4', '#1dd1a1'],
        text=[f'{acc:.1f}%' for acc in accuracy],
        textposition='auto',
    )])
    fig.update_layout(
        title="Точность моделей сердца (Accuracy %)",
        height=400,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

    # Информация об ансамбле
    st.subheader("🏗️ Архитектура ансамбля")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4>🎯 Состав ансамбля</h4>
            <ul>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>Extra Trees</li>
                <li>XGBoost</li>
                <li>LightGBM</li>
                <li>CatBoost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4>⚡ Преимущества</h4>
            <ul>
                <li>Высокая точность предсказаний</li>
                <li>Устойчивость к переобучению</li>
                <li>Оценка уверенности прогнозов</li>
                <li>Анализ согласия моделей</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def show_diabetes_ml_models_info(diabetes_metadata, diabetes_individual_models):
    """Компонент информации о ML моделях диабета"""
    st.subheader("🤖 Машинное обучение для диагностики диабета")

    # Производительность моделей
    performance = get_diabetes_performance_info()
    ensemble_info = get_diabetes_ensemble_info()

    # Создаем 7 колонок для всех моделей + ансамбль
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        ensemble_acc = float(performance.get('ensemble_accuracy', '0.971')) * 100
        st.metric("Ансамбль", f"{ensemble_acc:.1f}%")
    with col2:
        rf_acc = float(performance.get('random_forest', '0.972')) * 100
        st.metric("Random Forest", f"{rf_acc:.1f}%")
    with col3:
        gb_acc = float(performance.get('gradient_boosting', '0.970')) * 100
        st.metric("Gradient Boosting", f"{gb_acc:.1f}%")
    with col4:
        et_acc = float(performance.get('extra_trees', '0.972')) * 100
        st.metric("Extra Trees", f"{et_acc:.1f}%")
    with col5:
        xgb_acc = float(performance.get('xgboost', '0.970')) * 100
        st.metric("XGBoost", f"{xgb_acc:.1f}%")
    with col6:
        lgbm_acc = float(performance.get('lightgbm', '0.971')) * 100
        st.metric("LightGBM", f"{lgbm_acc:.1f}%")
    with col7:
        cb_acc = float(performance.get('catboost', '0.971')) * 100
        st.metric("CatBoost", f"{cb_acc:.1f}%")

    # Визуализация сравнения моделей
    models = ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'XGBoost', 'LightGBM', 'CatBoost', 'Ансамбль']
    accuracy = [rf_acc, gb_acc, et_acc, xgb_acc, lgbm_acc,
                float(performance.get('catboost', '0.971')) * 100, ensemble_acc]

    fig = go.Figure(data=[go.Bar(
        x=models,
        y=accuracy,
        marker_color=['#ff9a9e', '#fecfef', '#fccb94', '#a6c1ee', '#d4a5a5', '#c7ceea', '#1dd1a1'],
        text=[f'{acc:.1f}%' for acc in accuracy],
        textposition='auto',
    )])
    fig.update_layout(
        title="Точность моделей диабета (Accuracy %)",
        height=400,
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig, use_container_width=True)

    # Информация об ансамбле
    st.subheader("🏗️ Архитектура ансамбля")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4>🎯 Состав ансамбля</h4>
            <ul>
                <li>Random Forest</li>
                <li>Gradient Boosting</li>
                <li>Extra Trees</li>
                <li>XGBoost</li>
                <li>LightGBM</li>
                <li>CatBoost</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4>⚡ Особенности</h4>
            <ul>
                <li>Экстремально высокая точность</li>
                <li>Оптимизированные гиперпараметры</li>
                <li>Работа с категориальными признаками</li>
                <li>Интерпретируемость результатов</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def show_about():
    """Страница о проекте"""
    st.header("ℹ️ О проекте MedGuard AI")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 2rem; border-radius: 15px; margin: 2rem 0;">
        <h2 style="color: white; text-align: center;">🏥 Комплексная диагностика с искусственным интеллектом</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Наша миссия
        MedGuard AI — это инновационная система для ранней диагностики 
        сердечно-сосудистых заболеваний и диабета с использованием 
        передовых методов машинного обучения.

        ### 🔬 Технологический стек
        - **Машинное обучение**: Ансамбли моделей с высокой точностью
        - **Data Science**: Анализ медицинских данных и feature engineering
        - **Web технологии**: Streamlit для современного интерфейса
        - **Визуализация**: Plotly для интерактивных графиков

        ### ❤️ Диагностика сердца
        - Точность ансамбля: >73%
        - Данные: Cardiovascular Disease Dataset (70,000+ пациентов)
        - Признаки: возраст, пол, давление, холестерин, глюкоза, курение, алкоголь, активность
        - 6 моделей в ансамбле
        """)

    with col2:
        st.markdown("""
        ### 🍭 Диагностика диабета  
        - Точность ансамбля: >97%
        - Данные: Diabetes Dataset (100,000+ пациентов)
        - Признаки: возраст, пол, гипертония, заболевания сердца, курение, BMI, HbA1c, уровень глюкозы
        - 6 моделей в ансамбле
        - Детальный анализ факторов риска

        ### 📊 Научная основа
        Система обучена на реальных медицинских данных, 
        содержащих информацию о пациентах с различными 
        медицинскими показателями.

        ### ⚠️ Медицинское предупреждение
        Данная система предназначена для образовательных и 
        вспомогательных целей и не заменяет консультацию 
        квалифицированного медицинского специалиста.
        """)


if __name__ == "__main__":
    main()