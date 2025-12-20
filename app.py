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
try:
    from spark_analyzer import SparkDataAnalyzer, prepare_spark_ui
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

st.set_page_config(
    page_title="MedGuard AI - Умная диагностика заболеваний",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    st.markdown('<h1 class="main-header">🏥 MedGuard AI</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Умная система диагностики сердечно-сосудистых заболеваний и диабета на основе ансамбля ML моделей</p>',
        unsafe_allow_html=True)

    heart_ensemble, heart_scaler, heart_metadata, heart_individual_models = load_ensemble_model()
    diabetes_ensemble, diabetes_scaler, diabetes_metadata, diabetes_individual_models = load_diabetes_ensemble()

    # Навигация в сайдбаре
    with st.sidebar:
        st.markdown("## 🧭 Навигация")
        if SPARK_AVAILABLE:
            navigation_options = [
                "🏠 Главная",
                "❤️ Диагностика сердца",
                "🍭 Диагностика диабета",
                "📊 ML",
                "⚡ Spark Анализ",
                "ℹ️ О проекте"
            ]
        else:
            navigation_options = [
                "🏠 Главная",
                "❤️ Диагностика сердца",
                "🍭 Диагностика диабета",
                "📊 ML",
                "ℹ️ О проекте"
            ]

        page = st.radio("Навигация", navigation_options)
        # page = st.radio("Навигация", ["🏠 Главная", "❤️ Диагностика сердца", "🍭 Диагностика диабета", "📊 Анализ & ML","ℹ️ О проекте"])


        try:
            heart_df = pd.read_csv('data/heart_dataset.csv', sep=';')
            heart_total = len(heart_df)
            heart_healthy = len(heart_df[heart_df['cardio'] == 0])
            heart_sick = len(heart_df[heart_df['cardio'] == 1])

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
    elif page == "📊 ML":
        show_analysis_and_ml(heart_metadata, heart_individual_models, diabetes_metadata, diabetes_individual_models)
    elif page == "⚡ Spark Анализ":
        show_spark_analysis_page()
    elif page == "ℹ️ О проекте":
        show_about()


def show_spark_analysis_page():

    st.header("⚡ Анализ больших данных с apache spark")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
        <h3 style="color: white; margin: 0;">🚀 Распределенная обработка медицинских данных</h3>
    </div>
    """, unsafe_allow_html=True)

    if not SPARK_AVAILABLE:
        st.error("""
        ## ❌ Apache Spark не установлен

        Для использования Spark анализа установите:
        ```bash
        pip install pyspark==3.5.0
        ```

        Преимущества Spark:
        - Обработка миллионов записей за секунды
        - Распределенные вычисления на кластерах
        - SQL-like синтаксис для анализа
        - Интеграция с ML библиотеками
        """)
        return

    if 'spark_analyzer' not in st.session_state:
        with st.spinner("🚀 Запуск Apache Spark..."):
            try:
                st.session_state.spark_analyzer = SparkDataAnalyzer()
                st.success("✅ Spark готов к работе!")
            except Exception as e:
                st.error(f"❌ Ошибка запуска Spark: {e}")
                return

    spark_analyzer = st.session_state.spark_analyzer

    st.subheader("🔍 Выберите тип анализа")

    analysis_type = st.selectbox(
        "Тип анализа",
        [
            "📊 Общая статистика",
            "🎯 Факторы риска",
            "👥 Возрастные группы",
            "📈 Графики"
        ]
    )

    if st.button("▶️ Запустить анализ", type="primary", use_container_width=True):
        with st.spinner("🔄 Выполняю распределенный анализ..."):
            try:
                if analysis_type == "📊 Общая статистика":
                    show_spark_basic_stats(spark_analyzer)

                elif analysis_type == "🎯 Факторы риска":
                    show_spark_risk_factors(spark_analyzer)

                elif analysis_type == "👥 Возрастные группы":
                    show_spark_age_groups(spark_analyzer)

                elif analysis_type == "📈 Графики":
                    show_spark_visualizations_tab()

            except Exception as e:
                st.error(f"❌ Ошибка анализа: {e}")


    if st.button("🛑 Остановить Spark", type="secondary"):
        spark_analyzer.stop()
        del st.session_state.spark_analyzer
        st.success("Spark сессия остановлена")
        st.rerun()


def show_spark_statistics_tab():
    st.subheader("📊 Статистика данных")

    try:
        heart_df = pd.read_csv('data/heart_dataset.csv', sep=';')
        diabetes_df = pd.read_csv('data/diabetes_dataset.csv')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ❤️ Статистика сердца")

            st.metric("Всего пациентов", f"{len(heart_df):,}")
            st.metric("С заболеванием", f"{len(heart_df[heart_df['cardio'] == 1]):,}")
            st.metric("Здоровы", f"{len(heart_df[heart_df['cardio'] == 0]):,}")

            prevalence = len(heart_df[heart_df['cardio'] == 1]) / len(heart_df) * 100
            st.metric("Распространенность", f"{prevalence:.1f}%")

            heart_df['age_years'] = heart_df['age'] / 365
            st.metric("Средний возраст", f"{heart_df['age_years'].mean():.1f} лет")
            st.metric("Среднее давление", f"{heart_df['ap_hi'].mean():.0f}/{heart_df['ap_lo'].mean():.0f}")

        with col2:
            st.markdown("### 🩸 Статистика диабета")

            st.metric("Всего пациентов", f"{len(diabetes_df):,}")
            st.metric("С диабетом", f"{len(diabetes_df[diabetes_df['diabetes'] == 1]):,}")
            st.metric("Здоровы", f"{len(diabetes_df[diabetes_df['diabetes'] == 0]):,}")

            prevalence = len(diabetes_df[diabetes_df['diabetes'] == 1]) / len(diabetes_df) * 100
            st.metric("Распространенность", f"{prevalence:.1f}%")

            # Дополнительная статистика
            st.metric("Средний возраст", f"{diabetes_df['age'].mean():.1f} лет")
            st.metric("Средний BMI", f"{diabetes_df['bmi'].mean():.1f}")
            st.metric("Средний HbA1c", f"{diabetes_df['HbA1c_level'].mean():.1f}%")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")


def show_spark_visualizations_tab():
    try:
        heart_df = pd.read_csv('data/heart_dataset.csv', sep=';')
        diabetes_df = pd.read_csv('data/diabetes_dataset.csv')


        viz_tabs = st.tabs(["❤️ Сердце", "🩸 Диабет"])

        with viz_tabs[0]:
            st.markdown("### ❤️ Визуализации данных сердца")

            # Преобразуем возраст в годы для визуализации
            heart_df['age_years'] = heart_df['age'] / 365

            col1, col2 = st.columns(2)

            with col1:
                # Гистограмма возраста
                fig = px.histogram(heart_df, x='age_years', color='cardio',
                                   title="Распределение возраста по заболеваниям сердца",
                                   color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'},
                                   nbins=30)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Box plot давления
                fig = px.box(heart_df, x='cardio', y='ap_hi',
                             title="Систолическое давление по группам",
                             color='cardio',
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Дополнительные графики
            col1, col2 = st.columns(2)

            with col1:
                # Scatter plot давления
                fig = px.scatter(heart_df.sample(1000) if len(heart_df) > 1000 else heart_df,
                                 x='ap_hi', y='ap_lo',
                                 color='cardio',
                                 title="Систолическое vs Диастолическое давление",
                                 color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Распределение по полу
                gender_heart = heart_df.groupby(['gender', 'cardio']).size().reset_index(name='count')
                gender_heart['gender'] = gender_heart['gender'].map({1: 'Женщины', 2: 'Мужчины'})
                fig = px.bar(gender_heart, x='gender', y='count', color='cardio',
                             title="Распределение заболеваний по полу",
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with viz_tabs[1]:
            st.markdown("### 🩸 Визуализации данных диабета")

            col1, col2 = st.columns(2)

            with col1:
                # Гистограмма возраста
                fig = px.histogram(diabetes_df, x='age', color='diabetes',
                                   title="Распределение возраста по наличию диабета",
                                   color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'},
                                   nbins=30)
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Box plot BMI
                fig = px.box(diabetes_df, x='diabetes', y='bmi',
                             title="Индекс массы тела по группам",
                             color='diabetes',
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.scatter(diabetes_df.sample(1000) if len(diabetes_df) > 1000 else diabetes_df,
                                 x='blood_glucose_level', y='HbA1c_level',
                                 color='diabetes',
                                 title="Глюкоза vs HbA1c",
                                 color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                gender_diabetes = diabetes_df.groupby(['gender', 'diabetes']).size().reset_index(name='count')
                fig = px.bar(gender_diabetes, x='gender', y='count', color='diabetes',
                             title="Распределение диабета по полу",
                             color_discrete_map={0: '#1dd1a1', 1: '#ff6b6b'})
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Ошибка создания графиков: {e}")

def show_spark_complete_analysis(spark_analyzer):
    st.subheader("🚀 Полный анализ данных с визуализацией")

    if st.button("▶️ Запустить полный анализ", type="primary", use_container_width=True):
        with st.spinner("🔄 Выполняю полный анализ с визуализацией..."):
            try:
                heart_df = spark_analyzer.load_heart_data()
                diabetes_df = spark_analyzer.load_diabetes_data()

                if not heart_df or not diabetes_df:
                    st.error("❌ Не удалось загрузить данные")
                    return

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("📊 Анализ базовой статистики...")
                show_spark_basic_stats(spark_analyzer)
                progress_bar.progress(25)

                status_text.text("🎯 Анализ факторов риска...")
                show_spark_risk_factors(spark_analyzer)
                progress_bar.progress(50)

                status_text.text("📈 Анализ возрастных групп...")
                show_spark_age_groups(spark_analyzer)
                progress_bar.progress(75)

                status_text.text("📊 Создание графиков...")
                show_spark_visualizations_tab()
                progress_bar.progress(100)

                status_text.text("✅ Анализ завершен!")

                st.success("### 🎉 Полный анализ данных успешно выполнен!")

                st.markdown("""
                ### 📋 Что было проанализировано:
                1. **Базовая статистика** - общие показатели по датасетам
                2. **Факторы риска** - ключевые факторы для каждого заболевания
                3. **Возрастные группы** - распределение по возрастам
                4. **Визуализация** - графики и диаграммы для наглядности

                ### ⚡ Преимущества Spark анализа:
                - **Скорость**: Обработка 100,000+ записей за секунды
                - **Масштабируемость**: Работа с большими объемами данных
                - **Гибкость**: SQL-like синтаксис для сложных запросов
                - **Интеграция**: Совместимость с ML библиотеками
                """)

            except Exception as e:
                st.error(f"❌ Ошибка при выполнении анализа: {e}")


def show_spark_basic_stats(spark_analyzer):
    st.subheader("📊 Общая статистика")

    try:
        heart_df = pd.read_csv('data/heart_dataset.csv', sep=';')
        diabetes_df = pd.read_csv('data/diabetes_dataset.csv')

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ❤️ Статистика сердца")

            st.metric("Всего пациентов", f"{len(heart_df):,}")
            st.metric("С заболеванием", f"{len(heart_df[heart_df['cardio'] == 1]):,}")
            st.metric("Здоровы", f"{len(heart_df[heart_df['cardio'] == 0]):,}")

            prevalence = len(heart_df[heart_df['cardio'] == 1]) / len(heart_df) * 100
            st.metric("Распространенность", f"{prevalence:.1f}%")

            # Возрастная статистика
            heart_df['age_years'] = heart_df['age'] / 365
            st.metric("Средний возраст", f"{heart_df['age_years'].mean():.1f} лет")
            st.metric("Среднее давление", f"{heart_df['ap_hi'].mean():.0f}/{heart_df['ap_lo'].mean():.0f}")

        with col2:
            st.markdown("### 🩸 Статистика диабета")

            st.metric("Всего пациентов", f"{len(diabetes_df):,}")
            st.metric("С диабетом", f"{len(diabetes_df[diabetes_df['diabetes'] == 1]):,}")
            st.metric("Здоровы", f"{len(diabetes_df[diabetes_df['diabetes'] == 0]):,}")

            prevalence = len(diabetes_df[diabetes_df['diabetes'] == 1]) / len(diabetes_df) * 100
            st.metric("Распространенность", f"{prevalence:.1f}%")

            # Дополнительная статистика
            st.metric("Средний возраст", f"{diabetes_df['age'].mean():.1f} лет")
            st.metric("Средний ИМТ", f"{diabetes_df['bmi'].mean():.1f}")
            st.metric("Средний гликированный гемоглобин (HbA1c)", f"{diabetes_df['HbA1c_level'].mean():.1f}%")

    except Exception as e:
        st.error(f"❌ Ошибка загрузки данных: {e}")


def show_spark_risk_factors(spark_analyzer):
    st.subheader("🎯 Анализ факторов риска")

    heart_df = spark_analyzer.load_heart_data()

    if heart_df:
        st.markdown("### ❤️ Факторы риска сердечных заболеваний")
        risk_factors = spark_analyzer.analyze_risk_factors(heart_df, 'cardio')

        if risk_factors:
            for factor in risk_factors[:4]:
                col1, col2 = st.columns([2, 1])

                with col1:
                    factor_name = factor.get('factor_name_ru', factor['factor'])
                    st.markdown(f"**{factor_name}**")
                    st.caption(f"Заболеваемость при наличии фактора: {factor['disease_with_factor']:.1f}%")
                    st.caption(f"Заболеваемость без фактора: {factor['disease_without_factor']:.1f}%")

                with col2:
                    st.metric(
                        "Относительный риск",
                        f"{factor['relative_risk']:.2f}",
                        delta=f"+{factor['disease_with_factor'] - factor['disease_without_factor']:.1f}%"
                    )

                st.progress(min(factor['relative_risk'] / 5, 1.0))
                st.divider()

    diabetes_df = spark_analyzer.load_diabetes_data()

    if diabetes_df:
        st.markdown("### 🩸 Факторы риска диабета")
        risk_factors = spark_analyzer.analyze_risk_factors(diabetes_df, 'diabetes')

        if risk_factors:
            for factor in risk_factors[:4]:
                col1, col2 = st.columns([2, 1])

                with col1:
                    factor_name = factor.get('factor_name_ru', factor['factor'])
                    st.markdown(f"**{factor_name}**")
                    st.caption(f"Заболеваемость при наличии фактора: {factor['disease_with_factor']:.1f}%")
                    st.caption(f"Заболеваемость без фактора: {factor['disease_without_factor']:.1f}%")

                with col2:
                    st.metric(
                        "Относительный риск",
                        f"{factor['relative_risk']:.2f}",
                        delta=f"+{factor['disease_with_factor'] - factor['disease_without_factor']:.1f}%"
                    )

                st.progress(min(factor['relative_risk'] / 5, 1.0))
                st.divider()


def show_spark_age_groups(spark_analyzer):
    st.subheader("📈 Распределение по возрастным группам")

    tabs = st.tabs(["❤️ Сердце", "🩸 Диабет"])

    with tabs[0]:
        heart_df = spark_analyzer.load_heart_data()

        if heart_df:
            age_groups = spark_analyzer.analyze_age_groups(heart_df, 'cardio')

            total_patients = sum(group['total'] for group in age_groups)
            st.caption(f"📊 Всего проанализировано пациентов: {total_patients:,}")

            for group in age_groups:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Возрастная группа", group['age_group'])

                with col2:
                    st.metric("Всего пациентов", f"{group['total']:,}")

                with col3:
                    st.metric(
                        "Заболеваемость",
                        f"{group['disease_rate']:.1f}%",
                        f"{group['disease_count']:,} пациентов"
                    )

                st.divider()

    with tabs[1]:
        diabetes_df = spark_analyzer.load_diabetes_data()

        if diabetes_df:
            age_groups = spark_analyzer.analyze_diabetes_age_groups(diabetes_df)

            if not age_groups:
                st.error("⚠️ Не удалось проанализировать возрастные группы для диабета")
            else:
                total_patients = sum(group['total'] for group in age_groups)
                st.caption(f"📊 Всего проанализировано пациентов: {total_patients:,}")

                for group in age_groups:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Возрастная группа", group['age_group'])

                    with col2:
                        st.metric("Всего пациентов", f"{group['total']:,}")

                    with col3:
                        st.metric(
                            "Диабет",
                            f"{group['disease_rate']:.1f}%",
                            f"{group['disease_count']:,} пациентов"
                        )

                    st.divider()


def show_spark_full_analysis(spark_analyzer):
    st.subheader("🚀 Полный анализ данных")

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("🔄 Загрузка данных...")
    heart_df = spark_analyzer.load_heart_data()
    diabetes_df = spark_analyzer.load_diabetes_data()
    progress_bar.progress(25)

    status_text.text("📊 Анализ статистики...")

    col1, col2 = st.columns(2)

    with col1:
        if heart_df:
            heart_stats = spark_analyzer.get_basic_stats(heart_df, "Сердце")
            st.metric("Сердце: пациентов", f"{heart_stats.get('total_patients', 0):,}")

    with col2:
        if diabetes_df:
            diabetes_stats = spark_analyzer.get_basic_stats(diabetes_df, "Диабет")
            st.metric("Диабет: пациентов", f"{diabetes_stats.get('total_patients', 0):,}")

    progress_bar.progress(50)

    status_text.text("🎯 Анализ факторов риска...")

    if heart_df:
        heart_risks = spark_analyzer.analyze_risk_factors(heart_df, 'cardio')
        if heart_risks:
            st.markdown(
                f"**Топ фактор риска для сердца:** {heart_risks[0]['factor']} (риск: {heart_risks[0]['relative_risk']:.2f})")

    if diabetes_df:
        diabetes_risks = spark_analyzer.analyze_risk_factors(diabetes_df, 'diabetes')
        if diabetes_risks:
            st.markdown(
                f"**Топ фактор риска для диабета:** {diabetes_risks[0]['factor']} (риск: {diabetes_risks[0]['relative_risk']:.2f})")

    progress_bar.progress(75)

    status_text.text("🧹 Проверка качества данных...")

    if heart_df:
        heart_issues = spark_analyzer.detect_data_quality_issues(heart_df)
        heart_issue_count = len(heart_issues['missing_values']) + len(heart_issues['outliers'])
        st.markdown(f"**Проблемы с данными сердца:** {heart_issue_count}")

    if diabetes_df:
        diabetes_issues = spark_analyzer.detect_data_quality_issues(diabetes_df)
        diabetes_issue_count = len(diabetes_issues['missing_values']) + len(diabetes_issues['outliers'])
        st.markdown(f"**Проблемы с данными диабета:** {diabetes_issue_count}")

    progress_bar.progress(100)
    status_text.text("✅ Анализ завершен!")

    st.success("### 🎉 Полный анализ данных успешно выполнен!")

    st.markdown("""
    ### 📋 Что было проанализировано:
    1. **Загрузка данных** - Spark DataFrame с оптимизацией
    2. **Базовая статистика** - распределение заболеваний
    3. **Факторы риска** - относительные риски для ключевых показателей
    4. **Качество данных** - пропуски и аномалии

    ### ⚡ Преимущества Spark:
    - Обработка 100,000+ записей за секунды
    - Распределенные вычисления
    - SQL-like синтаксис для сложных запросов
    - Масштабируемость до кластеров
    """)


def show_model_error(model_type):
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

    if st.button("🔍 Проанализировать риск сердечных заболеваний", type="primary", use_container_width=True):
        with st.spinner("🔄 Анализируем данные с помощью ансамбля моделей..."):
            # Преобразование данных для нового датасета
            patient_data = prepare_heart_patient_data(
                age=age, gender=gender, height=height, weight=weight,
                ap_hi=ap_hi, ap_lo=ap_lo, cholesterol=cholesterol, gluc=gluc,
                smoke=smoke, alco=alco, active=active
            )

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

            prediction_result = smart_predict_diabetes(patient_data, ensemble, scaler, individual_models)

            if prediction_result is not None:
                display_diabetes_results(prediction_result, patient_data, metadata, ensemble, individual_models)


def display_heart_results(prediction_result, patient_data, metadata):
    st.markdown("---")
    st.header("🎯 Результаты умной диагностики сердца")

    risk_prob = prediction_result['ensemble_probability']
    confidence = prediction_result['confidence']
    individual_probas = prediction_result['individual_probabilities']

    risk_level, risk_text, color = get_risk_level(risk_prob, confidence)

    st.markdown(f"""
    <div class="risk-card {risk_level}">
        <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">{risk_text}</h2>
        <h3 style="font-size: 1.5rem; margin: 0;">Вероятность: {risk_prob:.1%}</h3>
        <p style="font-size: 1rem; margin-top: 0.5rem;">Уверенность прогноза: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
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

    st.subheader("🤖 Согласие моделей")

    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"

    st.markdown(f"""
    <div class="{confidence_class}">
        <h4>📈 Уверенность прогноза: {confidence:.1%}</h4>
        <p>Стандартное отклонение вероятностей: {prediction_result['agreement']:.3f}</p>
        <p><strong>Интерпретация:</strong> {'Высокое согласие' if confidence > 0.8 else 'Умеренное согласие' if confidence > 0.6 else 'Низкое согласие'} между моделями</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💡 Умные рекомендации")
    recommendations = get_recommendations(risk_level, confidence, individual_probas)

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #667eea;">
            <strong>{i}. {rec}</strong>
        </div>
        """, unsafe_allow_html=True)


def display_diabetes_results(prediction_result, patient_data, metadata, ensemble, individual_models):
    st.markdown("---")
    st.header("🎯 Результаты умной диагностики диабета")

    risk_prob = prediction_result['ensemble_probability']
    confidence = prediction_result['confidence']
    individual_probas = prediction_result['individual_probabilities']

    risk_level, risk_text, color = get_diabetes_risk_level(risk_prob, confidence)

    st.markdown(f"""
    <div class="risk-card {risk_level}">
        <h2 style="font-size: 2rem; margin-bottom: 0.5rem;">{risk_text}</h2>
        <h3 style="font-size: 1.5rem; margin: 0;">Вероятность: {risk_prob:.1%}</h3>
        <p style="font-size: 1rem; margin-top: 0.5rem;">Уверенность прогноза: {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
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

    st.subheader("🤖 Согласие моделей")

    confidence_class = "confidence-high" if confidence > 0.8 else "confidence-medium" if confidence > 0.6 else "confidence-low"

    st.markdown(f"""
    <div class="{confidence_class}">
        <h4>📈 Уверенность прогноза: {confidence:.1%}</h4>
        <p>Стандартное отклонение вероятностей: {prediction_result['agreement']:.3f}</p>
        <p><strong>Интерпретация:</strong> {'Высокое согласие' if confidence > 0.8 else 'Умеренное согласие' if confidence > 0.6 else 'Низкое согласие'} между моделями</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💡 Рекомендации по профилактике диабета")
    recommendations = get_diabetes_recommendations(risk_level, confidence, individual_probas)

    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #ff9a9e;">
            <strong>{i}. {rec}</strong>
        </div>
        """, unsafe_allow_html=True)


def show_analysis_and_ml(heart_metadata, heart_individual_models, diabetes_metadata, diabetes_individual_models):
    st.header("📊 ML Модели")

    tab2, tab3 = st.tabs(["❤️ Модели сердца", "🍭 Модели диабета"])

    with tab2:
        show_heart_ml_models_info(heart_metadata, heart_individual_models)

    with tab3:
        show_diabetes_ml_models_info(diabetes_metadata, diabetes_individual_models)


def show_heart_ml_models_info(heart_metadata, heart_individual_models):
    st.subheader("🤖 Машинное обучение для диагностики сердца")

    performance = get_model_performance_info()
    ensemble_info = get_ensemble_info()

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
    st.subheader("🤖 Машинное обучение для диагностики диабета")

    performance = get_diabetes_performance_info()
    ensemble_info = get_diabetes_ensemble_info()

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
        передовых методов машинного обучения и распределенных вычислений.

        ### 🔬 Технологический стек
        - **🤖 Машинное обучение**: Ансамбли моделей с высокой точностью
        - **📊 Data Science**: Анализ медицинских данных и feature engineering
        - **⚡️ Большие данные**: Apache Spark для распределенной обработки данных
        - **🌐 Web технологии**: Streamlit для современного интерфейса
        - **📈 Визуализация**: Plotly для интерактивных графиков


        ### ❤️ Диагностика сердца
        - **Точность ансамбля**: >73%
        - **Данные**: Cardiovascular Disease Dataset (70,000+ пациентов)
        - **Признаки**: возраст, пол, давление, холестерин, глюкоза, курение, алкоголь, активность
        - **Ансамбль**: 6 моделей (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Extra Trees)
        """)

    with col2:
        st.markdown("""
        ### 🍭 Диагностика диабета  
        - **Точность ансамбля**: >97%
        - **Данные**: Diabetes Dataset (100,000+ пациентов)
        - **Признаки**: возраст, пол, гипертония, заболевания сердца, курение, BMI, HbA1c, уровень глюкозы
        - **Ансамбль**: 6 моделей (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, Extra Trees)
        - **Детальный анализ факторов риска**

        ### 🏆 Преимущества нашего подхода
        - **Комбинация ML и Big Data** — точные прогнозы на больших данных
        - **Интерактивная аналитика** — мгновенные ответы на сложные запросы
        - **Визуализация** — профессиональные графики и диаграммы

        ### 📊 Научная основа
        Система обучена на реальных медицинских данных, 
        содержащих информацию о пациентах с различными 
        медицинскими показателями. 

        ### ⚠️ Медицинское предупреждение
        **Важно**: Данная система предназначена для **образовательных 
        и вспомогательных целей** и не заменяет консультацию 
        квалифицированного медицинского специалиста. 
        """)



if __name__ == "__main__":
    main()
