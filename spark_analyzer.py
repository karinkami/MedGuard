from typing import Dict, List, Optional, Any
import warnings

warnings.filterwarnings('ignore')

try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    import pyspark.sql.functions as F

    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("⚠️  Spark не установлен. Установите: pip install pyspark")


class SparkDataAnalyzer:

    def __init__(self):
        if not SPARK_AVAILABLE:
            raise ImportError("Apache Spark не установлен")

        print("🚀 Запуск Apache Spark...")

        try:
            self.spark = SparkSession.builder \
                .appName("MedGuardAI-Spark-Analyzer") \
                .master("local[*]") \
                .config("spark.driver.memory", "4g") \
                .config("spark.executor.memory", "4g") \
                .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
                .config("spark.sql.shuffle.partitions", "8") \
                .config("spark.default.parallelism", "8") \
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
                .getOrCreate()

            self.spark.sparkContext.setLogLevel("ERROR")

            print(f"✅ Spark готов! Версия: {self.spark.version}")

        except Exception as e:
            print(f"❌ Ошибка инициализации Spark: {e}")
            raise

    def load_heart_data(self) -> Optional[DataFrame]:
        try:
            print("❤️  Загрузка данных сердца...")

            df = self.spark.read \
                .option("delimiter", ";") \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv("data/heart_dataset.csv")

            df = df \
                .withColumn("age_years", col("age") / 365) \
                .withColumn("bmi", col("weight") / ((col("height") / 100) ** 2)) \
                .withColumn("pulse_pressure", col("ap_hi") - col("ap_lo")) \
                .withColumn("map_pressure", col("ap_lo") + ((col("ap_hi") - col("ap_lo")) / 3))

            print(f"📊 Загружено: {df.count():,} записей, {len(df.columns)} признаков")
            return df

        except Exception as e:
            print(f"❌ Ошибка загрузки данных сердца: {e}")
            return None

    def load_diabetes_data(self) -> Optional[DataFrame]:
        try:
            print("🩸 Загрузка данных диабета...")

            df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", "true") \
                .csv("data/diabetes_dataset.csv")

            print(f"📊 Загружено: {df.count():,} записей, {len(df.columns)} признаков")
            return df

        except Exception as e:
            print(f"❌ Ошибка загрузки данных диабета: {e}")
            return None

    def get_basic_stats(self, df: DataFrame, dataset_name: str) -> Dict:
        stats = {}

        try:
            total_count = df.count()
            stats['total_patients'] = total_count

            if 'cardio' in df.columns:
                target_col = 'cardio'
                disease_name = 'сердечных заболеваний'
            elif 'diabetes' in df.columns:
                target_col = 'diabetes'
                disease_name = 'диабета'
            else:
                target_col = None

            if target_col:
                disease_count = df.filter(col(target_col) == 1).count()
                disease_percent = (disease_count / total_count) * 100

                stats['disease_count'] = disease_count
                stats['disease_percent'] = disease_percent
                stats['disease_name'] = disease_name

                print(f"📈 {dataset_name}: {disease_count:,} пациентов с {disease_name} ({disease_percent:.1f}%)")

            if 'age_years' in df.columns:
                age_stats = df.select(
                    F.avg('age_years').alias('avg_age'),
                    F.stddev('age_years').alias('std_age'),
                    F.min('age_years').alias('min_age'),
                    F.max('age_years').alias('max_age')
                ).collect()[0]

                stats['age_stats'] = age_stats.asDict()

            if 'bmi' in df.columns:
                bmi_stats = df.select(
                    F.avg('bmi').alias('avg_bmi'),
                    F.stddev('bmi').alias('std_bmi')
                ).collect()[0]

                stats['bmi_stats'] = bmi_stats.asDict()

            return stats

        except Exception as e:
            print(f"⚠️ Ошибка получения статистики: {e}")
            return stats

    def analyze_risk_factors(self, df: DataFrame, target_col: str = 'cardio') -> List[Dict]:
        try:
            print(f"🔍 Анализ факторов риска для {target_col}...")

            risk_factors = []
            available_columns = df.columns

            factor_translation = {
                'has_hypertension': 'Гипертония',
                'is_obese': 'Ожирение (ИМТ ≥ 30)',
                'high_cholesterol': 'Высокий холестерин',
                'high_glucose': 'Высокий уровень глюкозы',
                'smoke': 'Курение',
                'alcohol': 'Употребление алкоголя',
                'inactive': 'Низкая физическая активность',
                'male_gender': 'Мужской пол',
                'age_60_plus': 'Возраст 60+ лет',
                'age_50_plus': 'Возраст 50+ лет',
                'hypertension': 'Гипертония в анамнезе',
                'heart_disease': 'Болезни сердца в анамнезе',
                'high_hba1c': 'Высокий HbA1c (≥6.5%)',
                'high_blood_glucose': 'Высокий уровень глюкозы в крови',
                'bmi_25_plus': 'Избыточный вес (ИМТ ≥25)',
                'bmi_30_plus': 'Ожирение (ИМТ ≥30)'
            }


            dataset_type = 'heart' if 'cardio' in available_columns else 'diabetes'

            if dataset_type == 'heart':
                if 'ap_hi' in available_columns and 'ap_lo' in available_columns:
                    # Гипертония
                    hypertension_stats = df.withColumn(
                        'has_hypertension',
                        (col('ap_hi') >= 140) | (col('ap_lo') >= 90)
                    )
                    hypertension_risk = self._calculate_risk(hypertension_stats, 'has_hypertension', target_col)
                    if hypertension_risk:
                        hypertension_risk['factor_name_ru'] = factor_translation.get('has_hypertension', 'Гипертония')
                        risk_factors.append(hypertension_risk)

                if 'bmi' in available_columns:
                    obesity_stats = df.withColumn('is_obese', col('bmi') >= 30)
                    obesity_risk = self._calculate_risk(obesity_stats, 'is_obese', target_col)
                    if obesity_risk:
                        obesity_risk['factor_name_ru'] = factor_translation.get('is_obese', 'Ожирение')
                        risk_factors.append(obesity_risk)

                if 'cholesterol' in available_columns:
                    cholesterol_stats = df.withColumn('high_cholesterol', col('cholesterol') >= 2)
                    cholesterol_risk = self._calculate_risk(cholesterol_stats, 'high_cholesterol', target_col)
                    if cholesterol_risk:
                        cholesterol_risk['factor_name_ru'] = factor_translation.get('high_cholesterol',
                                                                                    'Высокий холестерин')
                        risk_factors.append(cholesterol_risk)

                if 'gluc' in available_columns:
                    glucose_stats = df.withColumn('high_glucose', col('gluc') >= 2)
                    glucose_risk = self._calculate_risk(glucose_stats, 'high_glucose', target_col)
                    if glucose_risk:
                        glucose_risk['factor_name_ru'] = factor_translation.get('high_glucose', 'Высокая глюкоза')
                        risk_factors.append(glucose_risk)

                if 'smoke' in available_columns:
                    smoking_risk = self._calculate_risk(df, 'smoke', target_col)
                    if smoking_risk:
                        smoking_risk['factor_name_ru'] = factor_translation.get('smoke', 'Курение')
                        risk_factors.append(smoking_risk)

                if 'alco' in available_columns:
                    alcohol_risk = self._calculate_risk(df, 'alco', target_col)
                    if alcohol_risk:
                        alcohol_risk['factor_name_ru'] = factor_translation.get('alcohol', 'Алкоголь')
                        risk_factors.append(alcohol_risk)

                if 'active' in available_columns:
                    inactive_stats = df.withColumn('inactive', col('active') == 0)
                    inactive_risk = self._calculate_risk(inactive_stats, 'inactive', target_col)
                    if inactive_risk:
                        inactive_risk['factor_name_ru'] = factor_translation.get('inactive', 'Низкая активность')
                        risk_factors.append(inactive_risk)

                if 'gender' in available_columns:
                    gender_stats = df.withColumn('male_gender', col('gender') == 2)
                    gender_risk = self._calculate_risk(gender_stats, 'male_gender', target_col)
                    if gender_risk:
                        gender_risk['factor_name_ru'] = factor_translation.get('male_gender', 'Мужской пол')
                        risk_factors.append(gender_risk)

            elif dataset_type == 'diabetes':
                # Гипертония в анамнезе
                if 'hypertension' in available_columns:
                    hypertension_risk = self._calculate_risk(df, 'hypertension', target_col)
                    if hypertension_risk:
                        hypertension_risk['factor_name_ru'] = factor_translation.get('hypertension', 'Гипертония')
                        risk_factors.append(hypertension_risk)

                if 'heart_disease' in available_columns:
                    heart_disease_risk = self._calculate_risk(df, 'heart_disease', target_col)
                    if heart_disease_risk:
                        heart_disease_risk['factor_name_ru'] = factor_translation.get('heart_disease', 'Болезни сердца')
                        risk_factors.append(heart_disease_risk)

                if 'HbA1c_level' in available_columns:
                    hba1c_stats = df.withColumn('high_hba1c', col('HbA1c_level') >= 6.5)
                    hba1c_risk = self._calculate_risk(hba1c_stats, 'high_hba1c', target_col)
                    if hba1c_risk:
                        hba1c_risk['factor_name_ru'] = factor_translation.get('high_hba1c', 'Высокий HbA1c')
                        risk_factors.append(hba1c_risk)

                if 'blood_glucose_level' in available_columns:
                    glucose_stats = df.withColumn('high_blood_glucose', col('blood_glucose_level') >= 140)
                    glucose_risk = self._calculate_risk(glucose_stats, 'high_blood_glucose', target_col)
                    if glucose_risk:
                        glucose_risk['factor_name_ru'] = factor_translation.get('high_blood_glucose',
                                                                                'Высокая глюкоза в крови')
                        risk_factors.append(glucose_risk)

                if 'bmi' in available_columns:
                    bmi_25_stats = df.withColumn('bmi_25_plus', col('bmi') >= 25)
                    bmi_25_risk = self._calculate_risk(bmi_25_stats, 'bmi_25_plus', target_col)
                    if bmi_25_risk:
                        bmi_25_risk['factor_name_ru'] = factor_translation.get('bmi_25_plus', 'Избыточный вес')
                        risk_factors.append(bmi_25_risk)

                    bmi_30_stats = df.withColumn('bmi_30_plus', col('bmi') >= 30)
                    bmi_30_risk = self._calculate_risk(bmi_30_stats, 'bmi_30_plus', target_col)
                    if bmi_30_risk:
                        bmi_30_risk['factor_name_ru'] = factor_translation.get('bmi_30_plus', 'Ожирение')
                        risk_factors.append(bmi_30_risk)

                if 'age' in available_columns:
                    age_50_stats = df.withColumn('age_50_plus', col('age') >= 50)
                    age_50_risk = self._calculate_risk(age_50_stats, 'age_50_plus', target_col)
                    if age_50_risk:
                        age_50_risk['factor_name_ru'] = factor_translation.get('age_50_plus', 'Возраст 50+ лет')
                        risk_factors.append(age_50_risk)

                    age_60_stats = df.withColumn('age_60_plus', col('age') >= 60)
                    age_60_risk = self._calculate_risk(age_60_stats, 'age_60_plus', target_col)
                    if age_60_risk:
                        age_60_risk['factor_name_ru'] = factor_translation.get('age_60_plus', 'Возраст 60+ лет')
                        risk_factors.append(age_60_risk)

                if 'smoking_history' in available_columns:
                    smoking_stats = df.withColumn('smoking_risk',
                                                  (col('smoking_history') == 'current') |
                                                  (col('smoking_history') == 'former'))
                    smoking_risk = self._calculate_risk(smoking_stats, 'smoking_risk', target_col)
                    if smoking_risk:
                        smoking_risk['factor_name_ru'] = factor_translation.get('smoke', 'Курение (в т.ч. в прошлом)')
                        risk_factors.append(smoking_risk)

            risk_factors.sort(key=lambda x: x.get('relative_risk', 0), reverse=True)

            print(f"✅ Найдено {len(risk_factors)} факторов риска для {dataset_type}")
            return risk_factors[:10]  # Возвращаем топ-10

        except Exception as e:
            print(f"⚠️ Ошибка анализа факторов риска: {e}")
            return []

    def _calculate_risk(self, df: DataFrame, factor_col: str, target_col: str) -> Optional[Dict]:
        try:
            df.createOrReplaceTempView("temp_risk")

            result = self.spark.sql(f"""
                WITH stats AS (
                    SELECT 
                        {factor_col} as has_factor,
                        {target_col} as has_disease,
                        COUNT(*) as cnt
                    FROM temp_risk
                    GROUP BY {factor_col}, {target_col}
                ),
                pivot AS (
                    SELECT
                        MAX(CASE WHEN has_factor = 1 AND has_disease = 1 THEN cnt ELSE 0 END) as a,
                        MAX(CASE WHEN has_factor = 1 AND has_disease = 0 THEN cnt ELSE 0 END) as b,
                        MAX(CASE WHEN has_factor = 0 AND has_disease = 1 THEN cnt ELSE 0 END) as c,
                        MAX(CASE WHEN has_factor = 0 AND has_disease = 0 THEN cnt ELSE 0 END) as d
                    FROM stats
                )
                SELECT 
                    a, b, c, d,
                    (a * 1.0 / (a + b)) / (c * 1.0 / (c + d)) as relative_risk,
                    (a * 1.0 / (a + b)) * 100 as disease_with_factor,
                    (c * 1.0 / (c + d)) * 100 as disease_without_factor
                FROM pivot
            """).collect()[0]

            if result['b'] > 0 and result['c'] > 0:
                return {
                    'factor': factor_col,
                    'relative_risk': float(result['relative_risk']),
                    'disease_with_factor': float(result['disease_with_factor']),
                    'disease_without_factor': float(result['disease_without_factor']),
                    'total_with_factor': int(result['a'] + result['b']),
                    'total_without_factor': int(result['c'] + result['d'])
                }

            return None

        except Exception as e:
            print(f"⚠️ Ошибка расчета риска для {factor_col}: {e}")
            return None

    def analyze_age_groups(self, df: DataFrame, target_col: str = 'cardio') -> List[Dict]:
        try:
            if 'age_years' not in df.columns:
                df = df.withColumn('age_years', col('age') / 365)

            df.createOrReplaceTempView("temp_age")

            age_groups = self.spark.sql(f"""
                SELECT 
                    CASE 
                        WHEN age_years < 18 THEN 'до 18 лет'
                        WHEN age_years < 30 THEN '18-29 лет'
                        WHEN age_years < 40 THEN '30-39 лет'
                        WHEN age_years < 50 THEN '40-49 лет'
                        WHEN age_years < 60 THEN '50-59 лет'
                        WHEN age_years < 70 THEN '60-69 лет'
                        WHEN age_years < 80 THEN '70-79 лет'
                        ELSE '80+ лет'
                    END as age_group,
                    COUNT(*) as total,
                    SUM({target_col}) as disease_count,
                    AVG({target_col}) * 100 as disease_rate,
                    AVG(CASE WHEN {target_col} = 1 THEN age_years ELSE NULL END) as avg_age_sick,
                    AVG(CASE WHEN {target_col} = 0 THEN age_years ELSE NULL END) as avg_age_healthy
                FROM temp_age
                GROUP BY 1
                ORDER BY MIN(age_years)
            """).collect()

            print(f"✅ Проанализировано {len(age_groups)} возрастных групп для сердца")

            return [row.asDict() for row in age_groups]

        except Exception as e:
            print(f"⚠️ Ошибка анализа возрастных групп: {e}")
            return []

    def analyze_diabetes_age_groups(self, df: DataFrame) -> List[Dict]:
        try:
            print("📈 Анализ возрастных групп для диабета...")

            df.createOrReplaceTempView("diabetes_age_temp")

            age_groups = self.spark.sql("""
                SELECT 
                    CASE 
                        WHEN age < 18 THEN 'до 18 лет'
                        WHEN age < 30 THEN '18-29 лет'
                        WHEN age < 40 THEN '30-39 лет'
                        WHEN age < 50 THEN '40-49 лет'
                        WHEN age < 60 THEN '50-59 лет'
                        WHEN age < 70 THEN '60-69 лет'
                        WHEN age < 80 THEN '70-79 лет'
                        ELSE '80+ лет'
                    END as age_group,
                    COUNT(*) as total,
                    SUM(diabetes) as disease_count,
                    AVG(diabetes) * 100 as disease_rate,
                    AVG(bmi) as avg_bmi,
                    AVG(HbA1c_level) as avg_hba1c,
                    AVG(blood_glucose_level) as avg_glucose
                FROM diabetes_age_temp
                GROUP BY 1
                ORDER BY 
                    CASE 
                        WHEN age_group = 'до 18 лет' THEN 1
                        WHEN age_group = '18-29 лет' THEN 2
                        WHEN age_group = '30-39 лет' THEN 3
                        WHEN age_group = '40-49 лет' THEN 4
                        WHEN age_group = '50-59 лет' THEN 5
                        WHEN age_group = '60-69 лет' THEN 6
                        WHEN age_group = '70-79 лет' THEN 7
                        ELSE 8
                    END
            """).collect()

            print(f"✅ Проанализировано {len(age_groups)} возрастных групп для диабета")

            return [row.asDict() for row in age_groups]

        except Exception as e:
            print(f"⚠️ Ошибка анализа возрастных групп для диабета: {e}")
            return []

    def detect_data_quality_issues(self, df: DataFrame) -> Dict:
        issues = {
            'missing_values': {},
            'outliers': {},
            'inconsistencies': []
        }

        try:
            for column in df.columns:
                missing_count = df.filter(col(column).isNull()).count()
                if missing_count > 0:
                    issues['missing_values'][column] = missing_count

            # Проверка аномальных значений для числовых колонок
            numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]

            for col_name in numeric_cols[:5]:
                stats = df.select(
                    F.avg(col_name).alias('mean'),
                    F.stddev(col_name).alias('std')
                ).collect()[0]

                if stats['mean'] is not None and stats['std'] is not None:
                    lower_bound = stats['mean'] - 3 * stats['std']
                    upper_bound = stats['mean'] + 3 * stats['std']

                    outlier_count = df.filter(
                        (col(col_name) < lower_bound) | (col(col_name) > upper_bound)
                    ).count()

                    if outlier_count > 0:
                        issues['outliers'][col_name] = outlier_count

            return issues

        except Exception as e:
            print(f"⚠️ Ошибка проверки качества данных: {e}")
            return issues

    def stop(self):
        try:
            self.spark.stop()
            print("🛑 Spark сессия остановлена")
        except Exception as e:
            print(f"⚠️ Ошибка остановки Spark: {e}")


def prepare_spark_ui():

    if not SPARK_AVAILABLE:
        return {
            'available': False,
            'message': "⚠️ Apache Spark не установлен. Установите: pip install pyspark"
        }

    return {
        'available': True,
        'message': "✅ Apache Spark готов к работе!"
    }
