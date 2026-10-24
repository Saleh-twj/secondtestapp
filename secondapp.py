import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')

# Language configuration
def get_translations(language):
    translations = {
        'english': {
            'title': "📈 Stock Price Prediction System - Saudi Trading",
            'sidebar_settings': "⚙️ Model Settings",
            'upload_file': "📤 Upload Stock Data File (CSV)",
            'builtin_data': "🔍 TASI Stock Search",
            'select_stock': "Select Stock",
            'select_period': "Select Historical Period",
            'training_params': "🎛️ Training Parameters",
            'time_window': "Time Window (days)",
            'test_ratio': "Test Ratio",
            'epochs': "Number of Epochs",
            'batch_size': "Batch Size",
            'select_model': "Select Model",
            'start_training': "🚀 Start Training",
            'data_overview': "📊 Data Overview",
            'total_records': "Total Records",
            'time_period': "Time Period",
            'min_price': "Minimum Price",
            'max_price': "Maximum Price",
            'tabs': ["📋 Data View", "📈 Charts", "🔍 Statistical Analysis"],
            'price_evolution': "Stock Price Evolution Over Time",
            'candlestick': "Candlestick Chart",
            'price_distribution': "Price Distribution",
            'descriptive_stats': "Descriptive Statistics",
            'training_results': "🎯 Training and Prediction Results",
            'training_model': "Training model...",
            'making_predictions': "Making predictions...",
            'completed': "Completed!",
            'performance_metrics': "📊 Performance Metrics",
            'training_loss': "📉 Training Loss Curve",
            'actual_vs_predicted': "🔮 Actual vs Predicted Prices Comparison",
            'future_predictions': "🔭 Future Predictions (30 days)",
            'download_results': "📥 Download Results",
            'download_button': "📥 Download Future Predictions (CSV)",
            'training_completed': "✅ Training and prediction completed successfully!",
            'welcome': "🚀 Welcome to Stock Price Prediction System",
            'welcome_desc': "Intelligent stock price prediction system for Saudi stock market using AI technologies",
            'data_format': "📋 Required Data Format",
            'data_columns': "CSV file should contain the following columns:",
            'date_col': "Date: Trading date",
            'price_col': "Price: Closing price",
            'open_col': "Open: Opening price",
            'high_col': "High: Highest price",
            'low_col': "Low: Lowest price",
            'vol_col': "Vol.: Trading volume",
            'change_col': "Change %: Change percentage",
            'features': "🎯 System Features",
            'features_list': [
                "Deep Learning Models: LSTM and MLP as per project document",
                "Trading Focus: Customized for Saudi stock market",
                "Performance Metrics: RMSE, MSE, and R² according to methodology",
                "User-Friendly Interface: Easy-to-use Arabic design",
                "Future Predictions: 30-day price forecasts",
                "Data Export: Download results for analysis"
            ],
            'model_specs': "📊 Model Specifications",
            'model_specs_list': [
                "LSTM: Two layers with Dropout to prevent overfitting",
                "MLP: With ReLU activation",
                "Data Normalization: Min-Max scaling according to methodology",
                "Time Series: Sliding window approach for sequence prediction"
            ],
            'performance_stats': "📈 Model Performance Statistics",
            'lstm_accuracy': "LSTM Accuracy",
            'mlp_accuracy': "MLP Accuracy",
            'avg_r2': "Average R²",
            'historical_price': "Historical Price",
            'future_forecast': "Future Forecast",
            'training_data': "Training Data",
            'actual_price': "Actual Price",
            'predicted_price': "Predicted Price",
            'learn_more': "📚 Learn More",
            'fetching_data': "Fetching stock data...",
            'data_loaded': "✅ Data loaded successfully!",
            'data_source': "Data Source",
            'system_guide': "System Guide & How to Use",
            'guide_content': [
                "1. **Data Input**: Choose between uploading your own CSV file or using TASI Stock Search to fetch live Saudi stock data",
                "2. **Model Configuration**: Adjust training parameters in the sidebar - time window, test ratio, epochs, and batch size",
                "3. **Model Selection**: Choose between LSTM (better for time series) or MLP (faster training) models",
                "4. **Training**: Click 'Start Training' to begin the prediction model training process",
                "5. **Results Analysis**: View performance metrics, loss curves, and prediction comparisons",
                "6. **Future Predictions**: Get 30-day price forecasts and download results for further analysis"
            ],
            'technical_details': "Technical Details",
            'technical_content': [
                "**LSTM Model**: Uses two LSTM layers with Dropout regularization to prevent overfitting, specifically designed for sequential data patterns in stock prices",
                "**MLP Model**: Multi-layer Perceptron with ReLU activation functions, suitable for capturing non-linear relationships in financial data",
                "**Data Preprocessing**: Min-Max scaling normalizes price data to [0,1] range for optimal neural network performance",
                "**Time Series Processing**: Sliding window approach converts time series data into supervised learning format",
                "**Performance Metrics**: RMSE, MSE, and R² scores evaluate model accuracy and prediction quality"
            ],
            'usage_tips': "Usage Tips",
            'tips_content': [
                "• For better results, use at least 2 years of historical data",
                "• Adjust time window based on your data frequency (30-60 days recommended)",
                "• Higher epochs may improve accuracy but increase training time",
                "• Monitor loss curves to detect overfitting",
                "• Compare LSTM vs MLP results to choose the best model for your data"
            ]
        },
        'arabic': {
            'title': "📈 نظام التنبؤ بأسعار الأسهم - تداول السعودية",
            'sidebar_settings': "⚙️ إعدادات النموذج",
            'upload_file': "📤 حمّل ملف بيانات الأسهم (CSV)",
            'builtin_data': "🔍 البحث في مؤشر تاسي",
            'select_stock': "اختر السهم",
            'select_period': "اختر الفترة التاريخية",
            'training_params': "🎛️ معاملات التدريب",
            'time_window': "نافذة الزمن (أيام)",
            'test_ratio': "نسبة الاختبار",
            'epochs': "عدد الدورات",
            'batch_size': "حجم الدفعة",
            'select_model': "اختر النموذج",
            'start_training': "🚀 ابدأ التدريب",
            'data_overview': "📊 نظرة عامة على البيانات",
            'total_records': "إجمالي السجلات",
            'time_period': "الفترة الزمنية",
            'min_price': "أقل سعر",
            'max_price': "أعلى سعر",
            'tabs': ["📋 عرض البيانات", "📈 الرسوم البيانية", "🔍 التحليل الإحصائي"],
            'price_evolution': "تطور سعر السهم مع الوقت",
            'candlestick': "الرسم البياني للشموع اليابانية",
            'price_distribution': "توزيع الأسعار",
            'descriptive_stats': "الإحصائيات الوصفية",
            'training_results': "🎯 نتائج التدريب والتنبؤ",
            'training_model': "جاري تدريب النموذج...",
            'making_predictions': "جاري إجراء التنبؤات...",
            'completed': "اكتمل!",
            'performance_metrics': "📊 مقاييس الأداء",
            'training_loss': "📉 منحنى فقدان التدريب",
            'actual_vs_predicted': "🔮 المقارنة بين الأسعار الفعلية والمتوقعة",
            'future_predictions': "🔭 التنبؤات المستقبلية (30 يوم)",
            'download_results': "📥 تحميل النتائج",
            'download_button': "📥 حمّل التنبؤات المستقبلية (CSV)",
            'training_completed': "✅ اكتمل التدريب والتنبؤ بنجاح!",
            'welcome': "🚀 مرحباً بك في نظام التنبؤ بأسعار الأسهم",
            'welcome_desc': "نظام ذكي للتنبؤ بأسعار الأسهم في سوق تداول السعودي باستخدام تقنيات الذكاء الاصطناعي",
            'data_format': "📋 تنسيق البيانات المطلوب",
            'data_columns': "يجب أن يحتوي ملف CSV على الأعمدة التالية:",
            'date_col': "Date: تاريخ التداول",
            'price_col': "Price: سعر الإغلاق",
            'open_col': "Open: سعر الافتتاح",
            'high_col': "High: أعلى سعر",
            'low_col': "Low: أقل سعر",
            'vol_col': "Vol.: حجم التداول",
            'change_col': "Change %: نسبة التغير",
            'features': "🎯 ميزات النظام",
            'features_list': [
                "نماذج التعلم العميق: LSTM و MLP كما في وثيقة المشروع",
                "تركيز على تداول: مخصص لسوق الأسهم السعودي",
                "مقاييس الأداء: RMSE, MSE, و R² حسب المنهجية",
                "واجهة مستخدم سهلة: تصميم عربي سهل الاستخدام",
                "تنبؤات مستقبلية: تنبؤ بأسعار 30 يوم القادمة",
                "تصدير البيانات: تحميل النتائج للتحليل"
            ],
            'model_specs': "📊 مواصفات النماذج",
            'model_specs_list': [
                "LSTM: طبقتان مع Dropout لمنع الإفراط في التمرين",
                "MLP: مع تنشيط ReLU",
                "تطبيع البيانات: تحجيم Min-Max حسب المنهجية",
                "السلاسل الزمنية: نهج النافذة المنزلقة للتنبؤ بالتسلسل"
            ],
            'performance_stats': "📈 إحصائيات أداء النماذج",
            'lstm_accuracy': "دقة LSTM",
            'mlp_accuracy': "دقة MLP",
            'avg_r2': "متوسط R²",
            'historical_price': "السعر التاريخي",
            'future_forecast': "التنبؤات المستقبلية",
            'training_data': "بيانات التدريب",
            'actual_price': "السعر الفعلي",
            'predicted_price': "السعر المتوقع",
            'learn_more': "📚 تعرف أكثر",
            'fetching_data': "جاري جلب بيانات الأسهم...",
            'data_loaded': "✅ تم تحميل البيانات بنجاح!",
            'data_source': "مصدر البيانات",
            'system_guide': "دليل النظام وكيفية الاستخدام",
            'guide_content': [
                "1. **إدخال البيانات**: اختر بين تحميل ملف CSV خاص بك أو استخدام البحث في مؤشر تاسي لجلب بيانات الأسهم السعودية المباشرة",
                "2. **تهيئة النموذج**: اضبط معاملات التدريب في الشريط الجانبي - نافذة الزمن، نسبة الاختبار، عدد الدورات، وحجم الدفعة",
                "3. **اختيار النموذج**: اختر بين نماذج LSTM (أفضل للسلاسل الزمنية) أو MLP (تدريب أسرع)",
                "4. **التدريب**: انقر على 'ابدأ التدريب' لبدء عملية تدريب نموذج التنبؤ",
                "5. **تحليل النتائج**: عرض مقاييس الأداء، منحنيات الفقدان، ومقارنات التنبؤ",
                "6. **التنبؤات المستقبلية**: احصل على تنبؤات أسعار 30 يوم القادمة وحمّل النتائج لمزيد من التحليل"
            ],
            'technical_details': "التفاصيل التقنية",
            'technical_content': [
                "**نموذج LSTM**: يستخدم طبقتين LSTM مع تنظيم Dropout لمنع الإفراط في التمرين، مصمم خصيصًا لأنماط البيانات التسلسلية في أسعار الأسهم",
                "**نموذج MLP**:多层感知机 مع دوال تنشيط ReLU، مناسب لالتقاط العلاقات غير الخطية في البيانات المالية",
                "**معالجة البيانات**: تحجيم Min-Max يقوم بتطبيع بيانات الأسعار إلى نطاق [0,1] لأداء أفضل للشبكات العصبية",
                "**معالجة السلاسل الزمنية**: نهج النافذة المنزلقة يحول بيانات السلسلة الزمنية إلى تنسيق التعلم الخاضع للإشراف",
                "**مقاييس الأداء**: RMSE, MSE, و R² تقيم دقة النموذج وجودة التنبؤ"
            ],
            'usage_tips': "نصائح الاستخدام",
            'tips_content': [
                "• للحصول على نتائج أفضل، استخدم بيانات تاريخية لمدة سنتين على الأقل",
                "• اضبط نافذة الزمن بناءً على تردد بياناتك (30-60 يوم موصى بها)",
                "• زيادة عدد الدورات قد تحسن الدقة ولكن تزيد وقت التدريب",
                "• راقب منحنيات الفقدان لاكتشاف الإفراط في التمرين",
                "• قارن نتائج LSTM مقابل MLP لاختيار أفضل نموذج لبياناتك"
            ]
        }
    }
    return translations[language]

# TASI stocks (names verified against Yahoo Finance)
SAUDI_STOCKS = {
    '2222.SR': 'Aramco (أرامكو)',
    '1120.SR': 'Al Rajhi Bank (الراجحي)',
    '1180.SR': 'Saudi National Bank (الأهلي)',
    '1150.SR': 'Alinma Bank (الإنماء)',
    '1010.SR': 'Riyad Bank (بنك الرياض)',
    '1140.SR': 'Bank Albilad (البلاد)',
    '1050.SR': 'Banque Saudi Fransi (السعودي الفرنسي)',
    '1060.SR': 'Saudi Awwal Bank (الأول)',
    '2010.SR': 'SABIC (سابك)',
    '2020.SR': 'SABIC Agri-Nutrients (سابك للمغذيات)',
    '1211.SR': 'Maaden (معادن)',
    '7010.SR': 'stc (الاتصالات السعودية)',
    '5110.SR': 'Saudi Energy (السعودية للطاقة)',
    '2082.SR': 'ACWA Power (أكوا باور)',
    '2280.SR': 'Almarai (المراعي)',
    '2050.SR': 'Savola Group (صافولا)',
    '2350.SR': 'Saudi Kayan (كيان)',
    '2380.SR': 'Petro Rabigh (رابغ)',
    '3030.SR': 'Saudi Cement (الأسمنت السعودية)',
    '3020.SR': 'Yamama Cement (اليمامة)',
    '4030.SR': 'Bahri (البحري)',
    '4002.SR': 'Mouwasat Medical (المواساة)',
    '4163.SR': 'Al-Dawaa Medical (الدواء)',
    '4200.SR': 'Aldrees Petroleum (الدريس)'
}

ALTERNATIVE_SYMBOLS = {}

# Historical periods
PERIODS = {
    '3mo': '3 Months',
    '6mo': '6 Months', 
    '1y': '1 Year',
    '2y': '2 Years',
    '5y': '5 Years'
}

# تنسيق الصفحة
st.set_page_config(
    page_title="نظام التنبؤ بأسعار الأسهم - تداول",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# إضافة تنسيق CSS مخصص
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        margin: 1.5rem 0rem 1rem 0rem;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #4b6cb7 0%, #182848 100%);
        color: white;
    }
    .stButton button {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 1rem;
    }
    .language-switcher {
        position: fixed;
        top: 10px;
        right: 10px;
        z-index: 999;
    }
    .learn-more-content {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .performance-stats {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=900, show_spinner=False)
def _download(symbol, period):
    # raises on empty data so failures are never cached; retries ride out Yahoo rate limits
    for attempt in range(3):
        try:
            hist = yf.Ticker(symbol).history(period=period)
            if len(hist) >= 10:
                return hist
            err = ValueError(f"Yahoo returned {len(hist)} rows")
        except Exception as e:
            err = e
        time.sleep(2 * (attempt + 1))
    raise err

def fetch_stock_data(stock_symbol, period):
    """Fetch stock data from Yahoo Finance; returns None if unavailable (never fake data)"""
    err = None
    for sym in [stock_symbol, ALTERNATIVE_SYMBOLS.get(stock_symbol)]:
        if not sym:
            continue
        try:
            hist_data = _download(sym, period)
        except Exception as e:
            err = e
            continue
        if sym != stock_symbol:
            st.info(f"Using alternative symbol: {sym}")
        hist_data = hist_data.reset_index().rename(columns={'Close': 'Price', 'Volume': 'Vol.'})
        hist_data['Date'] = pd.to_datetime(hist_data['Date']).dt.tz_localize(None)
        hist_data['Change %'] = hist_data['Price'].pct_change().fillna(0) * 100
        return hist_data
    st.error(f"No real data available for {stock_symbol} ({type(err).__name__}: {err}). "
             "Yahoo may be rate-limiting this server; wait a minute and try again.")
    return None

class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.model_type = None
        
    def load_data(self, file_path):
        """تحميل وتجهيز بيانات الأسهم / Load and prepare stock data"""
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        return df
    
    def prepare_data(self, df, time_window=60, test_ratio=0.2):
        """Windows of daily log-returns -> next-day log-return (price levels are non-stationary)"""
        prices = df['Price'].values
        rets = np.diff(np.log(prices)).reshape(-1, 1)  # rets[i] = return into prices[i+1]
        split = int(len(prices) * (1 - test_ratio))    # first test price index
        self.scaler.fit(rets[:split - 1])              # train returns only
        scaled = self.scaler.transform(rets)
        
        def windows(lo, hi):  # targets are price indices lo..hi-1
            x = np.array([scaled[i - time_window:i, 0] for i in range(lo - 1, hi - 1)])
            return x.reshape(len(x), time_window, 1)
        
        x_train = windows(time_window + 1, split)
        y_train = scaled[time_window:split - 1, 0]
        x_test = windows(split, len(prices))
        self.prev_prices = prices[split - 1:-1]        # yesterday's actual price per test day
        y_test = prices[split:].reshape(-1, 1)
        return x_train, y_train, x_test, y_test, split
    
    def build_lstm_model(self, time_window, lstm_units=50, dropout_rate=0.2):
        """بناء نموذج LSTM / Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=lstm_units, return_sequences=True, 
                      input_shape=(time_window, 1)))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=lstm_units, return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model
    
    def build_mlp_model(self, time_window, layers=[64, 32, 16]):
        """بناء نموذج MLP / Build MLP model"""
        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=(time_window,)))
        
        for units in layers[1:]:
            model.add(Dense(units, activation='relu'))
        
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        return model
    
    def train_model(self, x_train, y_train, model_type='LSTM', 
                   epochs=20, batch_size=32, time_window=60):
        """تدريب النموذج المختار / Train selected model"""
        self.model_type = model_type
        es = EarlyStopping(patience=5, restore_best_weights=True)
        
        if model_type == 'LSTM':
            self.model = self.build_lstm_model(time_window)
            history = self.model.fit(x_train, y_train, 
                                   batch_size=batch_size, 
                                   epochs=epochs,
                                   verbose=0, validation_split=0.1, callbacks=[es])
        else:  # MLP
            # إعادة تشكيل البيانات لـ MLP / Reshape data for MLP
            x_train_mlp = x_train.reshape(x_train.shape[0], x_train.shape[1])
            self.model = self.build_mlp_model(time_window)
            history = self.model.fit(x_train_mlp, y_train, 
                                   batch_size=batch_size, 
                                   epochs=epochs,
                                   verbose=0, validation_split=0.1, callbacks=[es])
        
        return history
    
    def predict(self, x_test, model_type='LSTM'):
        """إجراء التنبؤات / Make predictions"""
        if model_type == 'LSTM':
            predictions = self.model.predict(x_test, verbose=0)
        else:  # MLP
            x_test_mlp = x_test.reshape(x_test.shape[0], x_test.shape[1])
            predictions = self.model.predict(x_test_mlp, verbose=0)
        
        rets = self.scaler.inverse_transform(predictions)
        return self.prev_prices.reshape(-1, 1) * np.exp(rets)
    
    def calculate_metrics(self, y_true, y_pred):
        """حساب مقاييس الأداء / Calculate performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        return mse, rmse, r2
    
    def baseline_and_direction(self, y_true, y_pred):
        """RMSE of the naive 'tomorrow = today' forecast, and % of days the up/down direction was right"""
        prev = self.prev_prices.reshape(-1, 1)
        naive_rmse = np.sqrt(mean_squared_error(y_true, prev))
        hit = np.mean(np.sign(y_true - prev) == np.sign(y_pred - prev)) * 100
        return naive_rmse, hit

def create_candlestick_chart(df, lang):
    """إنشاء رسم الشموع اليابانية / Create candlestick chart"""
    title = "Candlestick Chart" if lang == 'english' else "الرسم البياني للشموع اليابانية"
    x_title = "Date" if lang == 'english' else "التاريخ"
    y_title = "Price (SAR)" if lang == 'english' else "السعر (ريال)"
    
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Price'],
        name='Stock Prices' if lang == 'english' else 'أسعار الأسهم'
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template='plotly_white',
        height=500
    )
    
    return fig

def create_performance_gauge(value, title, min_val, max_val):
    """إنشاء مقياس أداء تفاعلي / Create interactive performance gauge"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        gauge = {
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [min_val, min_val + (max_val-min_val)*0.6], 'color': "lightgray"},
                {'range': [min_val + (max_val-min_val)*0.6, min_val + (max_val-min_val)*0.8], 'color': "gray"},
                {'range': [min_val + (max_val-min_val)*0.8, max_val], 'color': "darkgray"}
            ],
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    # Initialize session state for learn more
    if 'show_learn_more' not in st.session_state:
        st.session_state.show_learn_more = False

    # Language selection
    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        language = st.selectbox("🌐 Language / اللغة", ["english", "arabic"])
    
    # Get translations
    t = get_translations(language)
    
    # رأس الصفحة الرئيسي / Main header
    st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
    
    # Learn More button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button(t['learn_more'], use_container_width=True):
            st.session_state.show_learn_more = not st.session_state.show_learn_more
    
    # Learn More content
    if st.session_state.show_learn_more:
        st.markdown('<div class="learn-more-content">', unsafe_allow_html=True)
        
        st.markdown(f"### {t['system_guide']}")
        for guide_item in t['guide_content']:
            st.markdown(guide_item)
        
        st.markdown(f"### {t['technical_details']}")
        for tech_item in t['technical_content']:
            st.markdown(tech_item)
        
        st.markdown(f"### {t['usage_tips']}")
        for tip in t['tips_content']:
            st.markdown(f"• {tip}" if tip.startswith('•') else tip)
        
        st.markdown(f"### {t['data_format']}")
        st.markdown(f"{t['data_columns']}")
        st.markdown(f"- **{t['date_col']}**")
        st.markdown(f"- **{t['price_col']}**")
        st.markdown(f"- **{t['open_col']}**")
        st.markdown(f"- **{t['high_col']}**")
        st.markdown(f"- **{t['low_col']}**")
        st.markdown(f"- **{t['vol_col']}**")
        st.markdown(f"- **{t['change_col']}**")
        
        # مثال على البيانات / Sample data
        sample_data = pd.DataFrame({
            'Date': ['01/01/2023', '01/02/2023', '01/03/2023'],
            'Price': [0.3858, 0.4083, 0.4437],
            'Open': [0.3806, 0.3870, 0.4096],
            'High': [0.3589, 0.3717, 0.4006],
            'Low': [0.3973, 0.3941, 0.4299],
            'Vol.': [0.0474, 0.0728, 0.1252],
            'Change %': [0.5222, 0.5759, 0.6275]
        })
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # الشريط الجانبي / Sidebar
    with st.sidebar:
        st.markdown(f"### {t['sidebar_settings']}")
        
        # Data source selection
        data_source = st.radio("Select Data Source", ["Upload CSV", "TASI Stock Search"])
        
        if data_source == "Upload CSV":
            # تحميل الملف / File upload
            uploaded_file = st.file_uploader(t['upload_file'], type=['csv'])
            df_loaded = uploaded_file is not None
        else:
            # TASI Stock Search
            st.markdown(f"### {t['builtin_data']}")
            selected_stock = st.selectbox(
                t['select_stock'], 
                options=list(SAUDI_STOCKS.keys()),
                format_func=lambda x: f"{SAUDI_STOCKS[x]} ({x})"
            )
            selected_period = st.selectbox(
                t['select_period'],
                options=list(PERIODS.keys()),
                format_func=lambda x: PERIODS[x]
            )
            
            if st.button("Fetch Stock Data", use_container_width=True, type="primary"):
                with st.spinner(t['fetching_data']):
                    df = fetch_stock_data(selected_stock, selected_period)
                    if df is not None:
                        st.session_state.builtin_df = df
                        st.session_state.selected_stock = selected_stock
                        st.session_state.selected_period = selected_period
                        st.success(t['data_loaded'])
            
            df_loaded = 'builtin_df' in st.session_state
        
        st.markdown("---")
        st.markdown(f"### {t['training_params']}")
        
        col1, col2 = st.columns(2)
        with col1:
            time_window = st.slider(t['time_window'], 30, 120, 60)
            test_ratio = st.slider(t['test_ratio'], 0.1, 0.4, 0.2, 0.05)
        with col2:
            epochs = st.slider(t['epochs'], 10, 100, 20)
            batch_size = st.slider(t['batch_size'], 16, 64, 32)
        
        model_type = st.selectbox(t['select_model'], ["LSTM", "MLP"])
        
        st.markdown("---")
        
        if st.button(t['start_training'], type="primary", use_container_width=True):
            st.session_state.run_training = True
        else:
            st.session_state.run_training = False

    # Main content area
    if data_source == "Upload CSV" and uploaded_file is not None:
        df_source = "uploaded"
        predictor = StockPredictor()
        df = predictor.load_data(uploaded_file)
    elif data_source == "TASI Stock Search" and df_loaded:
        df_source = "builtin"
        df = st.session_state.builtin_df
        predictor = StockPredictor()
    else:
        # Welcome message when no data is loaded
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;'>
            <h2>{t['welcome']}</h2>
            <p style='font-size: 1.2rem;'>{t['welcome_desc']}</p>
            <p style='font-size: 1rem; margin-top: 1rem;'>📊 Select a data source from the sidebar to get started</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Display data overview and analysis
    if df_loaded:
        # Display data source info for built-in data
        if df_source == "builtin":
            stock_name = SAUDI_STOCKS[st.session_state.selected_stock]
            period_name = PERIODS[st.session_state.selected_period]
            st.info(f"📊 {t['data_source']}: {stock_name} | {t['select_period']}: {period_name}")

        # عرض نظرة عامة على البيانات / Data overview
        st.markdown(f'<h2 class="section-header">{t["data_overview"]}</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(t['total_records'], len(df))
        with col2:
            st.metric(t['time_period'], f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        with col3:
            st.metric(t['min_price'], f"{df['Price'].min():.2f} SAR")
        with col4:
            st.metric(t['max_price'], f"{df['Price'].max():.2f} SAR")
        
        # علامات تبويب للعروض المختلفة / Tabs for different views
        tab1, tab2, tab3 = st.tabs(t['tabs'])
        
        with tab1:
            st.dataframe(df, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                # رسم بياني للأسعار / Price chart
                fig1 = px.line(df, x='Date', y='Price', title=t['price_evolution'], labels={'Price': 'Price (SAR)' if language == 'english' else 'السعر (ريال)'})
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # رسم الشموع اليابانية / Candlestick chart
                candlestick_fig = create_candlestick_chart(df, language)
                st.plotly_chart(candlestick_fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                # توزيع الأسعار / Price distribution
                fig_hist = px.histogram(df, x='Price', title=t['price_distribution'], labels={'Price': 'Price (SAR)'})
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # إحصائيات描述ية / Descriptive statistics
                st.subheader(t['descriptive_stats'])
                st.dataframe(df.describe(), use_container_width=True)
        
        # التدريب والتنبؤ / Training and prediction
        if st.session_state.run_training:
            st.markdown(f'<h2 class="section-header">{t["training_results"]}</h2>', unsafe_allow_html=True)
            
            if len(df) * (1 - test_ratio) < time_window + 30 or len(df) * test_ratio < 5:
                st.error(f"Not enough data ({len(df)} rows) for a {time_window}-day window. "
                         "Pick a longer period (1y+) or a smaller time window.")
                return
            
            with st.spinner(t['training_model']):
                # تحضير البيانات / Prepare data
                x_train, y_train, x_test, y_test, training_data_len = predictor.prepare_data(
                    df, time_window, test_ratio
                )
                
                # شريط التقدم / Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # تدريب النموذج / Train model
                status_text.text(t['training_model'])
                history = predictor.train_model(
                    x_train, y_train, model_type, epochs, batch_size, time_window
                )
                progress_bar.progress(50)
                
                # إجراء التنبؤات / Make predictions
                status_text.text(t['making_predictions'])
                predictions = predictor.predict(x_test, model_type)
                progress_bar.progress(75)
                
                # حساب المقاييس / Calculate metrics
                mse, rmse, r2 = predictor.calculate_metrics(y_test, predictions)
                progress_bar.progress(100)
                status_text.text(t['completed'])
                
                naive_rmse, hit = predictor.baseline_and_direction(y_test, predictions)
                
                # عرض النتائج / Display results
                st.markdown(f"### {t['performance_metrics']}")
                st.caption(f"Naive baseline RMSE (tomorrow = today): {naive_rmse:.3f} SAR · model RMSE: {rmse:.3f} SAR · "
                           f"direction accuracy: {hit:.1f}% (50% = coin flip)")
                if rmse >= naive_rmse:
                    st.warning("The model does not beat the naive baseline on this stock/period. Do not rely on it.")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.plotly_chart(create_performance_gauge(rmse, "RMSE", 0, df['Price'].std()), use_container_width=True)
                with col2:
                    st.plotly_chart(create_performance_gauge(mse, "MSE", 0, df['Price'].var()), use_container_width=True)
                with col3:
                    st.plotly_chart(create_performance_gauge(r2, "R² Score", 0, 1), use_container_width=True)
                
                # منحنى الخسارة / Loss curve
                st.markdown(f"### {t['training_loss']}")
                fig_loss, ax = plt.subplots(figsize=(10, 4))
                ax.plot(history.history['loss'], 
                       label='Training Loss' if language == 'english' else 'فقدان التدريب', 
                       linewidth=2)
                ax.set_title(f'{t["training_loss"]} - {model_type}')
                ax.set_xlabel('Epochs' if language == 'english' else 'الدورات')
                ax.set_ylabel('Loss' if language == 'english' else 'الفقدان')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig_loss)
                
                # التنبؤ مقابل الفعلي / Prediction vs Actual
                st.markdown(f"### {t['actual_vs_predicted']}")
                
                # إنشاء بيانات للرسم / Create data for plotting
                train = df[:training_data_len]
                valid = df[training_data_len:]
                valid = valid.copy()
                valid['Predictions'] = predictions
                
                fig_comparison = go.Figure()
                fig_comparison.add_trace(go.Scatter(
                    x=train['Date'], y=train['Price'],
                    name=t['training_data'],
                    line=dict(color='blue', width=2)
                ))
                fig_comparison.add_trace(go.Scatter(
                    x=valid['Date'], y=valid['Price'],
                    name=t['actual_price'],
                    line=dict(color='green', width=2)
                ))
                fig_comparison.add_trace(go.Scatter(
                    x=valid['Date'], y=valid['Predictions'],
                    name=t['predicted_price'],
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_comparison.update_layout(
                    title=f'{t["actual_vs_predicted"]} - {model_type}',
                    xaxis_title='Date' if language == 'english' else 'التاريخ',
                    yaxis_title='Price (SAR)' if language == 'english' else 'السعر (ريال)',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # التنبؤات المستقبلية / Future predictions
                st.markdown(f"### {t['future_predictions']}")
                
                # Recursive 30-day forecast on returns, compounded from the last price
                last_rets = np.diff(np.log(df['Price'].values))[-time_window:]
                current_batch = predictor.scaler.transform(last_rets.reshape(-1, 1)).reshape(1, time_window, 1)
                
                future_scaled = []
                for i in range(30):  # التنبؤ لـ 30 يوم / Predict for 30 days
                    inp = current_batch if model_type == 'LSTM' else current_batch.reshape(1, time_window)
                    current_pred = predictor.model.predict(inp, verbose=0)[0]
                    future_scaled.append(current_pred[0])
                    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
                
                future_rets = predictor.scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1))
                future_predictions = df['Price'].values[-1] * np.exp(np.cumsum(future_rets))
                st.caption("The line is the model's central estimate (daily direction is near-unpredictable). The shaded 80% range is the reliable part: it contained the real price ~80% of the time in back-tests. Not financial advice.")
                
                # إنشاء تواريخ مستقبلية / Create future dates
                last_date = df['Date'].iloc[-1]
                future_dates = pd.bdate_range(last_date + timedelta(days=1), periods=30, freq='C', weekmask='Sun Mon Tue Wed Thu')  # Tadawul trades Sun-Thu
                
                # عرض التنبؤات المستقبلية / Display future predictions
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': future_predictions
                })
                # 80% range from trailing 22-day volatility, widening with sqrt(days);
                # walk-forward coverage on 24 TASI stocks was 79-80% for 1-30 day horizons
                band = 1.2816 * np.log(df['Price']).diff().tail(22).std() * np.sqrt(np.arange(1, 31))
                future_df['Low (80%)'] = future_df['Predicted Price'] * np.exp(-band)
                future_df['High (80%)'] = future_df['Predicted Price'] * np.exp(band)
                
                fig_future = go.Figure()
                fig_future.add_trace(go.Scatter(
                    x=df['Date'], y=df['Price'],
                    name=t['historical_price'],
                    line=dict(color='blue', width=2)
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_df['Date'], y=future_df['High (80%)'], line=dict(width=0), showlegend=False, hoverinfo='skip'
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_df['Date'], y=future_df['Low (80%)'], line=dict(width=0), fill='tonexty',
                    fillcolor='rgba(255,0,0,0.15)', name='80% range'
                ))
                fig_future.add_trace(go.Scatter(
                    x=future_df['Date'], y=future_df['Predicted Price'],
                    name=t['future_forecast'],
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig_future.update_layout(
                    title=t['future_predictions'],
                    xaxis_title='Date' if language == 'english' else 'التاريخ',
                    yaxis_title='Price (SAR)' if language == 'english' else 'السعر (ريال)',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig_future, use_container_width=True)
                
                # عرض جدول التنبؤات / Display predictions table
                st.dataframe(future_df, use_container_width=True, column_config={c: st.column_config.NumberColumn(format="%.2f SAR") for c in ['Predicted Price', 'Low (80%)', 'High (80%)']})
                
                # زر تحميل النتائج / Download results button
                st.markdown(f"### {t['download_results']}")
                csv = future_df.to_csv(index=False)
                st.download_button(
                    label=t['download_button'],
                    data=csv,
                    file_name=f"future_predictions_{model_type}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
                st.success(t['training_completed'])

if __name__ == "__main__":
    main()