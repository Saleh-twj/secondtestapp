import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------
# 📚 Texts & Translations
# ------------------------------
texts = {
    "Arabic": {
        'data_preview': '🔍 معاينة البيانات',
        'data_summary': '📋 ملخص البيانات',
        'total_records': 'إجمالي السجلات',
        'date_range': 'النطاق الزمني',
        'total_return': 'إجمالي العائد',
        'preprocessing': '🔄 معالجة البيانات',
        'model_training': '🤖 تدريب النموذج الذكي',
        'performance': '📊 أداء النموذج',
        'analytics': '📈 التحليلات المتقدمة',
        'price_prediction': '📊 تنبؤ الأسعار',
        'market_analysis': '🕯️ تحليل السوق',
        'loss_metrics': '📉 مقاييس الخسارة',
        'future_forecast': '🔮 التنبؤ المستقبلي',
        'export': '💾 تصدير النتائج',
        'download_predictions': '📥 تحميل التنبؤات',
        'model_arch_download': '📋 بنية النموذج',
        'training_report': '📄 تقرير التدريب',
        'welcome_title': '🚀 مرحباً بكم في منصة توقع الأسهم الذكية',
        'welcome_text': 'اختر مصدر البيانات أعلاه للبدء في التنبؤ بالأسهم باستخدام الذكاء الاصطناعي. قم بتحميل بيانات أسهم تاسي تلقائياً أو ارفع بيانات السوق التاريخية الخاصة بك.',
        'how_it_works': '📊 كيف يعمل:',
        'select_stock': 'اختيار السهم',
        'select_stock_desc': 'اختر من شركات تاسي أو ارفع بياناتك',
        'ai_training': 'التدريب الذكي',
        'ai_training_desc': 'الشبكات العصبية تتعلم أنماط السوق',
        'get_predictions': 'الحصول على تنبؤات',
        'get_predictions_desc': 'احصل على تنبؤات دقيقة للأسعار',
        'footer': '© 2024 مركز توقع الأسهم | مبنى باستخدام Streamlit & TensorFlow',
        'learn_more': '📚 اعرف المزيد',
        'learn_more_title': '📚 كيف يعمل نظام توقع الأسهم هذا',
        'learn_more_content': '<div class="learn-more-content" style="text-align: right; direction: rtl;">...full content...</div>'
    }
}

# ------------------------------
# 🌐 Language Selector
# ------------------------------
st.session_state.language = st.selectbox("🌐 Language / اللغة", ["English", "Arabic"], key="lang_selector")
t = texts[st.session_state.language]

# ------------------------------
# 🔧 Config Panel
# ------------------------------
with st.expander("Configuration", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_window = st.slider("Time Window", 5, 100, 60)
        test_ratio = st.slider("Test Ratio", 0.1, 0.5, 0.2)
    with col2:
        model_choice = st.selectbox("Model Type", ["LSTM", "MLP", "Hybrid"])
        layers = st.slider("Layers", 1, 5, 3)
    with col3:
        epochs = st.slider("Epochs", 10, 500, 100)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    with col4:
        forecast_days = st.slider("Forecast Days", 1, 30, 7)
        confidence_level = st.slider("Confidence Level", 0.8, 0.99, 0.95)

# ------------------------------
# 🏢 TASI Stock Search & Data Upload
# ------------------------------
tasi_stocks = {"Al Rajhi Bank": "1120.SR", "SABIC": "2010.SR"}  # example

data_source = st.radio("Data Source", ["Upload File", "TASI Search"], horizontal=True)
df = None

if data_source == "TASI Search":
    selected_stock = st.selectbox("Select Stock", options=list(tasi_stocks.keys()))
    period = st.selectbox("Historical Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    load_data = st.button("Load Data")
    if load_data:
        symbol = tasi_stocks[selected_stock]
        df = yf.download(symbol, period=period).reset_index()
else:
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    if uploaded_file is not None:
        if uploaded_file.name.endswith("xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

# ------------------------------
# 🔄 Normalization Fix
# ------------------------------
def normalize_stock_data(df):
    df = df.copy()
    numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    scaler = MinMaxScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

if df is not None and 'Close' in df.columns:
    try:
        df = normalize_stock_data(df)
        st.success("✅ Data normalized successfully!")
    except Exception as e:
        st.error(f"❌ Normalization failed: {e}")

# ------------------------------
# 🤖 Model Training
# ------------------------------
if df is not None and 'Close' in df.columns:
    close_prices = df['Close'].values.reshape(-1,1)
    scaler_close = MinMaxScaler()
    scaled_close = scaler_close.fit_transform(close_prices)

    X, y = [], []
    for i in range(time_window, len(scaled_close)):
        X.append(scaled_close[i-time_window:i,0])
        y.append(scaled_close[i,0])
    X, y = np.array(X), np.array(y)

    if model_choice == "LSTM":
        X = X.reshape((X.shape[0], X.shape[1], 1))

    split = int(len(X)*(1-test_ratio))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential()
    if model_choice == "MLP":
        model.add(Dense(256, activation="relu", input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.3))
        for i in range(layers-1):
            model.add(Dense(128//(i+1), activation="relu"))
            model.add(Dropout(0.2))
        model.add(Dense(1))
    elif model_choice == "LSTM":
        model.add(LSTM(150, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.3))
        for i in range(layers-1):
            model.add(LSTM(100//(i+1), return_sequences=(i<layers-2)))
            model.add(Dropout(0.2))
        model.add(Dense(1))
    else:  # Hybrid
        model.add(LSTM(100, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.3))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                        verbose=0)

# ------------------------------
# 📊 Predictions & Evaluation
# ------------------------------
    y_pred = model.predict(X_test)
    y_test_inv = scaler_close.inverse_transform(y_test.reshape(-1,1))
    y_pred_inv = scaler_close.inverse_transform(y_pred)

    mse = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv)/y_test_inv))*100

# ------------------------------
# 💾 Export Results Example
# ------------------------------
    pred_df = pd.DataFrame({
        'Actual': y_test_inv.flatten(),
        'Predicted': y_pred_inv.flatten(),
        'Abs_Error': np.abs(y_test_inv.flatten()-y_pred_inv.flatten())
    })
    st.download_button("Download Predictions", pred_df.to_csv(index=False), "predictions.csv")