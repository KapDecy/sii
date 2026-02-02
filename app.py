import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import time

import pandas as pd
import datetime
import sqlite3

# --- Инициализация БД ---
def init_db():
    conn = sqlite3.connect('parking_history.db')
    c = conn.cursor()
    # Создаем таблицу, если нет
    c.execute('''CREATE TABLE IF NOT EXISTS history
                 (timestamp TEXT, total_objects INTEGER, 
                  bicycles INTEGER, cars INTEGER)''')
    conn.commit()
    conn.close()

def save_to_db(total, bicycles, cars):
    conn = sqlite3.connect('parking_history.db')
    c = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO history VALUES (?, ?, ?, ?)", (now, total, bicycles, cars))
    conn.commit()
    conn.close()

# Запускаем один раз при старте
init_db()

# --- Настройки страницы ---
st.set_page_config(page_title="Учет транспорта", layout="wide")

st.title("🚲/🚗 Система учета транспорта на парковке")
st.write("Загрузка модели YOLOv8 для детекции и подсчета объектов.")

# --- Боковая панель ---
st.sidebar.header("Настройки")
confidence = st.sidebar.slider("Порог уверенности (Confidence)", 0.0, 1.0, 0.25)

# --- Загрузка модели ---
# Используем кэширование
@st.cache_resource
def load_model():
    model = YOLO('yolov8m.pt') 
    return model

try:
    model = load_model()
    st.sidebar.success("Модель успешно загружена!")
except Exception as e:
    st.sidebar.error(f"Ошибка загрузки модели: {e}")

# --- Выбор режима ---
source_type = st.sidebar.radio("Выберите источник:", ["Изображение", "Видео"])

# --- Логика для ИЗОБРАЖЕНИЙ ---
if source_type == "Изображение":
    uploaded_file = st.sidebar.file_uploader("Загрузить фото", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Отображаем оригинал
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Исходное изображение", width='stretch')
            
        # Кнопка запуска
        if st.sidebar.button("Распознать"):
            # Инференс
            results = model.predict(image, conf=confidence)
            
            # Отрисовка результата
            res_plotted = results[0].plot() # Рисует боксы на картинке
            
            # Подсчет статистики
            obj_count = len(results[0].boxes)
            
            with col2:
                st.image(res_plotted, caption="Результат обработки", width='stretch')
                
            # Вывод статистики
            st.metric(label="Обнаружено объектов", value=obj_count)
            
            st.success(f"Обработка завершена. Найдено: {obj_count}")
            
            cls_list = results[0].boxes.cls.cpu().numpy() # Получаем ID найденных классов
            num_bikes = int((cls_list == 1).sum())        # ID 1 - велосипед
            num_cars = int((cls_list == 2).sum())         # ID 2 - машина

            save_to_db(obj_count, num_bikes, num_cars) 

# --- Логика для ВИДЕО ---
elif source_type == "Видео":
    uploaded_video = st.sidebar.file_uploader("Загрузить видео", type=['mp4', 'avi', 'mov'])
    
    # 1. Инициализируем хранилище статистики в сессии, если его нет
    if 'video_stats' not in st.session_state:
        st.session_state['video_stats'] = []

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)
        
        st_frame = st.empty()
        st_stat = st.empty()
        
        # Это работает как ПАУЗА / СТОП
        run_processing = st.sidebar.checkbox("Запустить обработку", value=False)
        
        if run_processing:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.sidebar.warning("Видео закончилось")
                    break
                
                results = model.predict(frame, conf=confidence, classes=[1, 2, 3, 5, 7], verbose=False)
                
                res_frame = results[0].plot()
                obj_count = len(results[0].boxes)
                
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Считаем детали (велосипеды/машины)
                cls_list = results[0].boxes.cls.cpu().numpy()
                n_bikes = int((cls_list == 1).sum()) # 1 - велосипед
                n_cars = int((cls_list == 2).sum())  # 2 - машина
                
                new_row = {"Время": timestamp, "Всего": obj_count, "Велосипеды": n_bikes, "Машины": n_cars}
                st.session_state['video_stats'].append(new_row)
                
                # Отрисовка
                res_frame_rgb = cv2.cvtColor(res_frame, cv2.COLOR_BGR2RGB)
                st_frame.image(res_frame_rgb, channels="RGB", use_container_width=True)
                st_stat.metric("Объектов в кадре", obj_count)
        
        cap.release()

    # 3. ВЫВОД СТАТИСТИКИ (работает, даже если обработка остановлена)
    st.divider()
    st.subheader("📊 Итоговая статистика")
    
    if len(st.session_state['video_stats']) > 0:
        # Превращаем список из сессии в DataFrame
        df_stats = pd.DataFrame(st.session_state['video_stats'])
        
        # Показываем таблицу
        st.dataframe(df_stats, use_container_width=True)
        
        st.line_chart(df_stats, x="Время", y=["Велосипеды", "Машины"])
        
        if st.button("Очистить историю"):
            st.session_state['video_stats'] = []
            st.rerun()
    else:
        st.info("Запустите обработку, чтобы собрать данные.")
        
st.divider()
st.subheader("История и Отчетность")

# Кнопка выгрузки
if st.button("Показать/Скачать историю"):
    conn = sqlite3.connect('parking_history.db')
    df = pd.read_sql_query("SELECT * FROM history ORDER BY timestamp DESC", conn)
    conn.close()
    
    st.dataframe(df)
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать отчет (CSV/Excel)",
        data=csv,
        file_name='parking_report.csv',
        mime='text/csv',
    )
