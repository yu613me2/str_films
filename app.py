import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Загрузка модели
model = load_model('my_model.keras')

# Загрузка токенизатора
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Функция для очистки текста
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яА-ЯёЁ\s]', '', text)  # Оставляем только кириллицу и пробелы
    return text.strip()

# Функция для предсказания класса
def predict_class(text, model, tokenizer, max_length):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_class = 1 if prediction[0][0] > 0.5 else 0  # 1 - Bad, 0 - Good
    return 'Bad' if predicted_class == 1 else 'Good', prediction[0][0]

# Streamlit интерфейс
st.title("🎬 Классификация отзывов о фильмах")
st.write("Введите текст отзыва:")

text_input = st.text_area("Отзыв", height=150)

if st.button("Предсказать"):
    if text_input:
        predicted_class, confidence = predict_class(text_input, model, tokenizer, max_length=100)
        st.write(f'**Предсказанный класс:** {predicted_class}')
        st.write(f'**Уверенность модели:** {confidence:.2f}')
        
        # Визуализация метрик
        if predicted_class == 'Bad':
            st.markdown("<h3 style='color: red;'>Отзыв негативный!</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color: green;'>Отзыв положительный!</h3>", unsafe_allow_html=True)
    else:
        st.write("Пожалуйста, введите текст отзыва.")
