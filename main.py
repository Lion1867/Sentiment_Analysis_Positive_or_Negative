import streamlit as st
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import docx
import textract
import PyPDF2
import base64
import requests
from io import BytesIO
nltk.download('punkt')
nltk.download('wordnet')

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Функция для лемматизации текста
def lemmatize_text(text):
    tokens = word_tokenize(text)  # Токенизация текста
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Лемматизация токенов
    return ' '.join(lemmatized_tokens)

# Загрузка обученной модели
response = BytesIO(requests.get('https://raw.githubusercontent.com/Lion1867/Sentiment_Analysis_Positive_or_Negative/main/model_sentiment_analysis.joblib').content)
loaded_clf = joblib.load(response)

# Загрузка CountVectorizer
loaded_vectorizer = joblib.load('vectorizer_sentiment_analysis.joblib')

# Функция для чтения текста из файла
def read_text_from_file(file):
    if file.type == 'text/plain':
        text = file.read().decode('utf-8')
    elif file.type == 'application/msword' or file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file)
        text = '\n'.join([p.text for p in doc.paragraphs])
    elif file.type == 'application/pdf':
        pdf_file = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_file.pages:
            text += page.extract_text()
    else:
        text = textract.process(file).decode('utf-8')
    return text

# Функция для предсказания тональности отзыва
def predict_sentiment(text):
    new_text_lemmatized = lemmatize_text(text)
    new_text_vectorized = loaded_vectorizer.transform([new_text_lemmatized])
    predicted_class = loaded_clf.predict(new_text_vectorized)
    predicted_proba = loaded_clf.predict_proba(new_text_vectorized)
    return predicted_class[0], predicted_proba[0]

# Картинка сверху
def add_top_image():
    with open("3_kino.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    top_image = f'''
    <style>
    .top-image {{
        background-image: url(data:image/jpg;base64,{encoded_string});
        background-size: cover;
        height: 200px;
        width: 100%;
        position: relative;
    }}
    </style>
    <div class="top-image"></div>
    '''
    st.markdown(top_image, unsafe_allow_html=True)

# Функция для добавления CSS-стилей
def add_css(css_code):
    st.markdown(f"<style>{css_code}</style>", unsafe_allow_html=True)

# Добавляем легкий розовый фон ко всему контенту на странице
add_css("""
    .main .block-container {
        background-color: #FFFFFF;
    }
    .main {
        background-color: #FFF2F2;
    }
""")

# Заголовок приложения
add_top_image()
st.markdown("<h1 style='text-align: center; color: black;'>Анализ тональности отзывов <br>на английском языке о фильмах</h1>", unsafe_allow_html=True)

# Ввод текста или загрузка файла
input_type = st.radio('Выберите тип ввода', ['Текст', 'Файл'])

if input_type == 'Текст':
    new_text = st.text_area('Введите отзыв на АНГЛИЙСКОМ языке', max_chars=1000)
    if st.button('Предсказать тональность'):
        if len(new_text) > 0:
            predicted_class, predicted_proba = predict_sentiment(new_text)

            # Вывод соответствующего заголовка и картинки
            if predicted_class == 1:
                st.subheader("ПОЛОЖИТЕЛЬНЫЙ ОТЗЫВ")
                st.write("Вероятность положительного класса:", predicted_proba[1])
                st.write("Вероятность отрицательного класса:", predicted_proba[0])
                st.image("1_pos.jpg", use_column_width=True)
            else:
                st.subheader("ОТРИЦАТЕЛЬНЫЙ ОТЗЫВ")
                st.write("Вероятность положительного класса:", predicted_proba[1])
                st.write("Вероятность отрицательного класса:", predicted_proba[0])
                st.image("2_neg.jpg", use_column_width=True)
        else:
            st.warning('Пожалуйста, введите отзыв.')
else:
    file_uploader = st.file_uploader('Загрузите файл (Word, TXT или PDF формат)', type=['txt', 'docx', 'pdf'])
    if file_uploader:
        if file_uploader.size > 1000000:
            st.warning('Размер файла превышает лимит в 1 МБ.')
        else:
            text = read_text_from_file(file_uploader)
            if st.button('Предсказать тональность'):
                if len(text) > 0:
                    predicted_class, predicted_proba = predict_sentiment(text)
                    # Вывод соответствующего заголовка и картинки
                    if predicted_class == 1:
                        st.subheader("ПОЛОЖИТЕЛЬНЫЙ")
                        st.write("Вероятность положительного класса:", predicted_proba[1])
                        st.write("Вероятность отрицательного класса:", predicted_proba[0])
                        st.image("1_pos.jpg", use_column_width=True)
                    else:
                        st.subheader("ОТРИЦАТЕЛЬНЫЙ")
                        st.write("Вероятность положительного класса:", predicted_proba[1])
                        st.write("Вероятность отрицательного класса:", predicted_proba[0])
                        st.image("2_neg.jpg", use_column_width=True)
                else:
                    st.warning('Файл не содержит текста.')

# Закрывающий тег для вложенного контейнера
st.markdown("""</div>""", unsafe_allow_html=True)
