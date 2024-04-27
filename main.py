import streamlit as st
import joblib
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import docx
import textract

# Инициализация лемматизатора
lemmatizer = WordNetLemmatizer()

# Функция для лемматизации текста
def lemmatize_text(text):
    tokens = word_tokenize(text)  # Токенизация текста
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Лемматизация токенов
    return ' '.join(lemmatized_tokens)

# Загрузка обученной модели
loaded_clf = joblib.load('model_sentiment_analysis.joblib')

# Загрузка CountVectorizer
loaded_vectorizer = joblib.load('vectorizer_sentiment_analysis.joblib')

# Функция для чтения текста из файла
def read_text_from_file(file):
    if file.name.endswith('.txt'):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
    elif file.name.endswith('.docx'):
        doc = docx.Document(file)
        text = '\n'.join([p.text for p in doc.paragraphs])
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

# Заголовок приложения
st.title('Movie Review Sentiment Analysis')

# Ввод текста или загрузка файла
input_type = st.radio('Choose input type', ['Text', 'File'])
if input_type == 'Text':
    new_text = st.text_area('Enter a new review', max_chars=1000)
    if st.button('Predict sentiment'):
        if len(new_text) > 0:
            predicted_class, predicted_proba = predict_sentiment(new_text)
            st.write("Predicted class:", "positive" if predicted_class == 1 else "negative")
            st.write("Probability of positive class:", predicted_proba[1])
            st.write("Probability of negative class:", predicted_proba[0])
        else:
            st.warning('Please enter a review.')
else:
    file_uploader = st.file_uploader('Upload a file (Word or TXT format)', type=['txt', 'docx'])
    if file_uploader:
        if file_uploader.size > 1000000:
            st.warning('File size exceeds the limit of 1 MB.')
        else:
            text = read_text_from_file(file_uploader)
            if st.button('Predict sentiment'):
                if len(text) > 0:
                    predicted_class, predicted_proba = predict_sentiment(text)
                    st.write("Predicted class:", "positive" if predicted_class == 1 else "negative")
                    st.write("Probability of positive class:", predicted_proba[1])
                    st.write("Probability of negative class:", predicted_proba[0])
                else:
                    st.warning('The file does not contain any text.')
