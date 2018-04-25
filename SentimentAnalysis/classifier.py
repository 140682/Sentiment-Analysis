__author__ = 'ADN'
from sklearn.externals import joblib


class SentimentClassifier(object):
    def __init__(self):
        self.model = joblib.load("./ModelLogisticRegressionClassifier.pkl")
        self.vectorizer = joblib.load("./ModelCountVectorizer.pkl")
        self.classes_dict = {0: "негативный", 1: "позитивный", -1: "ошибка"}

    @staticmethod
    def get_probability_words(probability):
        if probability < 0.55:
            return "Нейтральный / не уверен"
        if probability < 0.7:
            return "Вероятно"
        if probability > 0.95:
            return "Определенно"
        else:
            return ""

    def predict_text(self, text):
        try:
            vectorized = self.vectorizer.transform([text])
            return self.model.predict(vectorized)[0],\
                   self.model.predict_proba(vectorized)[0].max()
        except:
            print("ошибка")
            return -1, 0.8

    def predict_list(self, list_of_texts):
        try:
            vectorized = self.vectorizer.transform(list_of_texts)
            return self.model.predict(vectorized),\
                   self.model.predict_proba(vectorized)
        except:
            print('ошибка')
            return None

    def get_prediction_message(self, text):
        prediction = self.predict_text(text)
        class_prediction = prediction[0]
        prediction_probability = prediction[1]
        return self.get_probability_words(prediction_probability) + " " + self.classes_dict[class_prediction]