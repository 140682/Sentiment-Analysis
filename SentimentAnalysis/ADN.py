from flask import Flask, render_template, request
from classifier import SentimentClassifier

app = Flask(__name__)

classifier = SentimentClassifier()

@app.route('/review', methods=['POST'])
def do_search():
		results = 'Введите отзыв';
		comment = request.form['text']
		if comment!='': results = classifier.get_prediction_message(comment)

		return render_template('review.html', text_ = comment, the_results=results,)

@app.route('/')
@app.route('/review')
def entry_page():
    return render_template('review.html')

if __name__ == '__main__':
    app.run(debug=True)

