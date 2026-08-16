import os
import pickle

from flask import Flask, render_template, request
from flask_cors import CORS
from newspaper import Article

app = Flask(__name__, template_folder="templates")
CORS(app)

with open("model.pickle", "rb") as file:
    model = pickle.load(file)


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/predict", methods=["POST"])
def predict():
    url = request.form.get("news", "").strip()

    if not url:
        return render_template(
            "main.html",
            prediction_text="Please enter a news article URL."
        )

    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()

        article_summary = article.summary

        prediction = model.predict([article_summary])[0]

        return render_template(
            "main.html",
            prediction_text=f"The news is '{prediction}'."
        )

    except Exception:
        return render_template(
            "main.html",
            prediction_text="Unable to process the URL. Please enter a valid news article URL."
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)