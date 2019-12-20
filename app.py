import os


from flask import Flask, request, render_template, send_from_directory
from helper import get_images

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/')
def search():
    return render_template('search.html')

@app.route('/', methods=['POST'])
def search_post():
    query = request.form['text']
    folder = "C:/Users/nolan/Desktop/Fetch_Image/images"
    image_names = get_images.get_top_n_images(folder,query,n=10)
    print(image_names)
    return render_template("results.html", image_names=image_names)

@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)