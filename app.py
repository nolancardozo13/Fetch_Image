import os
import argparse

from flask import Flask, request, render_template, send_from_directory
from helper import get_images
from helper import get_images_by_doc_2_vec, get_images_by_word_centroid_distance

app = Flask(__name__)

# Constants
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER_NAME = "images"
ABSOLUTE_PATH = None
RETRIEVAL_METHOD = "Doc2Vec"
IMAGE_SIZE_TO_BE_RETRIEVED = 10
IMAGE_FOLDER_PATH = None

# initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("--method", help="Set Retrieval Method to Doc2Vec/BM25/WordCentroidDistance", choices=['BM25', 'Doc2Vec', 'WCD'])
parser.add_argument("--size", help="Total number of images to be returned. Default is 10")
parser.add_argument("--relative",
                    help="Relative path difference from current working directory to image folder")
parser.add_argument("--imageFolder",
                    help="Image folder name where the images and captions are stored")
parser.add_argument("--totalFolderPath",
                    help="Absolute path to the image folder name where the images and captions are stored")

# read arguments from the command line
args = parser.parse_args()

# check for --method
if args.method:
    RETRIEVAL_METHOD = args.method

# check for --size
if args.size:
    IMAGE_SIZE_TO_BE_RETRIEVED = args.size

# check for --folder
if args.imageFolder:
    IMAGE_FOLDER_NAME = args.imageFolder

# check for --absolute
if args.relative:
    ABSOLUTE_PATH = args.relative

if args.totalFolderPath:
    IMAGE_FOLDER_PATH = args.totalFolderPath
else:
    if ABSOLUTE_PATH:
        IMAGE_FOLDER_PATH = os.path.join(APP_ROOT, ABSOLUTE_PATH, IMAGE_FOLDER_NAME)
    else:
        IMAGE_FOLDER_PATH = os.path.join(APP_ROOT, IMAGE_FOLDER_NAME)
if RETRIEVAL_METHOD == "Doc2Vec":
    model = get_images_by_doc_2_vec.create_doc_to_vector_for_given_images(IMAGE_FOLDER_PATH)
elif RETRIEVAL_METHOD == "WCD":
    model = get_images_by_word_centroid_distance.create_word_centroid_distance_for_given_images(IMAGE_FOLDER_PATH)


@app.route('/')
def search():
    return render_template('search.html')


@app.route('/', methods=['POST'])
def search_post():
    query = request.form['text']
    if RETRIEVAL_METHOD == "BM25":
        image_names = get_images.get_top_n_images(IMAGE_FOLDER_PATH, query, IMAGE_SIZE_TO_BE_RETRIEVED)
    elif RETRIEVAL_METHOD == "Doc2Vec":
        image_names = get_images_by_doc_2_vec.get_top_n_images(model, query, IMAGE_SIZE_TO_BE_RETRIEVED)
    elif RETRIEVAL_METHOD == "WCD":
        image_names = get_images_by_word_centroid_distance.get_top_n_images(model, query, IMAGE_SIZE_TO_BE_RETRIEVED)
    return render_template("results.html", image_names=image_names)


@app.route('/<filename>')
def send_image(filename):
    return send_from_directory(IMAGE_FOLDER_PATH, filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)