import os
from flask import *
from werkzeug.utils import secure_filename
from src import colorize

app = Flask(__name__, static_url_path = '/static', static_folder = 'static')

# Create a directory in a known location to save files to.
uploads_dir = os.path.join(app.instance_path, 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/upload-image', methods = ['POST'])
def upload_image():
    if request.method == 'POST':
        f = request.files['image']
        f.save(os.path.join(uploads_dir, secure_filename(f.filename)))
        output = colorize.colorize(uploads_dir)
        return render_template('colorized.html', picture = output, original_picture = f.filename)
