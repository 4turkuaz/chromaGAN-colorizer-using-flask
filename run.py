from flask import *

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['customFile']
        f.save(f.filename)
        return render_template('colorized.html', name = f.filename)
