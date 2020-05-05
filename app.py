from flask import *
from process_image import process_image
import os

app = Flask(__name__)

@app.route('/')
def upload():
	return render_template("index.html")

@app.route('/upload', methods=['POST'])
def success():
	if request.method == 'POST':
		f = request.files['file']
		image_file_name = 'working_dir/' + f.filename
		f.save(image_file_name)
		try:
			prediction = process_image(image_file_name)
		except:
			prediction = "Prediction failed"
		finally:
			delete_image_file(image_file_name)
		return render_template("index.html", result=prediction)

def delete_image_file(image_file_name):
	if image_file_name.startswith('working_dir/'):
		os.remove(image_file_name)

if __name__ == '__main__':
	app.run()
