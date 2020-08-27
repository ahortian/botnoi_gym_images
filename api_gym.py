from flask import Flask, flash, request ,jsonify, render_template
from werkzeug.utils import secure_filename
from myml import * # ทำการ import predict เข้ามา
import os

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

do_debug = "apicharthortiangtham" in os.getcwd()

# create Flask object
application = Flask(__name__)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return ('.' in filename) and (filename.rsplit('.', 1)[1].lower() 
                              in ALLOWED_EXTENSIONS)


# สร้าง Home Page
@application.route('/') 
def main():
    return 'Hello This is home page'


# สร้าง Request สำหรับ model
@application.route('/upload', methods=['GET', 'POST']) 
def upload_file():
    if request.method == 'POST': # ต้องผ่าน Post Medthod เท่านั้น
        # ถ้าไม่มีไฟล์แนบมา
        if 'file' not in request.files:
            flash('No file part')
            return 'No file part'

        file = request.files['file']
        # ถ้า User ไม่แนบไฟล์มา
        if file.filename == '':
            flash('No selected file')
            return 'No selected file'
        # ถ้ามีFileและtype ถูกต้อง
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Save File
            file.save(os.path.join(application.config['UPLOAD_FOLDER'],    
                             'image.png'))
            # นำรูปไปใส่ใน Model 
            label = predictImage('./uploads/image.png')
            return jsonify({'label': label})
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

if __name__ == '__main__':
    # application.debug = False
    # application.run(host='0.0.0.0', port=8080)
    # application.run(debug=False)
    application.run(debug=do_debug)