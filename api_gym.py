from flask import Flask, flash, request ,jsonify, render_template
from werkzeug.utils import secure_filename
from myml import * # ทำการ import predict เข้ามา
import os

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# prob threshold required for making a prediction
prob_threshold = 0.3

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
            label, y_prob = predictImage('./uploads/image.png')
            if len(y_prob[y_prob > prob_threshold]) >= 2 or len(y_prob[y_prob > prob_threshold]) <= 0:
                label = 'ไม่สามารถระบุชนิดได้ โปรดอัพโหลดภาพมุมอื่น'
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

# สร้าง Request สำหรับ model รับ url
@application.route('/url') 
def predict_img_from_url():
    #default_url = 'https://i.pinimg.com/originals/82/13/94/82139469411aefc48c7c42375ff56c9e.jpg'
    this_url = request.args.get('p_image_url', default='please provide url', type=str)
    try:
        label, y_prob = predictImageFromURL(this_url)
        print(y_prob)
        print(label)
        if len(y_prob[y_prob > prob_threshold]) >= 2 or len(y_prob[y_prob > prob_threshold]) <= 0:
            label = 'ไม่สามารถระบุชนิดได้ โปรดอัพโหลดภาพมุมอื่น'
    except:
        label = 'URL not valid. Please try other URLs.'
    return jsonify({'label': label})


if __name__ == '__main__':
    # application.debug = False
    # application.run(host='0.0.0.0', port=8080)
    # application.run(debug=False)
    application.run(debug=do_debug)


# #---------------------------------#
# # To use this api in Python
# #---------------------------------#
# import requests
# # upload your image to colab
# PATH_TO_INPUT_IMAGE = '/content/1062404-1573963307-167380.jpg'
# filess = {"file": open(PATH_TO_INPUT_IMAGE, "rb")}
# '''
# The instance "file" is created in the api_gym.py 
# See the line includig: 
# file = request.files['file']
# '''
# url = "https://gym-images-api.herokuapp.com/upload"
# response = requests.post(url, files=filess)
# print(response.text)