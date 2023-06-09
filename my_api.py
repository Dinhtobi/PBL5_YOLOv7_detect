from flask import Flask ,render_template
from flask import request
from flask_cors import CORS , cross_origin 
import os
import glob
import myYolov7
#Khởi tạo Flask Server Backend
app = Flask(__name__)

#Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS']  ='Content-Type'
app.config['UPLOAD_FOLDER'] = "static"

model = myYolov7.my_yolov7('last.pt','cpu',0.6)

def save_image(image):
    folder_path = "test_image"
    image_count = len(glob.glob(folder_path + "/*.jpg"))
    path_video = folder_path + "/anh{}".format(image_count+1)+".jpg"
    image.save(path_video)
    return path_video

def save_video(video):
    
    folder_path = "test_image"
    video_count = len(glob.glob(folder_path + "/*.mp4"))
    path_video = folder_path + "/video{}".format(video_count+1)+".mp4"
    video.save(path_video)
    return path_video

@app.route('/' , methods= ['POST' , 'GET'])
@cross_origin(origins='*')

def trangchu():
    return render_template('index.html')


@app.route('/predict' , methods= ['POST'])
@cross_origin(origins='*')

def predict_video_YOLOv7_proces():

    file = request.files['file']
    duoi = file.filename.split(".")[1]
    if duoi == "mp4":
       path_file =  save_video(file)
    elif duoi == "jpg":
       path_file =  save_image(file)
    imgs = path_file 
    listfile = os.listdir('test_image')
    count = len(listfile)
    savepath = 'output_img/image.jpg'
    model.detect(imgs,savepath,count)
    return "success"


#Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0' , port = '6868',use_reloader=True)
    
