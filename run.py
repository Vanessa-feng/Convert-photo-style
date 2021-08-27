
import os
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

from gan_model import create_model
import cv2
import tensorflow as tf
from PIL import Image as im
import numpy as np
import time


# todo: more pretty interface

# folder to upload pictures
UPLOAD_FOLDER = 'static/uploads/'
# what files can upload
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# start + config
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS']=ALLOWED_EXTENSIONS

# main route
@app.route('/')
def index():
    return render_template('upload.html')

# is file allowed to be uploaded?
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# file upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Get the name of the uploaded file
    if request.method == 'POST':
        files = request.files.getlist('file')
        print('files', files)

        urllist = [[],[],[],[],[],[]]
        test_images=[]

        for file in files:
            print("for file")
        # Check if the file is one of the allowed types/extensions
            if file and allowed_file(file.filename):
                print('allow file')
                # remove unsupported chars etc
                filename = secure_filename(file.filename)
                print(filename)
                #save path
                save_to=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                #save file
                file.save(save_to)
                print('[INFO] input save to', save_to)

                url = url_for('static', filename="uploads/"+filename)
                urllist[0].append(url)
                
                img = cv2.imread(save_to)
                img = cv2.resize(img, (256,256))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_images.append(img)

        test_images = np.array(test_images) / 127.5 - 1
        test_images = tf.data.Dataset.from_tensor_slices(test_images).batch(1)

        model1 = create_model()
        model1.build = True
        model1._is_graph_network = True
        model1.load_weights("models/van_model.h5")

        for i, img in enumerate(test_images):
            pred1 = model1.v_gen(img, training=False)[0].numpy()
            pred1 = (pred1 * 127.5 + 127.5).astype(np.uint8)
            data1 = im.fromarray(pred1)
            name1 = "results/file_" + time.strftime('%H-%M-%S', time.localtime()) + '-van-' + str(i) + '.jpg'
            data1.save("static/" + name1, 'JPEG')
            url_res1 = url_for('static', filename=name1)
            urllist[1].append(url_res1)


        model2 = create_model()
        model2.build = True
        model2._is_graph_network = True
        model2.load_weights("models/monet_model.h5")

        for i, img in enumerate(test_images):
            pred2 = model2.v_gen(img, training=False)[0].numpy()
            pred2 = (pred2 * 127.5 + 127.5).astype(np.uint8)
            data2 = im.fromarray(pred2)
            name2 = "results/file_" + time.strftime('%H-%M-%S', time.localtime()) + '-monet-' + str(i) + '.jpg'
            data2.save("static/" + name2, 'JPEG')
            url_res2 = url_for('static', filename=name2)
            urllist[2].append(url_res2)

        model3 = create_model()
        model3.build = True
        model3._is_graph_network = True
        model3.load_weights("models/Chinese_model.h5")
        
        for i, img in enumerate(test_images):
            pred3 = model3.v_gen(img, training=False)[0].numpy()
            pred3 = (pred3 * 127.5 + 127.5).astype(np.uint8)
            data3 = im.fromarray(pred3)
            name3 = "results/file_" + time.strftime('%H-%M-%S', time.localtime()) + '-China-' + str(i) + '.jpg'
            data3.save("static/" + name3, 'JPEG')
            url_res3 = url_for('static', filename=name3)
            urllist[3].append(url_res3)

        model4 = create_model()
        model4.build = True
        model4._is_graph_network = True
        model4.load_weights("models/comb0.7_model.h5")
        
        for i, img in enumerate(test_images):
            pred4 = model4.v_gen(img, training=False)[0].numpy()
            pred4 = (pred4 * 127.5 + 127.5).astype(np.uint8)
            data4 = im.fromarray(pred4)
            name4 = "results/file_" + time.strftime('%H-%M-%S', time.localtime()) + '-China-' + str(i) + '.jpg'
            data4.save("static/" + name4, 'JPEG')
            url_res4 = url_for('static', filename=name4)
            urllist[4].append(url_res4)

        model5 = create_model()
        model5.build = True
        model5._is_graph_network = True
        model5.load_weights("models/comb0.3_model.h5")
        
        for i, img in enumerate(test_images):
            pred5 = model5.v_gen(img, training=False)[0].numpy()
            pred5 = (pred5 * 127.5 + 127.5).astype(np.uint8)
            data5 = im.fromarray(pred5)
            name5 = "results/file_" + time.strftime('%H-%M-%S', time.localtime()) + '-China-' + str(i) + '.jpg'
            data5.save("static/" + name5, 'JPEG')
            url_res5 = url_for('static', filename=name5)
            urllist[5].append(url_res5)

        print("[INFO] Generate images saved")
    return render_template('result.html', urllist=urllist)

if __name__ == '__main__':
   app.run(debug=True)