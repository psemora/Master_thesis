# ---------------------------------------------------- #
# ----------------- Measure_program.py --------------- #
# ---------------------------------------------------- #
"""
Implemented algorithms with GUI for Master's thesis: Non–contact measurement of the dimensions of anal plate.
Author: Petr Šemora, 192026@vutbr.cz
"""
# Import necessary libraries
from scipy.spatial import distance as dist
from scipy import ndimage
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import math
from statistics import mean, median
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import environ
import tensorflow as tf
from tensorflow.keras.models import model_from_json
tf.autograph.set_verbosity(0)
import matplotlib
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import sys
import csv   
from matplotlib.backend_bases import MouseButton
import matplotlib.patches as patches
from matplotlib import transforms
import rawpy
from psd_tools import PSDImage
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from gui_ui import Ui_MainWindow  #from gui_ui file import UI_MainWindow class

#Main class for GUI
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.btn_find_ppm.clicked.connect(self.detect_ruler)
        self.btn_measure_plate.clicked.connect(self.detect_plate)
        self.btn_save_to_file.clicked.connect(self.save_to_file)
        self.MplWidget.canvas.mpl_connect('button_press_event', self.onpress)
        self.MplWidget.canvas.mpl_connect('button_release_event', self.onrelease)
        self.cb_image.activated[str].connect(self.handleItemPressed)
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_rotate.clicked.connect(self.rotate)
        self.btn_left.clicked.connect(self.rotate_left)
        self.btn_right.clicked.connect(self.rotate_right)
        self.tb_number_mm.textChanged[str].connect(self.set_number_mm)

        self.init()

        self.item = None
        self.cb_method.addItem("method A")
        self.cb_method.addItem("method B")
        self.cb_zoom.addItem("no zoom")
        self.cb_zoom.addItem("zoom")

        #Load weights for ruler and plate detection
        self.weights_ruler = "yolov3_ruler.weights" 
        self.weights_stitek = "yolov3_plate.weights" 
        
        
        try:
            #Load json models
            json_file = open('model1.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model1 = model_from_json(loaded_model_json)

            json_file = open('model2.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model2 = model_from_json(loaded_model_json)

            json_file = open('model3.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            self.model3 = model_from_json(loaded_model_json)
            
            #Load weights to models
            self.model1.load_weights("model1_weights.h5")
            self.model2.load_weights("model2_weights.h5")
            self.model3.load_weights("model3_weights.h5")
        except IOError as e:
            print('[Error] File not found!\nIOError: '+str(e))
            sys.exit()
        
    def init(self):
        #Init method - initializing variables for every new selected image
        self.flags = [0] * 8
        self.x_coords = [0]*4
        self.y_coords = [0]*4
        self.ppm = None
        self.length = 0
        self.width = 0
        self.length_metric = 0
        self.width_metric = 0
        self.lbl_ppm.setText("- px/mm")
        self.lbl_length.setText("- mm")
        self.lbl_width.setText("- mm")
        self.tb_angle.setText("0")
        self.angle = 0
        self.tb_number_mm.setText("1")
        self.number_mm = 1

    def open_dialog(self):
        #Open file explorer after button "btn_open" click
        dialog = QFileDialog()
        self.dir = dialog.getExistingDirectory(self, 'Select folder with images')
        print("[INFO] Selected folder:", self.dir)
        self.cb_image.clear()
        try:
            image_list = os.listdir(self.dir)
        except:
            return

        #Images file format, that can be opened are sorted into ComboBox cb_image
        for filename in image_list:
            if filename.endswith((".PNG", ".JPG", ".JPEG", ".RAW", ".SVG", ".GIF",".CR2", ".PSD", ".TIFF", ".BMP", ".png", ".jpg", ".jpeg", ".raw", ".svg", ".gif",".cr2", ".psd", ".tiff", ".bmp" )):
                self.cb_image.addItem(filename)
        
        
    def handleItemPressed(self):
        #Method capturing mouse click on image name in ComboBox cb_image  
        self.item = str(self.cb_image.currentText())
        print("[INFO] Image:", self.item)
        self.init()
        self.load_img()

    def load_img(self):
        #Load selected image 
        if self.item.endswith((".CR2", ".cr2")):
            raw = rawpy.imread(str(self.dir)+"/"+str(self.item))
            self.img = raw.postprocess()
        elif self.item.endswith((".PSD", ".psd")):
            psd = PSDImage.open(str(self.dir)+"/"+str(self.item))
            self.img = psd.composite()
            self.img = psd.numpy()
        else:
            self.img = cv2.imdecode(np.fromfile(str(self.dir)+"/"+str(self.item), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        #Show selected image on canvas
        self.MplWidget.canvas.axes.clear()
        self.MplWidget.canvas.axes.imshow(self.img)
        self.MplWidget.canvas.figure.subplots_adjust(left=0, right=1, bottom=0.06, top=0.97, hspace=0, wspace=0)
        self.MplWidget.canvas.draw()
    
    """
    Algorithms for manual finding scale of the image and measuring the dimensions of anal plate are below.
    """
    def rotate(self):  
        #Capturing angle on "tb_angle" TextBox and click on "btn_rotate" Button
        try:
            self.angle = int(self.tb_angle.text()) 
        except:
            QMessageBox.critical(self, 'Error', 'Value is allowed in <-180°, 180°>.')
            return
        try:
            if self.angle >= -180 and self.angle <= 180:
                self.rotate_image()
            else:
                QMessageBox.critical(self, 'Error', 'Value is allowed in <-180°, 180°>.')
        except:
            return
            
    def rotate_left(self):
        #Capturing click on "btn_left" Button
        try:
            self.angle = self.angle-1
            self.rotate_image()
        except:
            return

    def rotate_right(self):
        #Capturing click on "btn_right" Button
        try:
            self.angle = self.angle+1
            self.rotate_image()
        except:
            return

    def rotate_image(self):
        x_range = self.MplWidget.canvas.axes.get_xlim()
        y_range = self.MplWidget.canvas.axes.get_ylim()
        self.MplWidget.canvas.axes.clear()
        #Rotates the image for proper segmentation
        if self.angle == 90 or self.angle == 180 or self.angle == -90:
            self.img2=ndimage.rotate(self.img,-self.angle)
           
        #Rotates the image on canvas
        tr = transforms.Affine2D().rotate_deg_around(self.img.shape[1]/2, self.img.shape[0]/2, self.angle)
        self.MplWidget.canvas.axes.imshow(self.img, transform=tr+ self.MplWidget.canvas.axes.transData)
        self.MplWidget.canvas.axes.set_xlim(x_range)
        self.MplWidget.canvas.axes.set_ylim(y_range)
        self.MplWidget.canvas.figure.subplots_adjust(left=0, right=1, bottom=0.06, top=0.97, hspace=0, wspace=0)
        self.MplWidget.canvas.draw()
        self.tb_angle.setText(str(self.angle))
        self.draw_line()
        self.draw_rect()

    def onpress(self, event):
        #Capturing press on canvas with left mouse button
        if self.item == None:
            QMessageBox.critical(self, 'Error', 'No selected image')
            return
        else:
            if event.button == MouseButton.LEFT:
                click = event.xdata, event.ydata
                if None not in click:
                    self.x_press = int(event.xdata)
                    self.y_press = int(event.ydata)
                    self.x_range_press = self.MplWidget.canvas.axes.get_xlim()
                    self.y_range_press = self.MplWidget.canvas.axes.get_ylim()
            elif event.button == MouseButton.RIGHT:
                pass
        
            
    def onrelease(self, event):
        #Capturing release on canvas with left mouse button
        if event.button == MouseButton.LEFT:
            click = event.xdata, event.ydata
            if None not in click: 
                self.x_release = int(event.xdata)
                self.y_release = int(event.ydata)
                self.x_range_release = self.MplWidget.canvas.axes.get_xlim()
                self.y_range_release = self.MplWidget.canvas.axes.get_ylim()
                self.press_and_release(event.xdata, event.ydata)

    def press_and_release(self, xdata, ydata):
        #Capturing click (=press and release) on canvas with left mouse button
        if self.x_press == self.x_release and self.y_press == self.y_release and self.x_range_press == self.x_range_release and self.y_range_press == self.y_range_release: 
            xdata=round(xdata,3)
            ydata=round(ydata,3)
            #print(xdata, ydata)

            self.draw_point(xdata, ydata)
            self.draw_line()
            self.draw_rect()

            if self.flags[0] == 1 and self.flags[1] == 1 and (self.rb_ruler1.isChecked() or self.rb_ruler2.isChecked()):
                self.compute_ppm()
            if self.flags[2] == 1 and self.flags[3] == 1:
                self.compute_length()
                self.compute_width()


    def draw_point(self, xdata, ydata):
        #Draws points on canvas
        global point_mark1
        global point_mark2
        global point_mark3
        global point_mark4

        if self.rb_ruler1.isChecked():
            self.x_coords[0] = xdata
            self.y_coords[0] = ydata
            if self.flags[0] >= 1:
                point_mark1.remove()
            point_mark1, = self.MplWidget.canvas.axes.plot(self.x_coords[0], self.y_coords[0], marker='x', color="red")
            self.flags[0] = 1

        elif self.rb_ruler2.isChecked():
            self.x_coords[1] = xdata
            self.y_coords[1] = ydata
            if self.flags[1] >= 1:
                point_mark2.remove()
            point_mark2, = self.MplWidget.canvas.axes.plot(self.x_coords[1], self.y_coords[1], marker='x', color="red")
            self.flags[1] = 1

        elif self.rb_rect1.isChecked():
            self.x_coords[2] = xdata
            self.y_coords[2] = ydata
            if self.flags[2] >= 1:
                point_mark3.remove()
            point_mark3, = self.MplWidget.canvas.axes.plot(self.x_coords[2], self.y_coords[2], marker='x', color="blue")
            self.flags[2] = 1

        elif self.rb_rect2.isChecked():
            self.x_coords[3] = xdata
            self.y_coords[3] = ydata
            if self.flags[3] >= 1:
                point_mark4.remove()
            point_mark4, = self.MplWidget.canvas.axes.plot(self.x_coords[3], self.y_coords[3], marker='x', color="blue")
            self.flags[3] = 1
            self.MplWidget.canvas.draw()

    def draw_line(self):
        #Draws line on canvas between point_mark1 and point_mark2
        global line1
        if self.x_coords[0] and self.y_coords[0] and self.x_coords[1] and self.y_coords[1] is not None:
            if self.flags[6] >= 1:
                line1.remove()
            line1, = self.MplWidget.canvas.axes.plot((self.x_coords[0], self.x_coords[1]),(self.y_coords[0], self.y_coords[1]), color="red")
            self.flags[6] = 1
        self.MplWidget.canvas.draw()
        

    def draw_rect(self):
        #Draws rectangle on canvas between point_mark3 and point_mark4
        global rect
        if self.x_coords[2] and self.y_coords[2] and self.x_coords[3] and self.y_coords[3] is not None:
            if self.flags[7] >= 1:
                rect.remove()
            w = self.x_coords[3] - self.x_coords[2]
            h = self.y_coords[3] - self.y_coords[2]
            rect = patches.Rectangle((self.x_coords[2], self.y_coords[2]), w, h, linewidth=1, edgecolor='blue', facecolor='none')
            self.MplWidget.canvas.axes.add_patch(rect)
            self.flags[7] = 1
        self.MplWidget.canvas.draw()

    def set_number_mm(self):
        #Sets number of mm between point_mark1 and point_mark2 to find the scale of the image
        try:
            self.number_mm = int(self.tb_number_mm.text()) 
        except:
            QMessageBox.critical(self, 'Error', 'Only numbers are allowed.')
            
            return
        if self.number_mm == 0:
            QMessageBox.critical(self, 'Error', 'Minimum is 1.')
            self.number_mm = 1
            self.tb_number_mm.setText("1")
        if self.flags[0] == 1 and self.flags[1] == 1 or self.ppm is not None:
            self.compute_ppm()

    def compute_ppm(self):
        #Computes pixels per metric to find the scale of image
        self.ppm = dist.euclidean((self.x_coords[0], self.y_coords[0]), (self.x_coords[1], self.y_coords[1]))/self.number_mm
        self.ppm = round(self.ppm,3)
        self.lbl_ppm.setText(str(self.ppm)+" px/mm")
        print("[INFO] Manual - Number of pixels per mm: ", self.ppm)
        

    def compute_length(self):
        #Computes length of the plate, which is width of the rectangle
        self.length = np.abs(self.x_coords[3] - self.x_coords[2])
        self.length = round(self.length,3)
        if self.ppm is not None:
            self.length_metric = self.length / self.ppm
            self.length_metric = round(self.length_metric,3)
            self.lbl_length.setText(str(self.length_metric)+" mm")
            print(f'[INFO] Length of the plate: {self.length} px => {self.length_metric} mm.')
        else:
            print(f'[INFO] Length of the plate: {self.length} px.')
            
            
        
    def compute_width(self):    
        #Computes width of the plate, which is height of the rectangle
        self.width = np.abs(self.y_coords[3] - self.y_coords[2])
        self.width = round(self.width,3) 
        if self.ppm is not None:
            self.width_metric = self.width / self.ppm
            self.width_metric = round(self.width_metric,3)
            self.lbl_width.setText(str(self.width_metric)+" mm")
            print(f'[INFO] Width of the plate: {self.width} px => {self.width_metric} mm.')
        else:
            print(f'[INFO] Width of the  plate: {self.width} px.')

    def save_to_file(self):
        #Saves data to file "data.csv"
        if self.item is not None:
            csv.register_dialect("del", delimiter=";")
            with open(r"data.csv", "a") as f:
                writer = csv.writer(f, dialect="del")
                writer.writerow((self.item, self.ppm, self.length, self.width, self.length_metric, self.width_metric))
            print("[INFO] Data were succesfully saved.")
            QMessageBox.information(self, 'Save Data','Data were succesfully saved.')
        else:
            QMessageBox.critical(self, 'Error', 'No file to save.')

    """
    Algorithms for automatic finding scale of the image and measuring the dimensions of anal plate are below.
    """

    def detect_ruler(self):
        #Detects ruler 
        if self.angle == 90 or self.angle == 180 or self.angle == -90:
            self.img=self.img2
        try:
            x,y,w,h,_,_ = yolo_cord(self.img, self.weights_ruler)
            print("[INFO] Ruler detected")
        except:
            QMessageBox.critical(self, 'Error', 'No ruler detected.')
            return
        
        if x < 0:
            x = 0 
        if y < 0:
            y = 0
        if x+w > self.img.shape[1]:
            w = self.img.shape[1]-x
        if y+h > self.img.shape[0]:
            h = self.img.shape[0]-y

        method = str(self.cb_method.currentText())
        #Crops ruler from the image
        img = self.img.copy()
        ruler_crop = img[y:y+h, x:x+w]
        if method == "method A":
            self.method_A(ruler_crop)
        else:
            self.method_B(ruler_crop)
        

    def method_A(self, ruler_crop):
        #Method A for finding scale of the image using big black rectangles on ruler
        #Converts ruler crop into HSV color model
        try:
            hsv = cv2.cvtColor(ruler_crop, cv2.COLOR_RGB2HSV)
        except:
            QMessageBox.critical(self, 'Error', 'No ruler detected.')
            return
        #Converts ruler crop into binary black/white image
        lower_val = np.array([0,0,0])
        upper_val = np.array([180,150,100])
        mask = cv2.inRange(hsv, lower_val, upper_val)
        kernel = np.ones((5,5),np.uint8)
        
        #Dilade of the crop
        dil = cv2.dilate(mask, kernel, iterations = 1)
        #Erode of the crop 
        er = cv2.erode(dil, kernel, iterations = 4)
        #Finds big black rectangles on the crop
        cnts = cv2.findContours(er, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if cnts is not None:
            cnts = imutils.grab_contours(cnts)
        else:
            QMessageBox.critical(self, 'Error', 'No big contours found. Try method B.')
            return
        #Sorts all found contours and select only those contours that match the big black rectangles
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        big_cnts = []
        big_cnts2 = []
        big_cnts2_dA = []
        big_cnts2_dB = []
        big_cnts = cnts[:8]
        for c in big_cnts:
            #Min area bounding box of 8 biggest contours 
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) 
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)

            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            #Computes euclidean distance
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            #Select contours that match the dimensions of rectangles
            if dB > 1.4*dA and dB < 1.8*dA:
                big_cnts2.append(c)
                big_cnts2_dA.append(dA)
                big_cnts2_dB.append(dB)

        for c in big_cnts2:
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            #Draws selected contours
            cv2.drawContours(ruler_crop, [box.astype("int")], -1, (255, 0, 0), 2)
        
        WIDTH = 10 #constant for assigning the real dimension of the contour width
        if len(big_cnts2) == 0:
            QMessageBox.critical(self, 'Error', 'No big contours found. Try method B.')
            return
        else:
            dB_mean =  sum(big_cnts2_dB)/len(big_cnts2_dB) 
            self.ppm = (dB_mean / WIDTH) 
            #correction of the found dimension with experimental found values
            if self.ppm < 15:
                self.ppm = self.ppm*1.1
            elif self.ppm < 15 and self.ppm <20:
                self.ppm = self.ppm*1.08
            elif self.ppm < 20 and self.ppm <25:
                self.ppm = self.ppm*1.06
            else:
                self.ppm = self.ppm*1.04
            print("[INFO] Method A - Number of pixels per mm: ", self.ppm)
        
        self.ppm = round(self.ppm,3)
        self.lbl_ppm.setText(str(self.ppm)+" px/mm")

        #shows selected contours on the crop of the ruler if CheckBox "chb_show_ruler" is checked
        if self.chb_show_ruler.isChecked():
            plt.ion()
            plt.imshow(ruler_crop)
            plt.show()
        else:
            return

    def method_B(self, crop):
        #Method B for finding scale of the image using every square of milimeter
        #Converts ruler crop into shades of gray 
        ruler_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        #Blur of the crop
        blur = cv2.GaussianBlur(ruler_gray,(5,5),0)
        #Thresholds of the ruler crop
        ruler_thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        try:
            cnts = cv2.findContours(ruler_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except:
            QMessageBox.critical(self, 'Error', 'No detected contours.')
            return
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)
        #Sorts all found contours and select only those contours that match the square of milimeter
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        c_area = []
        ws = [] 
        hs = [] 
        cnts_close= []
        cnts_close_dA= []
        cnts_close_dB= []
        for c in cnts:
            #Min area bounding box of all found contours
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box) 
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = midpoint(tl, tr)
            (blbrX, blbrY) = midpoint(bl, br)

            (tlblX, tlblY) = midpoint(tl, bl)
            (trbrX, trbrY) = midpoint(tr, br)

            #Computes euclidean distance
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            #Selects contours that match the dimensions of squares
            if dA > 5 and math.isclose(dA, dB, abs_tol = 5):
                cnts_close.append(c)
                cnts_close_dA.append(dA)
                cnts_close_dB.append(dB)
                c_area.append(np.round(cv2.contourArea(c),0))

        #Discards contours with extreme area value
        c_area = c_area[int(len(c_area)/10):int(len(c_area)*9/10)]

        if len(cnts_close)==0 or len(c_area)==0:
            QMessageBox.critical(self, 'Error', 'No detected contours.')
            return
        else:
            for c in cnts_close:
                #Selects contours with optimal contour areas 
                if math.isclose(cv2.contourArea(c),median(c_area), abs_tol = 10):
                    x,y,w,h = cv2.boundingRect(c)
                    box = cv2.minAreaRect(c)
                    cv2.rectangle(crop,(x,y),(x+w,y+h),(255,0,0),1)
                    ws.append(w)
                    hs.append(h)        

        if len(ws) == 0:
            QMessageBox.critical(self, 'Error', 'Contours on ruler can not be detected.')
            return
        else:
            self.ppm = (mean(ws) + mean(hs)) / 2
            #Correction of the found dimension with experimental found values
            if self.ppm < 15:
                self.ppm = 1.25*self.ppm
            elif self.ppm < 15 and self.ppm <20:
                self.ppm = 1.2*self.ppm
            elif self.ppm < 20 and self.ppm <25:
                self.ppm = 1.15*self.ppm
            else:
                self.ppm = 1.1*self.ppm
            print("[INFO] Method B - Number of pixels per mm: ", self.ppm)

            self.ppm = round(self.ppm,3)
            self.lbl_ppm.setText(str(self.ppm)+" px/mm")
            #Shows selected contours on the crop of the ruler if CheckBox "chb_show_ruler" is checked
            if self.chb_show_ruler.isChecked():
                plt.ion()
                plt.imshow(crop)
                plt.show()
            else:
                return

        
    def detect_plate(self):
        #Detects anal plate
        if self.angle == 90 or self.angle == 180 or self.angle == -90:
            self.img=self.img2
        try:
            _,_,w,h, center_x, center_y = yolo_cord(self.img, self.weights_stitek)
            print("[INFO] Plate detected")
        except:
            QMessageBox.critical(self, 'Error', 'No plate detected.')
            return


        self.zoom = str(self.cb_zoom.currentText())
        #crops anal plate from the image
        if self.zoom == "no zoom":
            stitek_crop = self.img[center_y-160:center_y+160, center_x-240:center_x+240]
        else:
            stitek_crop = self.img[center_y-80:center_y+80, center_x-120:center_x+120]
            w = 480
            h = 320
            dim = (w, h)
            stitek_crop = cv2.resize(stitek_crop, dim, interpolation = cv2.INTER_AREA)
        self.segment_plate(stitek_crop)

    def segment_plate(self, stitek_crop):
        #Segmentation of anal plate
        try:
            img_expand = np.expand_dims(stitek_crop, axis = 0)
        except:
            QMessageBox.critical(self, 'Error', 'No plate detected.')
            return
        #Prediction using each model
        pred1 = self.model1.predict(img_expand)[0]
        pred2 = self.model2.predict(img_expand)[0]
        pred3 = self.model3.predict(img_expand)[0]
        #Model ensemble
        preds=np.array([pred1, pred2, pred3])
        weights = [1,1,1]
        weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
        pred = np.argmax(weighted_preds, axis=-1)
        #Postprocessing of segmented anal plate
        pred = (pred*255).astype(np.uint8)
        pred=cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        pred_copy = pred.copy()
        gray = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)
        edged = cv2.Canny(gray, 50, 100)
        cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        if len(cnts) == 0:
            QMessageBox.critical(self, 'Error', 'Plate can not be segmented')
            return
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        c = cnts[0]
        mask = np.zeros_like(edged , dtype=np.uint8)
        cv2.drawContours(mask, [c], 0, 255, -1)
        pred[mask==0] = (0,0,0)
        pred2 = np.where(pred==255, 255, stitek_crop)
        self.measure_plate(c, pred_copy, stitek_crop)

    def measure_plate(self, c, pred, pred2):
        #Measurement of the dimensions of anal plate
        #Fits ellipse to segmented scale
        ellipse = cv2.fitEllipse(c)
        #Finds angle for rotation
        (_,_),(_,_),angle = ellipse
        if angle > 90:
            angle = angle-180
        else:
            angle = angle
        #Rotates segmented scale perpendicular to the axes
        height, width = pred.shape[:2]
        center = (width/2, height/2)
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)
        rotated = cv2.warpAffine(src=pred, M=rotate_matrix, dsize=(width, height))
        pred2_rotated = cv2.warpAffine(src=pred2, M=rotate_matrix, dsize=(width, height))
        #Finds contour of the rotated scale
        edged = cv2.Canny(rotated, 50, 100)
        cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        c = cnts[0]
        #Fits rectangle of the rotated scale
        x,y,w,h = cv2.boundingRect(c)
        box = np.array([[x, y],[x+w, y],[x, y+h],[x+w, y+h]])
        box = perspective.order_points(box)
        #Draws the rectangle
        cv2.drawContours(pred2_rotated, [box.astype("int")], -1, (255, 0, 0), 1)
        #Rotates segmented scale back
        rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-angle, scale=1)
        pred2_back = cv2.warpAffine(src=pred2_rotated, M=rotate_matrix, dsize=(width, height))
        edged_back = cv2.warpAffine(src=edged, M=rotate_matrix, dsize=(width, height))
        edged_back = cv2.cvtColor(edged_back, cv2.COLOR_GRAY2RGB)
        #Draws contour of segmented scale to the crop of the scale
        edged_back[np.where((edged_back>[0, 0, 0]).all(axis=2))] = [255, 0, 0]
        pred2 = cv2.add(pred2_back,edged_back)

        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)

        #Computes euclidean distance of dimensions of anal plate in pixels
        if self.zoom == "no zoom":
            self.width = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            self.length = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
        else:
            self.width = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))/2
            self.length = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))/2
        
        #Computes dimensions of anal plate in milimeters
        if self.ppm is not None:
            self.length = round(self.length,3)
            self.length_metric = self.length / self.ppm
            self.length_metric = round(self.length_metric,3)
            self.lbl_length.setText(str(self.length_metric)+" mm")

            self.width = round(self.width,3)
            self.width_metric = self.width / self.ppm
            self.width_metric = round(self.width_metric,3)
            self.lbl_width.setText(str(self.width_metric)+" mm")

            print(f'[INFO] Length of the plate: {self.length} px => {self.length_metric} mm.')
            print(f'[INFO] Width of the plate: {self.width} px => {self.width_metric} mm.')
            
        else:
            print("[INFO] PPM is not set")
            print(f'[INFO] Length of the plate: {self.length} px.')
            print(f'[INFO] Width of the plate: {self.width} px.')

        #shows segmented anal plate on the crop of the scale if CheckBox "chb_show_plate" is checked
        if self.chb_show_plate.isChecked():
            plt.ion()
            plt.imshow(pred2)
            plt.show()
        else:
            return

def midpoint(ptA, ptB):
    #Method for counts midpoint of two points
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def yolo_cord(img, weights):
    #Method for objects detection using yolov3 and pretrained weights
    net = cv2.dnn.readNet(weights, "yolov3_testing.cfg")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)

                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
    return x,y,w,h, center_x, center_y

#Suppress warnings
def suppress_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
    environ['AUTOGRAPH_VERBOSITY'] = "1"


if __name__ == '__main__':
    suppress_warnings()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())