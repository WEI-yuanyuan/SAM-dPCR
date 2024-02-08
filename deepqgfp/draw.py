import sys
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import cv2
from io import BytesIO
import os
from threading import Thread
import tkinter as tk
from PIL import Image, ImageTk
from detect2

root = tk.Tk()
class Example():
    def __init__(self):
        h, w = 400,600
        xpad, ypad = 50, 50
        root.geometry("%dx%d"%(w,h))
        canvas = tk.Canvas(root, height=h, width=w)
        canvas.place(x=0, y=0)
        
        tklbl0 = tk.Label(text='Deep-learning enabled droplet digital PCR', font='bold')
        tklbl0.place(x=140, y=20)

        img = Image.open('img.png')

        tkimg1 = ImageTk.PhotoImage(img)
        tklbl1 = tk.Label(image=tkimg1)
        tklbl1.place(x=xpad, y=ypad) #x, y are starting corner

        plot = Image.open('plt.png')

        tkimg2 = ImageTk.PhotoImage(plot)
        tklbl2 = tk.Label(image=tkimg2)
        tklbl2.place(x=w//2+xpad, y=ypad) #x, y are starting corner

        tklbl3 = tk.Label(text='Results')
        tklbl3.place(x=xpad, y=300)
        tklbl4 = tk.Label(text='Droplet diameter')
        tklbl4.place(x=xpad+10, y=330)
        tklbl5 = tk.Label(text='Calculated concentration')
        tklbl5.place(x=xpad+10, y=360)

        tklbl6 = tk.Label(text='100',bg='#FFFFFF', width=10)
        tklbl6.place(x=xpad*5+10, y=330)
        tklbl7 = tk.Label(text='100',bg='#FFFFFF', width=10)
        tklbl7.place(x=xpad*5+10, y=360)

        tkbtn1 = tk.Button(root, text='Stop')
        tkbtn1.place(x=xpad*7, y=345)
        
        img = Image.open('logo1.png')

        tkimg3 = ImageTk.PhotoImage(img)
        tklbl8 = tk.Label(image=tkimg3)
        tklbl8.place(x=xpad*9, y=320) #x, y are starting corner

        plot = Image.open('logo2.png')

        tkimg4 = ImageTk.PhotoImage(plot)
        tklbl9 = tk.Label(image=tkimg4)
        tklbl9.place(x=xpad*9, y=360) #x, y are starting corner



def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'best.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    #parser.add_argument('--source', type=str, default=ROOT / 'set11/Test', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--source', type=str, default=0, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

if __name__ == '__main__':
    Example()

    root.mainloop()


