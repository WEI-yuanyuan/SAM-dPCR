import cellpose
from cellpose import models
from cellpose import io
from cellpose.io import imread
import os
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries 

from matplotlib.pylab import *
import cv2
from io import BytesIO
from threading import Thread
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import argparse

def getmasks(folder=None):
    if folder is None:
        folder = 'CNew'
    files = os.listdir(folder)

    model = models.Cellpose(gpu=True, model_type='cyto')

    imgs = [imread("%s/%s"%(folder, f)) for f in files]

    masks, flows, styles, dms = model.eval(imgs)

    for n in range(len(imgs)):
        mask = masks[n]
        areas = []
        circularities = []
        for i in range(np.bincount(mask.flatten()).shape[0] -1):
            ia = (mask==i+1).astype(int) 
            ip = find_boundaries(ia).astype(int) 
            circularity = 4*math.pi*ia.sum() / (ip.sum()**2)
            areas.append(ia.sum())
            circularities.append(circularity)
        yield(imgs[n], mask, areas, circularities)

def plotfunction():
    buf = BytesIO()
    plt.axis('off')
    plt.set_cmap('nipy_spectral')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    img = Image.open(buf)
    return img

class Example():
    def __init__(self): 
        root = tk.Tk()
        root.title("Auto-ICell Real-time Analysis")
        h, w = 400,600
        xpad, ypad = 50, 50
        root.geometry("%dx%d"%(w,h))
        root['bg']='white'
        canvas = tk.Canvas(root, height=h, width=w)
        canvas.place(x=0, y=0)
        canvas['bg']='white'
        
        tklbl0 = tk.Label(text='Auto-ICell Real-time Analysis',bg='#FFFFFF', font='bold')
        # tklbl0.configure(bg='white')
        #tklbl0.pack()
        #print(tklbl0.winfo_width()//2)
        tklbl0.place(x=w//2-150, y=ypad//2)

        img = Image.open('a.png')
        #iw, ih = img.size[:2]
        #img = img.resize((iw*h//2//ih, h//2))
        global tklbl1
        tkimg1 = ImageTk.PhotoImage(img)
        tklbl1 = tk.Label(image=tkimg1)
        tklbl1.place(x=xpad, y=ypad) #x, y are starting corner

        plot = Image.open('plt.png')
        #iw, ih = plot.size[:2]
        #plot = plot.resize((iw*h//2//ih, h//2))
        # print(img.size, plot.size)
        global tklbl2
        tkimg2 = ImageTk.PhotoImage(plot)
        tklbl2 = tk.Label(image=tkimg2)
        tklbl2.place(x=w//2+xpad, y=ypad) #x, y are starting corner

        tklbl3 = tk.Label(bg='#FFFFFF', text='Results')
        tklbl3.place(x=xpad, y=h*3//4)
        tklbl10 = tk.Label(bg='#FFFFFF', text='Cells detected')
        tklbl10.place(x=xpad+10, y=h*3//4+h//20)
        tklbl4 = tk.Label(bg='#FFFFFF', text='Cell areas')
        tklbl4.place(x=xpad+10, y=h*3//4+h//10+5)
        tklbl5 = tk.Label(bg='#FFFFFF', text='Cell circularities')
        tklbl5.place(x=xpad+10, y=h*3//4+h*3//20+10)

        global tklbl6, tklbl7, tklbl11
        tklbl11 = tk.Label(text='100',bg='#FFFFFF', width=15)
        tklbl11.place(x=xpad+w*3//10+10, y=h*3//4+h//20)
        tklbl6 = tk.Label(text='100',bg='#FFFFFF', width=15)
        tklbl6.place(x=xpad+w*3//10+10, y=h*3//4+h//10+5)
        tklbl7 = tk.Label(text='100',bg='#FFFFFF', width=15)
        tklbl7.place(x=xpad+w*3//10+10, y=h*3//4+h*3//20+10)

        def close():
            root.quit()

        tkbtn1 = tk.Button(root, text='Stop', command = close)
        tkbtn1.place(x=xpad*3//2+w*5//10, y=h*3//4+h*3//20-ypad//4)
        
        logo1 = Image.open('logo1.png')

        tkimg3 = ImageTk.PhotoImage(logo1)
        tklbl8 = tk.Label(bg='#FFFFFF',image=tkimg3)
        tklbl8.place(relx=0.7, y=h-ypad*2) #x, y are starting corner

        logo2 = Image.open('logo2.png')

        tkimg4 = ImageTk.PhotoImage(logo2)
        tklbl9 = tk.Label(bg='#FFFFFF',image=tkimg4)
        tklbl9.place(relx=0.7, y=h-ypad) #x, y are starting corner
        
        
        global cap
        # opt = parse_opt()
        # cap = run(**vars(opt))
        cap = getmasks(folder='C:/Users/Yuanyuan/Desktop/new/')

        def show():
            global cap, tklbl1, tklbl2
            if next(cap, None) is not None:
                start = time.time()
                img, mask, areas, circularities = next(cap)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
                ih, iw = img.shape[:2]
                img = Image.fromarray(img)
                img = img.resize((iw*h//2//ih, h//2))
                tkimgs = ImageTk.PhotoImage(img)
                tklbl1.config(image=tkimgs)
                tklbl1.image = tkimgs

                # for point in points:
                #     x, conf, cls = point
                #     if cls == 0:
                #         colour = 'blue'
                #     elif cls == 1:
                #         colour = 'red'
                #     if conf < 0.3:
                #         colour = 'gray'
                #     plt.scatter(x = x, y = conf, c=colour)
                plt.imshow(mask)
                dplt = plotfunction()
                iw, ih = dplt.size[:2]
                dplt = dplt.resize((iw*h//2//ih, h//2))
                tkimgp = ImageTk.PhotoImage(dplt)
                tklbl2.config(image=tkimgp)
                tklbl2.image = tkimgp

                tklbl6.config(text="%s"%(str(areas)))
                tklbl7.config(text="%s"%(str([round(c, 3) for c in circularities])))
                tklbl11.config(text="%d"%(len(areas)))
                
                
                end = time.time()
                timegui = end-start
                print("GUI showing time: ", timegui)


                root.after(10, show)
        show()
        root.mainloop()

if __name__ == '__main__':
    Example()