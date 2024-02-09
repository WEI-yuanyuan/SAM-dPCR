import cellpose
from cellpose import models
from cellpose import io
from cellpose.io import imread
import os
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries 
from skimage.filters import threshold_otsu

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pylab import *
import cv2
from io import BytesIO
from threading import Thread
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
import argparse

from visualize import *
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

matplotlib.use('TkAgg')


def getfiles(folder=None):
    if folder is None:
        folder = './data/test/input'
    files = os.listdir(folder)


    for file in files:
        if file.endswith((".png", ".jpg", "tif")):
            yield(file)

def plotfunction(input_file_path):
    # Taken from plot.py:
    with open(input_file_path) as f:
        data = np.array([line.split() for line in f.readlines()[1:]])
        areas = data[:, 0].astype(float)
        ious = data[:, 1].astype(float)
        stability_scores = data[:, 2].astype(float)
        classifications = data[:, 3]

        # Convert area to diameter
        diameters = 2 * np.sqrt(areas / np.pi)

        # Negate the diameter for negative classifications
        diameters[classifications == 'Negative'] *= -1

        # Plot 3D scatter plot of diameters, IoUs, and stability scores
        scatterFig = plt.figure(figsize=(8, 6))
        ax = scatterFig.add_subplot(111, projection='3d')
        # Set IoU and Stability Score axes to 0-1
        ax.set_ylim(np.min(ious), 1)
        ax.set_zlim(np.min(stability_scores), 1)
        # Set the background to white
        # Set the background to white
        scatterFig.patch.set_facecolor('white')
        ax.set_facecolor('white')  # This sets the background of the plot itself

        # Remove the panes (sides of the 3d box)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Set the linewidth of the gridlines
        ax.xaxis._axinfo["grid"]['linewidth'] = 0.5
        ax.yaxis._axinfo["grid"]['linewidth'] = 0.5
        ax.zaxis._axinfo["grid"]['linewidth'] = 0.5
        # Define custom colors
        colors = {'Positive': '#FF1111', 'Negative': '#1010FF'}

        # Plot each class with a different color
        for label in np.unique(classifications):
            indices = np.where(classifications == label)
            ax.scatter(diameters[indices], ious[indices], stability_scores[indices], color = colors[label], label=label, s=50, alpha=None)

        # Labeling
        ax.set_xlabel('Diameter/(pixels)', fontdict={'fontname': 'Arial', 'fontsize': 15}, labelpad=10)
        ax.set_ylabel('Predicted IoU', fontdict={'fontname': 'Arial', 'fontsize': 15}, labelpad=10)
        ax.set_zlabel('Stability score', fontdict={'fontname': 'Arial', 'fontsize': 15}, labelpad=10)
        ax.legend(prop = {'family': 'Arial'}, loc='upper left')

        # Adjust the aspect ratio
        ax.set_box_aspect([np.ptp(diameters), 40, 40])

        # Save the 3D scatter plot to buffer and display on GUI
        buf = BytesIO()
        scatterFig.savefig(buf, format='png', dpi=300)
        plt.close(scatterFig)
        print(f"Saved: {output_file_path}_3d_scatter.png")
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
        
        
        global cap, inputDirectory, outputDirectory, threshold_factor, mask_generator
        
        # From main.py:

        # Adjust these paths as needed
        inputDirectory = 'data/test/input'
        outputDirectory = 'data/test/output'
        threshold_factor = 1.2
        sys.path.append("..")
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        device = "cpu" # Change this to "cuda" to use the GPU
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)

        cap = getfiles(folder=inputDirectory)

        def show():
            global cap, tklbl1, tklbl2, inputDirectory, outputDirectory, threshold_factor, mask_generator
            if next(cap, None) is not None:
                start = time.time()
                filename = next(cap)

                # From visualize.py:

                input_file_path = os.path.join(inputDirectory, filename)
                file_base_name = os.path.splitext(filename)[0]
                output_file_path = os.path.join(outputDirectory, file_base_name + "_DL.png")

                targetImage = cv2.imread(input_file_path)
                print(f"Processing: {filename}")

                hsv_targetImage = cv2.cvtColor(targetImage, cv2.COLOR_BGR2HSV)
                clahe = cv2.createCLAHE(tileGridSize=(10,10))
                hsv_targetImage[:, :, 2] = clahe.apply(hsv_targetImage[:, :, 2])
                hsv_targetImage = cv2.merge((hsv_targetImage[:, :, 0], hsv_targetImage[:, :, 1], hsv_targetImage[:, :, 2]))
                img_compensated = cv2.cvtColor(hsv_targetImage, cv2.COLOR_HSV2BGR)

                annotations = mask_generator.generate(img_compensated)
                if len(annotations) == 0:
                    # skip to next file
                    root.after(2,show)

                imgSatu = hsv_targetImage[:, :, 2]
                imgSatu_np = np.array(imgSatu)

                threshold = threshold_otsu(imgSatu_np) * threshold_factor
                print(f"Threshold: {threshold}")

                sorted_anns = sorted(annotations, key=(lambda x: x['area']), reverse=True)
                annotations_to_remove = findFalsyAnnotationsIndex(sorted_anns=sorted_anns, removeArrayList=annotations_to_remove)
                sorted_anns = removeFalsyAnnotations(removeArrayList=annotations_to_remove, sorted_anns=sorted_anns)

                for ann in sorted_anns:
                    uint8_mask = (ann['segmentation'] * 255).astype(np.uint8)
                    _, uint8_mask = cv2.threshold(uint8_mask, 0, 255, cv2.THRESH_BINARY)
                    mean_value = cv2.mean(imgSatu, mask=uint8_mask)[0]

                    bbox = ann['bbox']

                    if mean_value >= threshold:
                        classes = 'Positive'
                    else:
                        classes = 'Negative'

                    ann['class'] = classes
                    targetImage = show_bbox(targetImage, bbox, classes)

                # cv2.imwrite(output_file_path, targetImage)
                # print(f"Saved: {output_file_path}")

                for ann in sorted_anns:
                    areas.append(ann['area'])
                    ious.append(ann['predicted_iou'])
                    s_scores.append(ann['stability_score'])
                    classifications.append(ann['class'])

                # Save the data to a file with the same name as the input file
                output_txt_file_path = os.path.splitext(output_file_path)[0] + ".txt"
                data = np.column_stack((
                    np.array(['area'] + areas),
                    np.array(['iou'] + ious),
                    np.array(['stability_score'] + s_scores),
                    np.array(['classification'] + classifications)
                ))
                np.savetxt(output_txt_file_path, data, fmt='%s')
                print(f"Saved: {output_txt_file_path}")

                ih, iw = targetImage.shape[:2]
                img = Image.fromarray(targetImage)
                img = img.resize((iw*h//2//ih, h//2))
                tkimgs = ImageTk.PhotoImage(img)
                tklbl1.config(image=tkimgs)
                tklbl1.image = tkimgs

                # Display the 3d scatter plot
                # plt.imshow(mask)
                dplt = plotfunction(output_txt_file_path)
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

                # Automatically display next file in folder.
                root.after(10, show)
        show()
        root.mainloop()

if __name__ == '__main__':
    Example()