# Import module
import tkinter as tk
from tkinter import Tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import numpy as np

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
import detect2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
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


# Create object
root = Tk()
root.bind("<Escape>", lambda e: root.quit())

# Adjust size
# root.geometry("800x800")
root.attributes('-fullscreen', False)

# Load YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
opt = parse_opt()
# # Load model
# device = select_device(opt.device)
# model = DetectMultiBackend(opt.weights, device=device, dnn=opt.dnn, data=opt.data, fp16=opt.half)


def main_page():
    def web_cam_func():
        def go_back_to_main_frame():
            cap.release()
            display_frame1.place_forget()
            display_frame2.place_forget()
            back_frame.place_forget()
            main_frame.place(relx=0.5, rely=0.5, width=500, height=500, anchor=tk.CENTER)

        main_frame.place_forget()
        width, height = 700, 700
        cap = detect2.run(**vars(opt))
        print(cap)
        # cap = cv2.VideoCapture(opt.source)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        display_frame1 = tk.Frame(root)
        display_frame1.place(relx=0.2, rely=0.5, width=600, height=700, anchor=tk.CENTER)

        display_frame1_label = tk.Label(display_frame1, text="Original video", font=('Rockwell', 16), bg="yellow")
        display_frame1_label.pack(side=tk.TOP)

        display_frame2 = tk.Frame(root)
        display_frame2.place(relx=0.8, rely=0.5, width=600, height=700, anchor=tk.CENTER)

        display_frame2_label = tk.Label(display_frame2, text="Detection", font=('Rockwell', 16), bg="yellow")
        display_frame2_label.pack(side=tk.TOP)

        back_frame = tk.Frame(root)
        back_frame.pack(side=tk.TOP, anchor=tk.NW)
        back_button = tk.Button(back_frame, text="BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
        back_button.pack()

        lmain = tk.Label(display_frame1)
        lmain1 = tk.Label(display_frame2)
        lmain.place(x=0, y=100, width=600, height=600)
        lmain1.place(x=0, y=100, width=600, height=600)
        n=0
        def show_frame(n):
            #print(i)
            frame = n
            #_, frame = cap.read()
            frame2 = frame#cv2.flip(frame, 1)
            #frame = frame.transpose(2,0,1)
            #frame = np.expand_dims(frame, 0)
            #print(frame.shape)
            cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
            # cv2image=np.expand_dims(cv2image, 0)
            # print(cv2image.shape)
            img = Image.fromarray(cv2image)

            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)

            # Perform inference
            #results = model(torch.Tensor(frame))
            #print(len(results))

            # if len(results):
            #     # Parse results and draw bounding boxes
            #     for *xyxy, conf, cls in reversed(results):
            #         if conf > 0.5:
            #             label = f'{model.names[int(cls)]} {conf:.2f}'
            #             cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
            #             cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                         (255, 0, 0), 2)

            # frame3 = cv2.flip(frame, 1)
            frame3 = frame
            cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
            img2 = Image.fromarray(cv2image2)

            imgtk2 = ImageTk.PhotoImage(image=img2)

            lmain1.imgtk = imgtk2
            lmain1.configure(image=imgtk2)

            lmain.after(1, show_frame(next(cap)))
        show_frame(next(cap))
        # for i in cap:
        #     show_frame(i)
        #     time.sleep(1)
        #     continue

    def upload_vid_func():
        def browse_file():
            def run_yolov5_on_video():
                def go_back_to_main_frame():
                    cap.release()
                    display_frame1.place_forget()
                    display_frame2.place_forget()
                    back_frame.place_forget()
                    main_frame.place(relx=0.5, rely=0.5, width=500, height=500, anchor=tk.CENTER)

                browse_frame.place_forget()
                width, height = 700, 700
                #  print(file_path)
                cap = cv2.VideoCapture(file_path)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                display_frame1 = tk.Frame(root)
                display_frame1.place(relx=0.2, rely=0.5, width=600, height=700, anchor=tk.CENTER)

                display_frame1_label = tk.Label(display_frame1, text="Original video", font=('Rockwell', 16),
                                                bg="yellow")
                display_frame1_label.pack(side=tk.TOP)

                display_frame2 = tk.Frame(root)
                display_frame2.place(relx=0.8, rely=0.5, width=600, height=700, anchor=tk.CENTER)

                display_frame2_label = tk.Label(display_frame2, text="Detection", font=('Rockwell', 16), bg="yellow")
                display_frame2_label.pack(side=tk.TOP)

                back_frame = tk.Frame(root)
                back_frame.pack(side=tk.TOP, anchor=tk.NW)
                back_button = tk.Button(back_frame, text="BACK", font=("Rockwell", 12), command=go_back_to_main_frame)
                back_button.pack()

                lmain = tk.Label(display_frame1)
                lmain1 = tk.Label(display_frame2)
                lmain.place(x=0, y=100, width=600, height=600)
                lmain1.place(x=0, y=100, width=600, height=600)

                def show_frame():

                    _, frame = cap.read()
                    # frame2 = cv2.flip(frame, 1)
                    frame2 = frame
                    cv2image = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
                    img = Image.fromarray(cv2image)

                    imgtk = ImageTk.PhotoImage(image=img)
                    lmain.imgtk = imgtk
                    lmain.configure(image=imgtk)

                    # Perform inference
                    results = model(frame)

                    # Parse results and draw bounding boxes
                    for *xyxy, conf, cls in results.xyxy[0]:
                        if conf > 0.5:
                            label = f'{model.names[int(cls)]} {conf:.2f}'
                            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])),
                                          (255, 0, 0), 2)
                            cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 0), 2)

                    # frame3 = cv2.flip(frame, 1)
                    frame3 = frame
                    cv2image2 = cv2.cvtColor(frame3, cv2.COLOR_BGR2RGBA)
                    img2 = Image.fromarray(cv2image2)

                    imgtk2 = ImageTk.PhotoImage(image=img2)

                    lmain1.imgtk = imgtk2
                    lmain1.configure(image=imgtk2)

                    lmain.after(1, show_frame)

                show_frame()

            filename = filedialog.askopenfilename(filetypes=[("video files", "*.*")])
            file_path = os.path.abspath(filename)

            run_yolov5_on_video()

        main_frame.place_forget()

        browse_frame = tk.Frame(root, bg="orange")
        browse_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        browse_button = tk.Button(browse_frame, text="Browse", font=("Rockwell", 20), bg="Yellow", fg="white",
                                  command=browse_file)
        browse_button.pack()

    main_frame = tk.Frame(root, bg="orange")

    main_frame.place(relx=0.5, rely=0.5, width=500, height=500, anchor=tk.CENTER)

    web_cam = tk.Button(main_frame, text="Web cam", command=web_cam_func, bg="yellow", fg="purple",
                        font=('Rockwell', 18))

    web_cam.place(x=10, y=100)

    upload_vid = tk.Button(main_frame, text="Upload Video", command=upload_vid_func, bg="yellow", fg="purple",
                           font=('Rockwell', 18))

    upload_vid.place(x=300, y=100)

if __name__ == '__main__':
    main_page()

    Title_label = tk.Label(root, text="YOLOv5 Object detection", font=('Rockwell', 20), bg="yellow")
    Title_label.pack(side=tk.TOP)

    Exit_label = tk.Label(root, text="Press excape to quit", font=('Rockwell', 20), bg="yellow")
    Exit_label.pack(side=tk.BOTTOM)

    # Execute tkinter
    root.mainloop()