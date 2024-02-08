import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
import numpy as np
 
matplotlib.use('TkAgg')

def has_multiple_components(mask): # Function to determine if a mask has more than one connected component
    # Apply connected components analysis
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # If there's more than one label (excluding the background), it has multiple components
    return num_labels > 2  # background label is 0, one object would make this 2

def show_bbox(image, bbox, classes, thickness=2): # Function to draw bounding boxes
    if classes == 'Positive':
        color = (0, 0, 255)
    elif classes == 'Negative':
        color = (255, 0, 0)

    x1, y1, width, height = bbox
    x2 = x1 + width
    y2 = y1 + height
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image

def findFalsyAnnotationsIndex(sorted_anns, removeArrayList): # Function to find annotations with multiple components
    for index_to_remove, ann in enumerate(sorted_anns):
        uint8_mask = (ann['segmentation'] * 255).astype(np.uint8)
        if uint8_mask is not None and has_multiple_components(uint8_mask):
            removeArrayList.add(index_to_remove)
    
    return removeArrayList

def removeFalsyAnnotations(sorted_anns, removeArrayList): # Function to remove annotations with multiple components
    for i in sorted(removeArrayList, reverse=True):
        # print(f"Removing: {sorted_anns[i]} at index {i}")
        del sorted_anns[i]
    return sorted_anns

def show(inputDirectory, outputDirectory, threshold_factor, mask_generator): # Function to classify droplet as bright or dark and visualize the results   
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    for filename in os.listdir(inputDirectory):
        areas = []
        ious = []
        s_scores = []
        classifications = []
        annotations_to_remove = set()

        if filename.endswith((".png", ".jpg", "tif")):  # Check for PNG and JPG files (change extensions as needed)
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
            # cv2.imshow('targetImage', hsv_targetImage[:, :, 2])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # cv2.imshow('targetImage', hsv_targetImage[:, :, 1])
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # generate annotations by SAM
            annotations = mask_generator.generate(img_compensated)
            if len(annotations) == 0:
                return False
            
            imgSatu = hsv_targetImage[:, :, 2]
            imgSatu_np = np.array(imgSatu)
            # imgValue_np = cv2.equalizeHist(imgValue_np)
            # cv2.imshow('imgValue', imgValue_np)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            
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

            cv2.imwrite(output_file_path, targetImage)
            print(f"Saved: {output_file_path}")

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
            #add total number of droplets
            diameters=[]
            for i in range(len(areas)):
                diameters.append(np.sqrt(areas[i]/np.pi)*2)
            diametermean=np.mean(diameters)
            diameterstd=np.std(diameters)
            # save diameters to txt file
            new_output_txt_file_path = os.path.join(inputDirectory, file_base_name + "_diameter.txt")
            with open(new_output_txt_file_path, 'a') as f:
                for item in diameters:
                    f.write("%s\n" % item)
                f.close()
            #add mean and variance of area
            areamean=np.mean(areas)
            areavar=np.var(areas)
            total = len(classifications)
            pos = classifications.count('Positive')
            neg = classifications.count('Negative')
            areastd=np.std(areas)
            #create new txt file to inputfile path
            new_output_txt_file_path = os.path.join(inputDirectory, file_base_name + "_stat.txt")
            with open(new_output_txt_file_path, 'a') as f:
                f.write(f"total: {total}\n")
                f.write(f"positive: {pos}\n")
                f.write(f"negative: {neg}\n")
                f.write(f"mean: {areamean}\n")
                f.write(f"variance: {areavar}\n")
                f.write(f"standard deviation: {areastd}\n")
                f.write(f"diameter mean: {diametermean}\n")
                f.write(f"diameter standard deviation: {diameterstd}\n")
                f.close()
            print(f"Saved: {output_txt_file_path}")

