import pickle
import cv2
import os
import os.path
from os import path as p
from pathlib import Path


def xyxy_to_xywh(box):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    w = x2 - x1
    h = y2 - y1
    return [x1, y1, w, h]


def main():
    this_file_path = Path(os.path.dirname(os.path.abspath(__file__)))

    FILE = this_file_path.__str__() + "/results.pkl"
    with open(FILE, 'rb') as fi:
        results = pickle.load(fi)

    classes = results['classes']
    class_filter = ["person"]

    background_color = (255, 255, 255, 0)

    # util method to convert the detectron2 box format

    # Cut out masks
    print("Extracting objects...")
    index = 0
    for path in results['instances']:
        for i in range(len(results['instances'][path].pred_masks)):
            mask_class = classes[results['instances'][path].pred_classes[i]]

            # Check if class is in filter
            if mask_class in class_filter:
                # make everything transparent except the mask
                mask = results['instances'][path].pred_masks[i]
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                x = 0
                y = 0
                for line in mask:
                    for column in line:
                        if not column:
                            img[x, y] = background_color
                        y += 1
                    y = 0
                    x += 1

                # Cropping image to the size of the objects bounding box
                box = results['instances'][path].pred_boxes[i]
                box = box.tensor.numpy()[0]
                box = xyxy_to_xywh(box)
                img = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]

                # Save image
                new_img_path = this_file_path.__str__() + '/' + mask_class + "_" + str(index) + "_" + str(i) + ".png"
                print(new_img_path)
                cv2.imwrite(new_img_path, img)
        index += 1


if __name__ == '__main__':
    main()
