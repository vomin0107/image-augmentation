import numpy as np
import time
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import xml.etree.ElementTree as ET
import os
from pascal_voc_writer import Writer

AUGMENTATION_NUMBER = 8  # How many times data increased
NUMBER_OF_FILES = 10000  # 50-images and 50-annotations per step


def read_annotation(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bounding_box_list = []

    file_name = root.find('filename').text
    for obj in root.iter('object'):

        object_label = obj.find("name").text
        for box in obj.findall("bndbox"):
            x_min = int(box.find("xmin").text)
            y_min = int(box.find("ymin").text)
            x_max = int(box.find("xmax").text)
            y_max = int(box.find("ymax").text)

        bounding_box = [object_label, x_min, y_min, x_max, y_max]
        bounding_box_list.append(bounding_box)

    return bounding_box_list, file_name


def read_train_dataset(img_xml_list, index_read, length):
    images_read = []
    annotations_read = []

    flag_read = False
    for i in range(index_read * 100, (index_read + 1) * 100):
        if i < length:
            if 'jpg' in img_xml_list[i].lower() or 'jpeg' in img_xml_list[i].lower() or 'png' in img_xml_list[i].lower():
                images_read.append(cv2.imread(image_path + img_xml_list[i], 1))
                annotation_file = img_xml_list[i].replace(img_xml_list[i].split('.')[-1], 'xml')
                bounding_box_list, file_name = read_annotation(image_path + annotation_file)
                annotations_read.append((bounding_box_list, annotation_file, file_name))
        else:
            flag_read = True

    images_read = np.array(images_read, dtype=object)

    return images_read, annotations_read, flag_read

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

augmented_dir = './auged/'
img_path = './'
xml_path = 'xml/'
image_path = './'


if __name__ == "__main__":
    start = time.time()
    image_xml_list = os.listdir(image_path)
    ''' A simple and common augmentation sequence '''
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips 좌우반전 // 상하반전은 Flipud
        iaa.Crop(percent=(0, 0.1)),  # random crops 크롭(확대)
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 1))  # 흐리게
        ),
        # Add a value to all pixels in an image.
        iaa.Add((-50, 50)),  # 모든 픽셀 수치를 변경
        # Apply random four point perspective transformations to images.
        iaa.PerspectiveTransform(scale=(0.01, 0.1)),  # keep_size=False),  # 네 점을 임의의 위치로 이동(끌기), 크롭이랑 비슷한 결과를 낼 수도 있음.
        # # Fill one or more rectangular areas in an image using a fill mode.
        iaa.Cutout(nb_iterations=2),  # 랜덤으로 2개의 사각형을 잘라냄 (기본값 20%), size=0.2
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),  # 대비
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # 가우시안노이즈
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.7, 1.3), per_channel=0.3),  # 모든 픽셀 수치를 변경
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(  # 이미지 변환
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # resize
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 이미지 이동
            rotate=(-20, 20),  # 회전
            shear=(-8, 8)  # 기울이기
        ),
        # Remove coordinate-based augmentables exceeding an out of image fraction.
        iaa.RemoveCBAsByOutOfImageFraction(0.5)  # 원본 이미지가 일정 이상 잘릴경우 annotate 취소
    ], random_order=True)  # apply augmenters in random order

    file_log = open('log_aug.txt', 'w')
    create_folder(augmented_dir + img_path)
    create_folder(augmented_dir + xml_path)

    for total_index in range(NUMBER_OF_FILES):
        print('step', total_index + 1)
        images, annotations, flag = read_train_dataset(image_xml_list, total_index, len(image_xml_list))

        for index in range(len(images)):
            image = images[index]
            boxes = annotations[index][0]
            copy_boxes = []

            ia_bounding_boxes = []
            for box in boxes:
                ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))
            bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

            # Example batch of images.
            copy_images = np.array(
                [image for _ in range(AUGMENTATION_NUMBER)],
                dtype=np.uint8
            )
            for i in range(AUGMENTATION_NUMBER):
                copy_boxes.append(bbs)

            # do augmentation and draw augmentation result
            images_aug, boxes_aug = seq(images=copy_images, bounding_boxes=copy_boxes)

            for idx in range(AUGMENTATION_NUMBER):
                image = images_aug[idx]
                box = boxes_aug[idx]
                image_boxed = box.draw_on_image(image, size=10, color=[0, 0, 255])

                new_image_file = annotations[index][2].replace('.', '_' + str(idx) + '.')
                cv2.imwrite(augmented_dir + img_path + new_image_file, image)

                h, w = np.shape(image)[0:2]
                voc_writer = Writer(new_image_file, w, h)

                for i in range(len(box.bounding_boxes)):
                    bb_box = box.bounding_boxes[i]
                    x_min = int(bb_box.x1)
                    x_max = int(bb_box.x2)
                    y_min = int(bb_box.y1)
                    y_max = int(bb_box.y2)
                    x_center = (x_min + x_max) / 2
                    y_center = (y_min + y_max) / 2
                    if 0 <= x_center <= w and 0 <= y_center <= h:
                        if x_min < 0:
                            x_min = 0
                        if y_min < 0:
                            y_min = 0
                        if x_max > w:
                            x_max = w
                        if y_max > h:
                            y_max = w
                        voc_writer.addObject(boxes[i][0], x_min, y_min, x_max, y_max)

                voc_writer.save(augmented_dir + xml_path + annotations[index][1].replace('.', '_' + str(idx) + '.'))
            log = str(total_index * 50 + index + 1) + '/' + str(int(len(image_xml_list) / 2)) + ' - '\
                  + new_image_file + ' - ' + str(round(time.time() - start, 0)) + 'sec\n'
            file_log.write(log)
            print(log)
        if flag:
            print('done')
            file_log.close()
            exit(0)