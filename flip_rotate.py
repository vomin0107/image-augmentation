import cv2
import glob
import os
import shutil
import numpy as np
import xml.etree.ElementTree as ET


def calculate_point(p1, p2):
    return str(int(p1)-int(p2))


def generate_rotated_image(img, type, name):
    img90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # 시계방향으로 90도 회전
    img180 = cv2.rotate(img, cv2.ROTATE_180)  # 180도 회전
    img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 반시계방향으로 90도 회전
    img_flip_hor = cv2.flip(img, 1)  # 좌우반전
    img_flip_ver = cv2.flip(img, 0)  # 상하반전

    cv2.imwrite('images/converted/'+name+type+'.jpg', img)
    cv2.imwrite('images/converted/'+name+type+'_90.jpg', img90)
    cv2.imwrite('images/converted/'+name+type+'_180.jpg', img180)
    cv2.imwrite('images/converted/'+name+type+'_270.jpg', img270)
    cv2.imwrite('images/converted/'+name+type+'_flip_hor.jpg', img_flip_hor)
    cv2.imwrite('images/converted/'+name+type+'_flip_ver.jpg', img_flip_ver)


def generate_rotated_xml(doc, angle, name):
    doc.getroot().find("filename").text = name+'_default'+angle+'.jpg'
    doc.write('./images/converted/'+name+'_default'+angle+'.xml')
    doc.getroot().find("filename").text = name+'_bright'+angle+'.jpg'
    doc.write('./images/converted/'+name+'_bright'+angle+'.xml')
    doc.getroot().find("filename").text = name+'_dark'+angle+'.jpg'
    doc.write('./images/converted/'+name+'_dark'+angle+'.xml')
    doc.getroot().find("filename").text = name+'_blur'+angle+'.jpg'
    doc.write('./images/converted/'+name+'_blur'+angle+'.xml')


image_files = glob.glob('./images/*.jpg')

try:
    if not os.path.exists('./images/converted/'):
        os.makedirs('./images/converted/')
except OSError:
    print ('Error: Creating directory. ' +  './images/converted/')

num = 0
for image in image_files:
    # print(image)
    file_name = image.split('\\')[-1].split('.')[0]
    xml_name = image.split('.jpg')[0] + '.xml'
    # print(xml_name)
    print(file_name)
    img = cv2.imread(image, cv2.IMREAD_COLOR)

    img_gray = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    bright_val = int((255-img_gray.mean())/1.5)
    dark_val = int(img_gray.mean()/1.5)

    bright_bias = np.ones(img.shape, dtype="uint8") * bright_val
    dark_bias = np.ones(img.shape, dtype="uint8") * dark_val
    img_bright = cv2.add(img, bright_bias)
    img_dark = cv2.subtract(img, dark_bias)

    img_blur= cv2.blur(img, (5, 5), anchor=(-1, -1), borderType=cv2.BORDER_DEFAULT)  # 흐림
    # 커널 크기는 이미지에 흐림 효과를 적용할 크기를 설정합니다. 크기가 클수록 더 많이 흐려집니다.
    # 앵커 포인트는 커널에서의 중심점을 의미합니다. (-1, -1)로 사용할 경우, 자동적으로 커널의 중심점으로 할당합니다.
    # 픽셀 외삽법은 이미지를 흐림 효과 처리할 경우, 영역 밖의 픽셀은 추정해서 값을 할당합니다.

    generate_rotated_image(img, '_default', file_name)
    generate_rotated_image(img_dark, '_dark', file_name)
    generate_rotated_image(img_bright, '_bright', file_name)
    generate_rotated_image(img_blur, '_blur', file_name)

    h, w, c = img.shape

    doc = ET.parse(xml_name)
    root = doc.getroot()
    filename_tag = root.find("filename")
    generate_rotated_xml(doc, '', file_name)

    root = doc.getroot()
    for tags in root.iter("object"):
        # 기존 좌표 구하기
        xmin = tags.find("bndbox").findtext("xmin")
        ymin = tags.find("bndbox").findtext("ymin")
        xmax = tags.find("bndbox").findtext("xmax")
        ymax = tags.find("bndbox").findtext("ymax")

        # 좌표 재생성
        tags.find("bndbox").find("xmin").text = ymin
        tags.find("bndbox").find("ymin").text = calculate_point(w, xmax)
        tags.find("bndbox").find("xmax").text = ymax
        tags.find("bndbox").find("ymax").text = calculate_point(w, xmin)
        # xml 파일에 다시 쓰기
        generate_rotated_xml(doc, '_270', file_name)

    doc = ET.parse(xml_name)
    root = doc.getroot()
    for tags in root.iter("object"):
        # 기존 좌표 구하기
        xmin = tags.find("bndbox").findtext("xmin")
        ymin = tags.find("bndbox").findtext("ymin")
        xmax = tags.find("bndbox").findtext("xmax")
        ymax = tags.find("bndbox").findtext("ymax")

        # 좌표 재생성
        tags.find("bndbox").find("xmin").text = calculate_point(h, ymax)
        tags.find("bndbox").find("ymin").text = xmin
        tags.find("bndbox").find("xmax").text = calculate_point(h, ymin)
        tags.find("bndbox").find("ymax").text = xmax
        # xml 파일에 다시 쓰기
        generate_rotated_xml(doc, '_90', file_name)

    doc = ET.parse(xml_name)
    root = doc.getroot()
    for tags in root.iter("object"):
        # 기존 좌표 구하기
        xmin = tags.find("bndbox").findtext("xmin")
        ymin = tags.find("bndbox").findtext("ymin")
        xmax = tags.find("bndbox").findtext("xmax")
        ymax = tags.find("bndbox").findtext("ymax")

        # 좌표 재생성
        tags.find("bndbox").find("xmin").text = calculate_point(w, xmax)
        tags.find("bndbox").find("ymin").text = calculate_point(h, ymax)
        tags.find("bndbox").find("xmax").text = calculate_point(w, xmin)
        tags.find("bndbox").find("ymax").text = calculate_point(h, ymin)
        # xml 파일에 다시 쓰기
        generate_rotated_xml(doc, '_180', file_name)

    doc = ET.parse(xml_name)
    root = doc.getroot()
    for tags in root.iter("object"):
        # 기존 좌표 구하기
        xmin = tags.find("bndbox").findtext("xmin")
        ymin = tags.find("bndbox").findtext("ymin")
        xmax = tags.find("bndbox").findtext("xmax")
        ymax = tags.find("bndbox").findtext("ymax")

        # 좌표 재생성
        tags.find("bndbox").find("xmin").text = calculate_point(w, xmax)
        tags.find("bndbox").find("ymin").text = ymin
        tags.find("bndbox").find("xmax").text = calculate_point(w, xmin)
        tags.find("bndbox").find("ymax").text = ymax
        # xml 파일에 다시 쓰기
        generate_rotated_xml(doc, '_flip_hor', file_name)

    doc = ET.parse(xml_name)
    root = doc.getroot()
    for tags in root.iter("object"):
        # 기존 좌표 구하기
        xmin = tags.find("bndbox").findtext("xmin")
        ymin = tags.find("bndbox").findtext("ymin")
        xmax = tags.find("bndbox").findtext("xmax")
        ymax = tags.find("bndbox").findtext("ymax")

        # 좌표 재생성
        tags.find("bndbox").find("xmin").text = xmin
        tags.find("bndbox").find("ymin").text = calculate_point(h, ymax)
        tags.find("bndbox").find("xmax").text = xmax
        tags.find("bndbox").find("ymax").text = calculate_point(h, ymin)
        # xml 파일에 다시 쓰기
        generate_rotated_xml(doc, '_flip_ver', file_name)

    num += 1