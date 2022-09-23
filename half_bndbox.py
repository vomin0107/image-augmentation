import glob
import shutil
import xml.etree.ElementTree as ET


xml_files = glob.glob('xml/*.xml')


def indent(elem, level = 0):
    i = "\n" + level * "   "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "   "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def make_new_box(xmin_tag, ymin_tag, xmax_tag, ymax_tag, name_tag, pose_tag, truncated_tag, difficult_tag, occluded_tag, objects):
    object_tag = ET.Element("object")
    bndbox_tag = ET.Element("bndbox")

    bndbox_tag.append(xmin_tag)
    bndbox_tag.append(ymin_tag)
    bndbox_tag.append(xmax_tag)
    bndbox_tag.append(ymax_tag)
    object_tag.append(bndbox_tag)
    object_tag.append(name_tag)
    object_tag.append(pose_tag)
    object_tag.append(truncated_tag)
    object_tag.append(difficult_tag)
    object_tag.append(occluded_tag)

    objects.append(object_tag)


shutil.copytree("./xml", "./xml_aug/xml")


for xml in xml_files:
    doc = ET.parse(xml)
    root = doc.getroot()
    objects = []
    for tags in root.iter("object"):
        # 기본 정보 구하기
        name = tags.findtext("name")
        pose = tags.findtext("pose")
        truncated = tags.findtext("truncated")
        difficult = tags.findtext("difficult")
        occluded = tags.findtext("occluded")

        # 기존 좌표 구하기
        xmin = tags.find("bndbox").findtext("xmin")
        ymin = tags.find("bndbox").findtext("ymin")
        xmax = tags.find("bndbox").findtext("xmax")
        ymax = tags.find("bndbox").findtext("ymax")
        xmid = int((int(xmax)+int(xmin))/2)
        ymid = int((int(ymax)+int(ymin))/2)

        xmin_left_tag = ET.Element("xmin")
        xmin_right_tag = ET.Element("xmin")
        ymin_up_tag = ET.Element("ymin")
        ymin_bottom_tag = ET.Element("ymin")
        xmax_right_tag = ET.Element("xmax")
        xmax_left_tag = ET.Element("xmax")
        ymax_up_tag = ET.Element("ymax")
        ymax_bottom_tag = ET.Element("ymax")
        name_tag = ET.Element("name")
        pose_tag = ET.Element("pose")
        truncated_tag = ET.Element("truncated")
        difficult_tag = ET.Element("difficult")
        occluded_tag = ET.Element("occluded")

        xmin_right_tag.text = str(xmid)
        xmax_right_tag.text = str(xmax)
        xmin_left_tag.text = str(xmin)
        xmax_left_tag.text = str(xmid)
        ymin_up_tag.text = str(ymid)
        ymax_up_tag.text = str(ymax)
        ymin_bottom_tag.text = str(ymin)
        ymax_bottom_tag.text = str(ymid)
        name_tag.text = name
        pose_tag.text = pose
        truncated_tag.text = truncated
        difficult_tag.text = difficult
        occluded_tag.text = occluded

        make_new_box(xmin_left_tag, ymin_bottom_tag, xmax_left_tag, ymax_up_tag, name_tag, pose_tag, truncated_tag, difficult_tag, occluded_tag, objects)
        make_new_box(xmin_right_tag, ymin_bottom_tag, xmax_right_tag, ymax_up_tag, name_tag, pose_tag, truncated_tag, difficult_tag, occluded_tag, objects)
        make_new_box(xmin_left_tag, ymin_up_tag, xmax_right_tag, ymax_up_tag, name_tag, pose_tag, truncated_tag, difficult_tag, occluded_tag, objects)
        make_new_box(xmin_left_tag, ymin_bottom_tag, xmax_right_tag, ymax_bottom_tag, name_tag, pose_tag, truncated_tag, difficult_tag, occluded_tag, objects)

    for obj in objects:
        root.append(obj)

    indent(root)
    ET.dump(root)
    doc.write("xml_aug/" + xml, encoding="utf-8", xml_declaration=False)
