import numpy as np
import xmltodict


def parse_xml(xml_path):
    """解析 xml 文件以返回注释边界框的坐标"""
    # 打开文件
    with open(xml_path, 'rb') as f:
        xml_dict = xmltodict.parse(f)  # 解析xml文件
        # print(xml_dict)
        bndboxs = list()
        objects = xml_dict["annotation"]["object"]  # 有多少个物体对象
        # print(objects)

        if isinstance(objects, list):  # 如果是一个列表
            # 表示有多个对象
            for obj in objects:
                obj_name = obj["name"]  # 对象名称
                difficult = obj["difficult"]  # 是否容易识别
                if "car".__eq__(obj_name) and difficult != 1:
                    # 如果为car 且 好识别
                    bndbox = obj["bndbox"]
                    bndboxs.append((int(bndbox["xmin"]), int(bndbox["ymin"]), int(bndbox["xmax"]), int(bndbox["ymax"]),))
        elif isinstance(objects, dict):
            # 不是一个列表，只有一个对象
            obj_name = objects["name"]
            difficult = objects["difficult"]
            if "car".__eq__(obj_name) and difficult != 1:
                bndbox = objects["bndbox"]
                bndboxs.append((int(bndbox["xmin"]), int(bndbox["ymin"]), int(bndbox["xmax"]), int(bndbox["ymax"])))
        else:
            pass
        # print(bndboxs)

        return np.array(bndboxs)

