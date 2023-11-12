import os
import cv2,json
import torch
from pathlib import Path
from loguru import logger

with open('name2idx.json', 'r') as file:
    id_data = json.load(file)

def get_id(card_index):
        for key, value in id_data.items():
            if value == card_index:
                return key
class VertinsEye:
    def __init__(self,device, yolo_path, model_path):
        try:
            self.device = torch.device(device)  # 设置推理设备
            self.torch_model = torch.hub.load(yolo_path, 'custom',
                                    model_path, source='local',
                                    force_reload=True)  # 加载本地yolov5模型(需要修改路径和文件)
            logger.info("The VertinsEye is initialized successfully")
        except Exception as e:
            logger.exception(f"The VertinsEye is initialization failed: {e}")
    
    def detect_card(self,image,size):
        #card_area = image[720:1080,0:2400]
        torch_model = self.torch_model.to(self.device)
        results = torch_model(image, size=size)#推理
        card_info_list = []  # 用于存储所有卡牌信息的列表

        # cv2.imshow("image",image)
        # cv2.waitKey(0)

        logger.info(f"card detect: {results}")#输出卡牌检测日志信息
        try:  # 尝试
            xyxy = results.pandas().xyxy[0].values
            xmins, ymins, xmaxs, ymaxs, class_list, confidences = xyxy[:, 0], xyxy[:, 1],\
                                                                    xyxy[:, 2], xyxy[:, 3], xyxy[:,5], xyxy[:, 4]
            for xmin, ymin, xmax, ymax, class_l, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
                if conf >= 0.5:
                    #整合单个卡牌信息
                    card_info = {
                        "class": get_id(class_l),
                        "position": [int(xmin), int(ymin), int(xmax), int(ymax),int(class_l)],
                        "level" : 1,
                    }
                    card_info_list.append(card_info)

            with open('card_detect_info.json', 'w') as file:
                json.dump(card_info_list, file, indent=4)
            logger.info("Detect card successfully")

        except Exception as e:
            logger.exception(f"Detect card failed: {e}")



        
    
"""
ADB截屏
"""
def screenshot():
    adb_command = "adb shell screencap -p /sdcard/screenshot.png"
    pull_command = "adb pull /sdcard/screenshot.png"#拉取图片

    os.system(adb_command)
    os.system(pull_command)
    image = cv2.imread("screenshot.png")
    #image = cv2.resize(image, (1600, 900), interpolation=cv2.INTER_LINEAR)
    return image


if __name__ == "__main__":
    #创建卡牌检测日志,文件大小达到10MB时进行轮转
    logger.add("detect.log",rotation="10 MB")
    
    logger.info("Program start")

    image = screenshot()

    #初始化推理设备
    VertinsEye_card = VertinsEye("cpu", "/home/gavin/VertinsEye/yolov5",
                                     "/home/gavin/VertinsEye/exp8/weights/best.pt")  # 初始化设置
    cards = VertinsEye_card.detect_card(image,640)
    
    
