import os
import cv2,json
import torch,time
import numpy as np
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
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # opencv读取图片为BGR格式，需要转换为RGB格式
        torch_model = self.torch_model.to(self.device)
        results = torch_model(input_image, size=size)#推理
        card_info_list = []  # 用于存储所有卡牌信息的列表

        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        
        logger.info(f"card detect: {results}")#输出卡牌检测日志信息
        try:  # 尝试
            xyxy = results.pandas().xyxy[0].values
            xmins, ymins, xmaxs, ymaxs, class_list, confidences = xyxy[:, 0], xyxy[:, 1],\
                                                                    xyxy[:, 2], xyxy[:, 3], xyxy[:,5], xyxy[:, 4]
            for xmin, ymin, xmax, ymax, class_l, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
                if conf >= 0.8:

                    level_area = image[int(ymin)-45:int(ymax)-232,int(xmin):int(xmax)]
                    Color_lower = np.array([3, 100, 100])
                    Color_upper = np.array([30, 255, 255])

                    hsv = cv2.cvtColor(level_area, cv2.COLOR_BGR2HSV)#HSV颜色设置
                    mask = cv2.inRange(hsv, Color_lower, Color_upper)#提取在HSV范围内的图像
                    
                    level_contours = []
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#识别轮廓
                    if len(contours) > 0:
                        for i in range(0,len(contours)):
                            #大招卡等级为0，大招卡的类别字符串中包含3
                            if cv2.contourArea(contours[i]) >= 80 and str(get_id(class_l)) not in "3":
                                level_contours.append(contours[i])
                    else:
                        logger.warning("Level Error")
                    print(get_id(class_l),len(level_contours))

                    
                    #整合单个卡牌信息
                    card_info = {
                        "class": get_id(class_l),
                        "position": [int(xmin), int(ymin), int(xmax), int(ymax),int(class_l)],
                        "level" : len(level_contours),
                    }
                    
                    
                    card_info_list.append(card_info)
                    cv2.imshow("mask",mask)
                    cv2.waitKey(0)
                    
            with open('card_detect_info.json', 'w') as file:
                json.dump(card_info_list, file, indent=4)
            logger.info("Detect card successfully")

        except Exception as e:
            logger.exception(f"Detect card failed: {e}")
    def detect_enemy(self,image,size):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        torch_model = self.torch_model.to(self.device)
        results = torch_model(input_image, size=size)#推理
        card_info_list = []  # 用于存储所有卡牌信息的列表

        # cv2.imshow("image",image)
        # cv2.waitKey(0)
        
        logger.info(f"card detect: {results}")#输出卡牌检测日志信息
        try:  # 尝试
            xyxy = results.pandas().xyxy[0].values
            xmins, ymins, xmaxs, ymaxs, class_list, confidences = xyxy[:, 0], xyxy[:, 1],\
                                                                    xyxy[:, 2], xyxy[:, 3], xyxy[:,5], xyxy[:, 4]

        except:
            pass


        
    
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
    now = time.time()
    #创建卡牌检测日志,文件大小达到10MB时进行轮转
    logger.add("detect.log",rotation="10 MB")
    
    logger.info("Program start")

    image = screenshot()

    #初始化推理设备
    VertinsEye_Card = VertinsEye("cpu", "/home/gavin/VertinsEye/yolov5","/home/gavin/VertinsEye/exp8/weights/best.pt")
    
    #exp3/best.pt为临时文件，用于识别敌方角色
    VertinsEye_Enemy = VertinsEye("cpu","/home/gavin/VertinsEye/yolov5","/home/gavin/VertinsEye/exp3/weights/best.pt")


     cards = VertinsEye_Card.detect_card(image,640)

    
    print(time.time()-now)
    
