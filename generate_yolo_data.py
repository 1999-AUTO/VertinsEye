import cv2,time
from tqdm import tqdm
import numpy,sys,random
import os
sys.path.append(r"cards")#进入cards目录下
from aname import *

card_dict = name_to_english
card_list = []

#卡牌对象类
class Card:
    def __init__(self,card_position_list):
        coefficient = random.uniform(0.8,1.2)
        
        #卡牌高按比例随机缩放
        card_height = 209*coefficient
        card_width = 160*coefficient

        #计算四个点的坐标
        self.top_left = [random.randint(0,round(1600-160*coefficient)),random.randint(0,round(900-209*coefficient))]
        self.low_left = [self.top_left[0],self.top_left[1]+card_height]
        self.top_right = [self.top_left[0]+card_width,self.top_left[1]]
        self.low_right = [self.top_left[0]+card_width,self.top_left[1]+card_height]

        #取整数
        self.top_left = [round(pos) for pos in self.top_left]
        self.low_left = [round(pos) for pos in self.low_left]
        self.top_right = [round(pos) for pos in self.top_right]
        self.low_right = [round(pos) for pos in self.low_right]

       
#生成卡牌函数
def Generate_Card(num,background_image):
    card_position_list = [];cards = []

    #生成num个卡牌实例对象
    for i in range(num):
        cards.append(Card(card_position_list))

    for card in cards:
         #如果卡牌坐标列表长度为0，则为首次生成，可直接添加信息
        if len(card_position_list) == 0:
            card_position_list.append([card.top_left,card.low_left,card.top_right,card.low_right])
            cv2.rectangle(background_image,(card.top_left),(card.low_right), (32,43,232), 10)#画出生成的第一张卡牌
        if True:
            #生成的卡牌数量
            for k in range(0,len(card_position_list)):
                #每个卡牌的四个坐标点都需要判断是否重合
                for dot in [card.top_left,card.low_left,card.top_right,card.low_right]:
                    dot_x_pos = dot[0]
                    dot_y_pos = dot[1]
                    check_bool = card_position_list[k][0][0] < dot_x_pos < card_position_list[k][2][0] and card_position_list[k][0][1] < dot_y_pos < card_position_list[k][3][1]
                    #给每个卡牌的左上角标上红点
                    cv2.circle(background_image,(card_position_list[k][0][0],card_position_list[k][0][1]),10,(32,43,232),-1)
                    if check_bool == True:#如果出现覆盖情况
                        return False#返回假，即重新生成一轮
         
            #添加信息 
            card_position_list.append([card.top_left,card.low_left,card.top_right,card.low_right])
            
    #返回
    return card_position_list
                        
    
        
    
if __name__ == "__main__":
    
    try:
        os.makedirs(r'card_models/images')
        os.makedirs(r'card_models/labels')
    except:
        pass
    classes_file = open(r"card_models/labels/classes.txt", "w", encoding='utf-8')
    card_amount = 0
    for name_key in card_dict:
        for i in range(1,4):
            
            card_string = str(card_dict[name_key]) + str(i)
            classes_file.writelines(card_string+"\n")
            card_list.append(card_string)
            card_amount += 1


    image_counts = 0
    while image_counts < 10:#生成十张图片
        background = cv2.imread("background/1.jpg")
        background = cv2.resize(background,(1600,900))
        card_position_list = Generate_Card(6,background)#平均每张图片放六张卡牌

        if card_position_list != False:#如果生成的图片不存在覆盖现象
            #画框
            for card in card_position_list:
                background = cv2.rectangle(background,(card[0][0],card[0][1]),(card[3][0],card[3][1]), (0, 255, 0), 5)
            image_counts += 1
            #显示
            cv2.imshow("img",background)
            cv2.waitKey(0)

    