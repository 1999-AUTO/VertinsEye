import cv2
import time
from tqdm import tqdm
import numpy,sys,random
import os
import glob
import json
from loguru import logger

from cards.aname import *

from typing import Optional

Image = numpy.ndarray

card_dict = name_to_english
card_list = []

#卡牌对象类
class Card:
    def __init__(self,card_position_list):
        coefficient_h = random.uniform(0.8,1.2)
        coefficient_w = random.uniform(0.8,1.2)
        
        #卡牌高按比例随机缩放
        card_height = 209*coefficient_h
        card_width = 160*coefficient_w

        #计算四个点的坐标
        self.top_left = [random.randint(0,round(1600-card_width)),random.randint(0,round(900-card_height))]
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
    
    #判断是否有重合
    for i, card1 in enumerate(cards):
        for j, card2 in enumerate(cards):
            if i == j:
                continue
            for x, y in [card2.top_left,card2.low_left,card2.top_right,card2.low_right]:
                if card1.top_left[0] < x < card1.top_right[0] and card1.top_left[1] < y < card1.low_left[1]:
                    return False
    card_position_list = [[card.top_left,card.low_left,card.top_right,card.low_right] for card in cards]
    #返回
    return card_position_list

def get_backgrounds(target_size: Optional[tuple[int, int]] = (1600, 900)):
    fnames = glob.glob("background/*.jpg")
    results: list[Image] = []
    for fname in fnames:
        img = cv2.imread(fname)
        assert img is not None
        results.append(img)
    if target_size is not None:
        results = [cv2.resize(img, target_size) for img in results]
    return results

def get_skills():
    results: dict[str, Image] = {}
    for name in card_reflect:
        fname = f"cards/{name}.png"
        img = cv2.imread(fname)
        if img is None:
            logger.warning(f"{fname} does not exist")
            continue
        results[name] = img
    return results

if __name__ == "__main__":
    random.seed(42)
    for dataset_type in ['train', 'val', 'test']:
        os.makedirs(f'card_models/images/{dataset_type}', exist_ok=True)
        os.makedirs(f'card_models/labels/{dataset_type}', exist_ok=True)
    classes_file = open(r"card_models/labels/classes.txt", "w", encoding='utf-8')
    card_amount = 0
    for name_key in card_dict:
        for i in range(1,4):
            
            card_string = str(card_dict[name_key]) + str(i)
            classes_file.writelines(card_string+"\n")
            card_list.append(card_string)
            card_amount += 1
            
    bgrs = get_backgrounds()
    skills = get_skills()
    skill_names = list(skills.keys()) # for fast random access
    name2idx = {name: idx for idx, name in enumerate(skill_names)}
    json.dump(name2idx, open("name2idx.json", "w", encoding='utf-8'), ensure_ascii=False, indent=4)

    tot_train = 1000
    tot_val = tot_train * 0.1
    tot_test = tot_train * 0.1
    tot_all = tot_train + tot_val + tot_test
    image_counts = 0
    iter_bar = tqdm(total=tot_all)
    while image_counts < tot_all:#生成十张图片
        img = random.choice(bgrs).copy()
        card_num = random.randint(4, 10) #每张图片放置4-10张卡牌
        card_position_list = Generate_Card(card_num, numpy.zeros_like(img))#平均每张图片放六张卡牌

        if card_position_list:#如果生成的图片不存在覆盖现象
            #画框
            # for card in card_position_list:
            #     img = cv2.rectangle(img, (card[0][0],card[0][1]),(card[3][0],card[3][1]), (0, 255, 0), 5)
            
            labels = []
            for pos in card_position_list:
                name = random.choice(card_list)
                skill_img = skills[name]
                cx, cy = (pos[0][0] + pos[3][0]) / 2, (pos[0][1] + pos[3][1]) / 2
                w, h = pos[3][0] - pos[0][0], pos[3][1] - pos[0][1]
                skill_img = cv2.resize(skill_img, (w, h))
                img[pos[0][1]:pos[3][1], pos[0][0]:pos[3][0]] = skill_img
                
                cx, cy = cx / img.shape[1], cy / img.shape[0]
                w, h = w / img.shape[1], h / img.shape[0]
                labels.append((name2idx[name], cx, cy, w, h))
            
            image_counts += 1
            #显示
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
            
            if image_counts <= tot_train:
                dataset_type = 'train'
            elif image_counts <= tot_train + tot_val:
                dataset_type = 'val'
            else:
                dataset_type = 'test'
            
            cv2.imwrite(f"card_models/images/{dataset_type}/{image_counts}.jpg", img)
            # write label
            with open(f"card_models/labels/{dataset_type}/{image_counts}.txt", "w", encoding='utf-8') as f:
                for label in labels:
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
            
            iter_bar.update(1)