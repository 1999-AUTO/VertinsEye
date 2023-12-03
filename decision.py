import time,json
import loguru as logger
from detect import *

def read_card_information():
    with open('card_detect_info.json', 'r') as file:
        data = json.load(file)
    return data

class Card:#创建Card类
    #初始化
    def __init__(self):
        pass

    def extract_card(self,n,classes,level):
        combination_count = 0#合卡次数
        push_card = str(level[n-1]) +" "+str(classes[n-1])#打出的卡牌信息保存

        classes.pop(n-1);level.pop(n-1)#删除抽掉的卡牌信息
        list_lenght = len(classes) - 1
        for i in range(1,list_lenght):
            if i == list_lenght:#实时判断
                break
            #因为抽掉卡牌后可能会出现多次合卡情况，需要循环判定，循环条件不成立会继续往下执行后续代码
            while classes[i-1] == classes[i] and level[i-1] == level[i]:#卡牌的旁边一项类型和等级相同触发合卡条件为True
                classes.pop(i);level.pop(i)#自然合卡时会合并，删掉一张卡的信息
                level[i-1] = level[i-1] + 1#合并的卡牌等级加一
                list_lenght -= 1
                combination_count += 1#合卡次数+1

        return (classes,level,push_card,combination_count)
    """
    enumerate_all_possibilities列举所有卡牌组合函数
    传参说明
        ownership_list---最开始识别到的卡牌所属人物列表
        card_level_list---最开始识别到的卡牌等级列表
    返回
        operation_sequence---所有卡牌组合的出牌顺序
        card_order---所有卡牌组合的卡牌顺序
    """
    def enumerate_all_possibilities(self,ownership_list,card_level_list):
        old_ownership_list = []
        old_card_level_list = []
        operation_sequence = [];card_order = []
        old_ownership_list.append(ownership_list.copy())
        old_card_level_list.append(card_level_list.copy())
        for i in range(1,len(ownership_list)+1):
            information = card.extract_card(i,ownership_list,card_level_list)
            push_card1 = information[2]
            score1 = information[3]

            old_ownership_list.append(ownership_list.copy())
            old_card_level_list.append(card_level_list.copy())
            for j in range(1,len(ownership_list)-1):
                information = card.extract_card(j, ownership_list, card_level_list)
                push_card2 = information[2]
                score2 = information[3]

                old_ownership_list.append(ownership_list.copy())
                old_card_level_list.append(card_level_list.copy())

                for k in range(1,len(ownership_list)-1):
                    information = card.extract_card(k, ownership_list, card_level_list)

                    push_card3 = information[2]
                    score3 = information[3]

                    ownership_list = old_ownership_list[2].copy()
                    card_level_list = old_card_level_list[2].copy()

                    operation_sequence_string = str(i) +" "+ str(j) +" "+ str(k)
                    operation_sequence.append(operation_sequence_string)

                    score_sum = score1+score2+score3
                    card_order.append([push_card1,push_card2,push_card3,score_sum])

                ownership_list = old_ownership_list[1].copy()
                card_level_list = old_card_level_list[1].copy()

            ownership_list = old_ownership_list[0].copy()
            card_level_list = old_card_level_list[0].copy()

        return (operation_sequence,card_order)



    def choose_best(self,tendency):
        pass

if __name__ == "__main__":
    global card
    #创建卡牌检测日志,文件大小达到10MB时进行轮转
    logger.add("log/running .log",rotation="10 MB")
    logger.info("Program start")
    image = screenshot()

    #初始化推理设备
    VertinsEye_Card = VertinsEye("cpu", "/home/gavin/VertinsEye/yolov5","/home/gavin/VertinsEye/exp8/weights/best.pt")
    VertinsEye_Card.detect_card(image,640,0.8)

    card = Card()
    card_data = read_card_information()
    level = [];classes = []
    for i in range(0,len(card_data)):
        level.append(card_data[i]['level'])
        classes.append(card_data[i]['class'])
    print(level,classes)

    print("-"*44)

    print(card.enumerate_all_possibilities(classes,level))
    
    

    

