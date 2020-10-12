#Nikesh Ghimire
#Accuracy check

#Importing modules
import os
import json
import ast

def compileInterval(path):

    for i in os.listdir(path):
        if i.endswith('.JSON'):
            break

    with open(path + i) as inFile:
        data = json.load(inFile)

    s_list = [[round(eval(k),2), round(v,2)] for k, v in data['silences'].items()] 
    
    return s_list

def checkPlay(int_list, currFrame):

    currTime = currFrame/30

    for i in int_list:
        if currTime > i[0] and currTime < i[1]:
            return False
        else:
            pass
    
    return True

def getOutput(path):

    in_file = open(path, 'r')
    results = in_file.read()

    return eval(results)
    # currTime = currFrame/30

def accOut(lst1, lst2):

    accList = []

    for i in lst2:
        time = i[0] * 1/30
        if checkPlay(lst1 , i[0]) and i[1] == 'playing':
            accList.append(i)
        elif not checkPlay(lst1, i[0]) and i[1] == 'n_playing':
            accList.append(i)

    return str(round(len(accList)/len(lst2) * 100, 2)) + ' % accuracy'

if __name__ == '__main__':

    label_list = compileInterval('cnn_set/guitar_2/vid_3/')
    # print(label_list[0])
    pred_list = getOutput('output/test/IMG_0029_op_pred_m2.txt')
    # print(pred_list[0])

    print(accOut(label_list, pred_list))
