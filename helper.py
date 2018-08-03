import json
import re
import os
import jieba
import string
from zhon.hanzi import punctuation
from pprint import pprint
import numpy as np
import copy
import  threading
import sys
import tensorflow as tf

punc = list(string.punctuation) + list(punctuation) + [' ']
all_set = []

def remove_punc(sentence):
    new_sentence = []
    for word in sentence:
        if word not in punc:
            new_sentence.append(word)
    return new_sentence

def split_sentence(sentence):
    sentence = list(jieba.cut(sentence))
    sentence = remove_punc(sentence)
    return sentence

def find(article, word):
    index = []
    for i in range(len(article)):
        if word == article[i]:
            index.append(i)
    return index


def dfs(index, tmp_list, n):
    global all_set
    if len(tmp_list) == n:
        all_set.append(copy.copy(tmp_list))
        return
    for i in range(len(index)):
        for j in range(len(index[i])):
            if tmp_list == [] or index[i][j] > tmp_list[-1]:
                dfs(index[i+1:], tmp_list+[index[i][j]], n)


def filter_index(index):
    initial_value = min(index[0])
    new_index = [index[0]]
    for i in range(1,len(index)):
        tmp_index = []
        for j in range(len(index[i])):
            if index[i][j] > initial_value:
                tmp_index.append(index[i][j])
        if tmp_index == []:
            return []
        new_index.append(tmp_index)
        initial_value = min(tmp_index)
    return new_index


def find_forward_backward(index, i):
    initial_value = index[i][0]
    foward = []
    foward_init = initial_value
    backward = []
    backward_init = initial_value
    for j in range(i-1,-1,-1):
        flag = False
        for k in range(len(index[j])):
            if index[j][k] == foward_init - 1:
                foward.append(index[j][k])
                foward_init -= 1
                flag = True
                break
        if flag == False:
            break

    for j in range(i+1, len(index)):
        flag = False
        for k in range(len(index[j])):
            if index[j][k] == backward_init + 1:
                backward.append(index[j][k])
                backward_init += 1
                flag = True
                break
        if flag == False:
            break

    return foward,backward


def filter_filter_index(index):
    #找到长度是1的
    i = 0
    while i < len(index):
        if len(index[i]) == 1:
            #向前和向后找
            if i == 0:
                i += 1
            else:
                forward, backward = find_forward_backward(index,i)
                start = 0
                current_index = i
                for current_index in range(i-1,i-1-len(forward),-1):
                    index[current_index] = [forward[start]]
                    start += 1
                start = 0
                for current_index in range(i+1,i+1+len(backward)):
                    index[current_index] = [backward[start]]
                    start += 1
                i = max(current_index,i) +  1
        else:
            i += 1
    return index

def only_single_value(index):
    new_index = []
    for i in range(len(index)):
        if len(index[i]) == 1:
            new_index.append(index[i][0])
        else:
            return index, False
    return new_index,True

# def get_set(index):
#     global all_set
#     all_set = []
#     #说明只有一个值
#     if len(index) == 1:
#         return index[0][0], 'yes'
#
#     index = filter_index(index)
#     if index == []:
#         return [], 'no'
#
#     index = filter_filter_index(index)
#     index, flag = only_single_value(index)
#
#     if flag == True:
#         return index, 'yes'
#
#     print("index ", index)
#     dfs(index, [], len(index))
#     print("all_set  : " , all_set)
#     one_num = 0
#     index = -1
#     for i in range(len(all_set)):
#         one_num_tmp = len(np.where(np.diff(np.array(all_set[i]))==1)[0])
#         if one_num_tmp > one_num:
#             one_num = one_num_tmp
#             index = i
#     if index == -1:
#         return  all_set[0], 'no'
#
#     #print(index)
#     #print(all_set[index])
#     return all_set[index], 'yes'


def get_set(index):
    global all_set
    all_set = []
    #说明只有一个值
    if len(index) == 1:
        return index[0][0], 'yes'

    index = filter_index(index)
    if index == []:
        return [], 'no'

    index = filter_filter_index(index)
    index, flag = only_single_value(index)

    if flag == True:
        return index, 'yes'
    all_set = []
    for i in range(len(index)):
        all_set.append(index[i][0])
    print(all_set)
    return all_set, 'yes'


def answer_in_article(answers, article):

    def func(answer, article):
        index = []
        for word in answer:
            index_tmp = find(article, word)
            if index_tmp == []:
                return [], 'no'
            index.append(index_tmp)

        index,type = get_set(index)
        return index, type

    res_value = []
    type_value = ""
    article = article.split()
    for answer in answers:
        answer = answer.split()
        print(answer)
        res,type = func(answer, article)
        if res == []:
            return [],'no'
        res_value.append(res)
        type_value = type
    return res_value,type_value


def get_count(data_arrays):
    ls = [len(data_array) for data_array in data_arrays]
    count_array = [0]
    for i in range(1,len(ls)):
        count_array.append(ls[i]+count_array[i-1])
    return count_array

def job(count, datas):
    for data in datas:
        article = ""
        answer = []
        for key, value in data.items():
            if type(value) == str:
                data[key] = ' '.join(split_sentence(value))
                if key == 'article_content':
                    article = data[key]
            else:
                for v in value:
                    for key, value in v.items():
                        v[key] = ' '.join(split_sentence(value))
                        if key == 'answer':
                            answer.append(v[key])
        # 判断答案是否在原文中
        print(count)
        res, type_value = answer_in_article(answer, article)
        #print(res)
        if res == []:
            json.dump(data, open(os.path.join('wrong_datas', str(count) + '.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=4)
        else:
            data['ans_index'] = res
            if type_value == 'yes':
                json.dump(data, open(os.path.join('right_datas', str(count) + '.json'), 'w', encoding='utf-8'),
                      ensure_ascii=False, indent=4)
            else:
                json.dump(data, open(os.path.join('right_no_consistent', str(count) + '.json'), 'w', encoding='utf-8'),
                          ensure_ascii=False, indent=4)
        count += 1


datas = json.load(open('train.json', encoding='utf-8'))
job(0,datas)
# thread_num = 4
# data_array = np.array_split(datas, thread_num)
# count_array = get_count(data_array)
# print(count_array)
#
#
# for i in range(thread_num):
#     t = threading.Thread(target=job(count_array[i],data_array[i]))
#     t.start()
#     t.join()

