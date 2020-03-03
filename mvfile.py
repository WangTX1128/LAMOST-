import pandas as pd
import os
import shutil
import multiprocessing
import numpy as np
origin_path = '/root/test_data/'
target_path = '/root/code/bad_data/'

def mv_file(csvfile):
    print('moving %s'%csvfile)
    i = csvfile.split('.')[0]
    i = i + '.txt'
    shutil.move(origin_path+i,target_path+i)

result=[]
file = open("error_name.txt") 
for line in file:
    name = line.split("\t")[0]
    result.append(name)
file.close()

pool = multiprocessing.Pool()
pool.map(mv_file,result)
pool.close()
pool.join()