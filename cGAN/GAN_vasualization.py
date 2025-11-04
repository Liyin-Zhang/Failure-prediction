# 画出预测结果的气象分布
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['simsun']
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import pickle
import pickle5 as picklew
import pandas as pd
import numpy as np
import math
from datetime import datetime,timedelta
import copy
import seaborn as sns
from calendar import monthrange



def main():


    train_data=pd.read_csv('GAN_data/train/train_0313_1_clustered.csv', encoding='gbk')
    grouped = train_data.groupby('feeder_name')
    df_list=[]
    # header = ['WIN', 'PRE', 'PRS', 'TMP', 'SHU']
    header=["WIN_extreme", "PRE_extreme", "PRS_extreme", "SHU_extreme", "TMP_extreme"]
    n=0
    for feeder_name, group in grouped:
        n+=1
        flag='真实'
        df = {}
        df['feeder_name'] = feeder_name
        df['flag'] = flag
        for i in header:
            l = group[i].tolist()
            df[i] = sum(l) / len(l)
        df_list.append(df)
    print(n)

    gen_data = pd.read_csv('GAN_data/gen_data/cGAN_restored_features_0402.csv', encoding='gbk')
    grouped = gen_data.groupby('feeder_name')
    # header = ['WIN', 'PRE', 'PRS', 'TMP', 'SHU']
    feeder_l=[]
    for feeder_name, group in grouped:
        flag = '生成'
        df = {}
        df['feeder_name'] = feeder_name
        df['flag'] = flag
        for i in header:
            l = group[i].tolist()
            df[i]=sum(l) / len(l)
        if (df['PRE_extreme']>0.3):
            # print(feeder_name)
            feeder_l.append(feeder_name)
        df_list.append(df)
    print(feeder_l)

    merged_df = pd.DataFrame(df_list)
    # 定义不同 label 对应的点大小
    size_mapping = {'真实': 10, '生成':30, }  # 直径大小
    merged_df['size'] = merged_df['flag'].map(size_mapping)
    # 定义不同 label 对应的透明度（alpha 值）
    alpha_mapping = {'真实': 0.4, '生成':1, }
    merged_df['alpha'] =  merged_df['flag'].map(alpha_mapping)
    # 设定固定的 flag 顺序
    flag_order = ['真实', '生成', ]  # 自定义顺序

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=merged_df, x='PRE_extreme', y='WIN_extreme', hue='flag', size='size',palette='Set2',hue_order=flag_order)
    # 添加标题和网格
    # plt.title('跳闸次数为'+str(failure_t))
    plt.grid(True)
    plt.tight_layout()

if __name__ == '__main__':

    main()
    plt.show()