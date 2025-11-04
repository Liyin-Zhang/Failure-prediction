# 参数
import pickle
#
# status='train' # 生成训练或测试数据
# time_begin=0
# time_end=2800

# status='evaluate' # 生成训练或测试数据
# time_begin=2800
# time_end=3672

gen_data_name='1013_1.csv' # 生成文件名称


embed_dim=24#16
feature_num= 6
max_node_number=1000
learning_rate=0.0001
stepsize=200
gama=0.8
batchsize=256






def read_climate_data(y,m):
    dir = f"{y}_{m:02d}"
    # print(dir)
    with open('../../climate/pickle_data/'+dir+'/GD_WIN_'+dir+'.pickle', 'rb') as f:
        WIN = pickle.load(f)
    with open('../../climate/pickle_data/' + dir + '/GD_PRE_' + dir + '.pickle', 'rb') as f:
        PRE = pickle.load(f)
    with open('../../climate/pickle_data/' + dir + '/GD_PRS_' + dir + '.pickle', 'rb') as f:
        PRS = pickle.load(f)
    with open('../../climate/pickle_data/' + dir + '/GD_SHU_' + dir + '.pickle', 'rb') as f:
        SHU = pickle.load(f)
    with open('../../climate/pickle_data/'+dir+'/GD_TMP_'+dir+'.pickle', 'rb') as f:
        TMP = pickle.load(f)

    return WIN,PRE,PRS,SHU,TMP


max_value = {
    'node_weights': 61,
    'pole_material':1,

    'TMP': 302,
    'PRS': 101400,
    'SHU': 0.023,
    'PRE':79,
    'WIN':3.5,

    'TMP_delta': 2.7,
    'PRS_delta': 240,
    'SHU_delta': 0.0014,
    'PRE_delta':70,
    'WIN_delta':2.5,
}

min_value = {
    'node_weights': 0,
    'pole_material': 0,

    'TMP': 285,
    'PRS': 92000,
    'SHU': 0.0097,
    'PRE': 0,
    'WIN': 0.008,

    'TMP_delta': -5.5,
    'PRS_delta': -700,
    'SHU_delta': -0.0047,
    'PRE_delta': -64,
    'WIN_delta': -2,
}
