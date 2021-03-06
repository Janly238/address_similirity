import numpy as np
import pandas as pd
from csv import reader
import time
import cpca
import jieba
from gensim import corpora, models, similarities

def read_data():
    #导入数据,并对数据做一些处理
    df = pd.read_excel("data/public_company.xlsx", dtype={'公司代码': 'str'})
    addr_df = df[["公司代码", "公司简称", "注册地址"]]
    addr_df["注册地址"]= addr_df["注册地址"].apply(lambda x: str(x).strip())
    return addr_df

# df = read_data()
# print(df)

def get_dataset(addr_df):
    '''
    #对地址做一个标准化处理,其中导入cpca的包进行处理
    1.将使用cpca之后按照省市区拼接
    2.将使用cpca之后按照省市拼接

    '''
    start = time.clock()
    
    location_str = []
    for i in addr_df['注册地址']:
        location_str.append(i.strip())
        
    addr_cp = cpca.transform(location_str,cut=False,open_warning=False)
    
    #给结果表拼接唯一识别代码
    e_data = addr_df[["公司代码", "公司简称"]]
    addr_cpca = pd.concat([e_data, addr_cp], axis=1)
    print(addr_cpca)
    '''
        公司代码   公司简称    省    市    区                                    地址
    0       1   平安银行  广东省  深圳市  罗湖区                             深南东路5047号
    1  000002  万  科Ａ  广东省  深圳市  盐田区                         大梅沙环梅路33号万科中心
    2  000004   国农科技  广东省  深圳市  南山区            中心路（深圳湾段）3333号中铁南方总部大厦503室
    3  000005   世纪星源  广东省  深圳市  罗湖区                             深南东路5046号
    4  000006   深振业Ａ  广东省  深圳市  罗湖区                             深南东路5048号
    5  000007    全新好  广东省  深圳市  福田区                     梅林街道梅康路8号理想时代大厦6楼
    6  000008   神州高铁  北京市  北京市  海淀区                   高梁桥斜街59号院1号楼16层1606
    7  000009   中国宝安  广东省  深圳市                      笋岗东路1002号宝安广场A座28-29层
    8  000010  *ST美丽  广东省  深圳市  宝安区  新安街道海旺社区宝兴路6号海纳百川总部大厦B座17层1701-1703室
    9  000011   深物业A  广东省  深圳市                             人民南路国贸大厦39、42层
    '''
    
    #1.区不为空
    addr_cpca_1 = addr_cpca[(addr_cpca['省']!= '')&(addr_cpca['市']!= '') & (addr_cpca['区']!= '')]
    addr_cpca_1= addr_cpca_1.dropna()
    
    addr_cpca_11= addr_cpca_1[(addr_cpca['地址']!='')]
    addr_cpca_12= addr_cpca_11. dropna(subset=['地址'])
    
    #将前三个字段完全拼接在一起进行分组然后组内进行相似度分析
    addr_cpca_12['省市区'] = addr_cpca_12['省'] + addr_cpca_12['市'] + addr_cpca_12['区']
    
    addr_cpca_12['省市区长度']=addr_cpca_12['省市区'].apply(lambda x: len(x))
    count_1 = addr_cpca_12['省市区'].value_counts().reset_index()
    count_1= count_1.rename(columns={'index':'省市区', '省市区':'个数'})
    print(addr_cpca_12)
    print(count_1)
    '''
    省市区  个数
    0  广东省深圳市罗湖区   3
    1  北京市北京市海淀区   1
    2  广东省深圳市福田区   1
    3  广东省深圳市盐田区   1
    4  广东省深圳市宝安区   1
    5  广东省深圳市南山区   1
    '''
    
    count_delete_1= count_1[count_1['个数']==1]
    dataset_1 = pd.merge(addr_cpca_12, count_delete_1, on = '省市区', how = 'left')
    print(dataset_1)
    '''merge合并，个数为1的填1，没有的话填nan
            公司代码   公司简称    省    市    区                                    地址        省市区  省市区长度   个数
    0       1   平安银行  广东省  深圳市  罗湖区                             深南东路5047号  广东省深圳市罗湖区      9  NaN
    1  000002  万  科Ａ  广东省  深圳市  盐田区                         大梅沙环梅路33号万科中心  广东省深圳市盐田区      9  1.0
    2  000004   国农科技  广东省  深圳市  南山区            中心路（深圳湾段）3333号中铁南方总部大厦503室  广东省深圳市南山区      9  1.0
    3  000005   世纪星源  广东省  深圳市  罗湖区                             深南东路5046号  广东省深圳市罗湖区      9  NaN
    4  000006   深振业Ａ  广东省  深圳市  罗湖区                             深南东路5048号  广东省深圳市罗湖区      9  NaN
    5  000007    全新好  广东省  深圳市  福田区                     梅林街道梅康路8号理想时代大厦6楼  广东省深圳市福田区      9  1.0
    6  000008   神州高铁  北京市  北京市  海淀区                   高梁桥斜街59号院1号楼16层1606  北京市北京市海淀区      9  1.0
    7  000010  *ST美丽  广东省  深圳市  宝安区  新安街道海旺社区宝兴路6号海纳百川总部大厦B座17层1701-1703室  广东省深圳市宝安区      9  1.0
    '''
    dataset_1= dataset_1[dataset_1['个数']!=1]
    '''筛个数不等于1的
    公司代码  公司简称    省    市    区         地址        省市区  省市区长度  个数
    0       1  平安银行  广东省  深圳市  罗湖区  深南东路5047号  广东省深圳市罗湖区      9 NaN
    3  000005  世纪星源  广东省  深圳市  罗湖区  深南东路5046号  广东省深圳市罗湖区      9 NaN
    4  000006  深振业Ａ  广东省  深圳市  罗湖区  深南东路5048号  广东省深圳市罗湖区      9 NaN
    '''

    print(dataset_1)
    
    #2.区为空  ,同样操作
    addr_cpca_2 = addr_cpca[(addr_cpca['省']!= '')&(addr_cpca['市']!= '') & (addr_cpca['区']== '')]
    addr_cpca_2 = addr_cpca_2.dropna()
    
    addr_cpca_21= addr_cpca_2[(addr_cpca['地址']!='')]
    addr_cpca_22= addr_cpca_21.dropna(subset=['地址'])
    
    #将前三个字段完全拼接在一起进行分组然后组内进行相似度分析
    addr_cpca_22['省市区'] = addr_cpca_22['省'] + addr_cpca_22['市']
    
    addr_cpca_22['省市区长度']=addr_cpca_22['省市区'].apply(lambda x: len(x))
    count_2 = addr_cpca_22['省市区'].value_counts().reset_index()
    count_2= count_2.rename(columns={'index':'省市区', '省市区':'个数'})
    
    count_delete_2 = count_2[count_2['个数']==1]
    dataset_2 = pd.merge(addr_cpca_22, count_delete_2, on = '省市区', how = 'left')
    dataset_2 = dataset_2[dataset_2['个数']!=1]
    
    
    print("Time used:", (time. clock()-start), "s")
    
    return dataset_1, dataset_2

# dataset_1, dataset_2 = get_dataset(df)
# print(dataset_1)
# print("-----------")
# print(dataset_2)

def cal_similar(doc_goal, document, ssim = 0.1):
#def cal_similar(doc_goai, document):
    '''
    分词;计算文本相似度
    doc_goal,短文本,目标文档
    document,多个文本,被比较的多个文档
    '''
    all_doc_list=[]
    for doc in document:
        doc= "".join(doc)
        doc_list=[word for word in jieba.cut(doc)]
        all_doc_list.append(doc_list)

    #目标文档
    doc_goal = "".join(doc_goal)
    doc_goal_list = [word for word in jieba.cut(doc_goal)]
    
    #被比较的多个文档
    dictionary = corpora.Dictionary(all_doc_list)  #先用dictionary方法获词袋
    corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]  #使用doc2bow制作预料库
    tfidf = models.TfidfModel(corpus)#使用TF-DF模型对料库建模
    
    #目标文档
    doc_goal_vec = dictionary.doc2bow(doc_goal_list)
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features = len(dictionary.keys()))#对每个目标文档,分析测文档的相似度
    print("index,",index)
    #开始比较
    sims = index[tfidf[doc_goal_vec]]
    print("sims:",sims)
    #similary= sorted(enumerate(sims),key=lambda item: -item[1])#根据相似度排序
    
    addr_dict={"被比较地址": document, "相似度": list(sims)}
    print("addr_dict:",addr_dict)
    '''
    addr_dict: {'被比较地址': ['深南东路5046号', '深南东路5048号'], '相似度': [0.0, 0.0]}
    '''

    similary = pd.DataFrame(addr_dict)
    similary["目标地址"] = doc_goal
    similary_data = similary[["目标地址", "被比较地址", "相似度"]]
    similary_data= similary_data[similary_data["相似度"]>=ssim]
    print("similary_data:",similary_data)
    '''
    similary_data:         目标地址      被比较地址  相似度
    0  深南东路5047号  深南东路5046号  0.0
    1  深南东路5047号  深南东路5048号  0.0
    '''
    
    return similary_data

def cycle_first(single_data):
    
    single_value = single_data.loc[:,["公司代码","地址"]].values #提取地址
    
    cycle_data = pd. DataFrame([])
    for key, value in enumerate(single_value):
        if key < len(single_data)-1:
            doc_goal=list(value)[1:]
            document=list(single_data["地址"])[key+1:]
            cycle = cal_similar(doc_goal, document, ssim=0)
            cycle['目标地址代码'] = list(single_data["公司代码"])[key]
            cycle['被比较地址代码'] = list(single_data["公司代码"])[key+1:]
            cycle = cycle[["目标地址代码","目标地址", "被比较地址代码", "被比较地址", "相似度"]]
            #print("循环第",key,"个地址,得到表的行数,",len(cycle),",当前子循环计算进度,",key/len(cycle))
        cycle_data = cycle_data.append(cycle)
        cycle_data = cycle_data.drop_duplicates()
   
    return cycle_data


def get_collect(dataset):

    start = time. clock()
    
    #获取单个省市区的文档
    collect_data = pd.DataFrame([])
    ssq=list(set(dataset['省市区']))
    for v, word in enumerate(ssq):
        single_data = dataset[dataset['省市区'] == word]
        print("循环第",v,"个省市区地址为:",word,",当前此区地址有:",len(single_data),",当前计算进度为:{:.1f}%" .format(v*100/len(ssq)))
        cycle_data = cycle_first(single_data)
        print("cycle_data",cycle_data)
        '''
        目标地址代码       目标地址 被比较地址代码      被比较地址  相似度
        0       1  深南东路5047号  000005  深南东路5046号  0.0
        1       1  深南东路5047号  000006  深南东路5048号  0.0
        0  000005  深南东路5046号  000006  深南东路5048号  0.0
        '''
        collect_data = collect_data.append(cycle_data)#将每个市区得到的结果放入一张表
        
        print("Time: %s" %time.ctime())
        print("-----------------------------------------------------------------------")
    
    print("Time used:",(time.clock() - start), "s")
    
    return collect_data


def run_(par = 0):
    #调用上述函数
    addr_df = read_data()
    dataset_1, dataset_2 = get_dataset(addr_df)
    #dataset. to_csv("../data/addr_data/document_address.csv", index =False)
    #dataset. to_csv( ". /data/addr_data/document_address. csv", index False)
    collect_data_1 = get_collect(dataset_1)
    collect_data_2 = get_collect(dataset_2)
    collect_data = pd.concat([collect_data_1, collect_data_2], axis=0)
    collect_data = collect_data[collect_data["相似度"]>=par].sort_values(by=["相似度"], ascending=[False])
    
    collect_data["相似度"] = collect_data["相似度"].apply(lambda x: ('%.2f' % x))

    return collect_data


collect_data = run_(par = 0.1)
print(collect_data)
#collect_data.to_excel("data/result.xlsx")