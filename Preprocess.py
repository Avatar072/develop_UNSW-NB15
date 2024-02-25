import warnings
import os
import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from mytoolfunction import generatefolder,SaveDataToCsvfile,SaveDataframeTonpArray,CheckFileExists,splitdatasetbalancehalf,printFeatureCountAndLabelCountInfo
from mytoolfunction import RealpaceSymboltoTheMostCommonValue,ReplaceMorethanTenthousandQuantity
filepath = "D:\\develop_UNSW-NB15\\UNSW-NB15 dataset"
today = datetime.date.today()
today = today.strftime("%Y%m%d")
# 在D:\\Labtest20230911\\data\\dataset_original產生天日期的資料夾
generatefolder(filepath + "\\", "dataset_AfterProcessed")
generatefolder(filepath + "\\dataset_AfterProcessed\\", today)
print("pandas version",pd.__version__)

### label encoding
def label_Encoding(label):
    label_encoder = preprocessing.LabelEncoder()
    # 將列中的數據類型統一轉換
    dataset[label] = dataset[label].astype(str)
    dataset[label] = label_encoder.fit_transform(dataset[label])
    # dataset[label] = label_encoder.fit_transform(dataset[label]).astype(str)
    dataset[label].unique()


def label_encoding(label, df):
    label_encoder = preprocessing.LabelEncoder()
    original_values = df[label].unique()
    
    df[label] = label_encoder.fit_transform(df[label])
    encoded_values = df[label].unique()
    
    return original_values, encoded_values, df  

### 載入特徵名稱
def ReadFeatureName():
    df = pd.read_csv(filepath + "\\CSV Files\\NUSW-NB15_features.csv",encoding='ISO-8859-1')
    # 提取"Name"列的值
    featurname = df['Name'].tolist()
    # 打印"Name"列的值
    print(featurname)
    # 创建一个空的DataFrame
    df = pd.DataFrame()
    # 提取"Name"列的值并作为新的列添加到DataFrame
    for columns_name in featurname:
        df[columns_name] = None

    df.to_csv(filepath + "\\CSV Files\\feature_columns.csv", index=False)
    # 回傳特徵名稱
    return  featurname


# 加载UNSW-NB15数据集
def writeData(file_path):
    # 读取CSV文件并返回DataFrame
    df = pd.read_csv(file_path,encoding='ISO-8859-1',low_memory=False)
    # 載入特徵名稱
    df.columns = ReadFeatureName()
    # 因normal的值是空白會讀不到才做取代 替换空白值为'normal'
    df['attack_cat'] = df['attack_cat'].fillna('normal')
    # 将每列中的 "-" 和空白值替换为最常见值
    df = RealpaceSymboltoTheMostCommonValue(df)
    # 找到不包含NaN、Infinity和"inf"值的行 這句用panda 1.3.5版本才不會報錯
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    # 获取倒数第二列和最后一列的欄位值
    second_last_column = df.iloc[:, -2].copy()
    last_column = df.iloc[:, -1].copy()
    # 互换倒数第二列和最后一列的欄位值 把攻擊類別欄位放在最後面
    df.iloc[:, -2] = last_column
    df.iloc[:, -1] = second_last_column

    return df

### 检查要合并的多个DataFrame的特征名是否一致
def check_column_names(dataframes):
    # 获取第一个DataFrame的特征名列表
    reference_columns = list(dataframes[0].columns)

    # 检查每个DataFrame的特征名是否都与参考特征名一致
    for df in dataframes[1:]:
        if list(df.columns) != reference_columns:
            return False

    return True

### merge多個DataFrame
def mergeData():
    # 创建要合并的DataFrame列表
    dataframes_to_merge = []

    # 添加每个CSV文件的DataFrame到列表
    dataframes_to_merge.append(writeData(filepath + "\\CSV Files\\UNSW-NB15_1.csv"))
    dataframes_to_merge.append(writeData(filepath + "\\CSV Files\\UNSW-NB15_2.csv"))
    dataframes_to_merge.append(writeData(filepath + "\\CSV Files\\UNSW-NB15_3.csv"))
    dataframes_to_merge.append(writeData(filepath + "\\CSV Files\\UNSW-NB15_4.csv"))

    # 检查特征名是否一致
    if check_column_names(dataframes_to_merge):
        # 特征名一致，可以进行合并
        result = pd.concat(dataframes_to_merge)
        return result
    else:
        # 特征名不一致，需要处理这个问题
        print("特征名不一致，请检查并处理特征名一致性")
        return None

def LastColumnDoLabelEncode(df):
    # if true_or_fasle != True:
    # 先存一次Label未做encode的csv方便後面noniid實驗    
    df.to_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000_and_UndoLabelencode_minmax.csv", index=False)
    # 做encode
    original_type_values, encoded_type_values, df = label_encoding("Label", df)
    print("Original Type Values:", original_type_values)
    print("Encoded Type Values:", encoded_type_values)
    # dataset.to_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000_and_doLabelencode.csv", index=False)
    return df

if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed.csv")!=True):
    #合併後完整的資料 
    dataset = mergeData()
    # 保存到CSV文件，同时将header设置为True以包括特征名行
    print("after rmove dirty data\n",dataset['Label'].value_counts())
    dataset.to_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed.csv", index=False, header=True)
else:
    dataset= pd.read_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed.csv")
    print("count\n",dataset['Label'].value_counts())

# ct_ftp_cmd欄位的空白無法去掉 所以用手補0
if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000.csv")!=True):
    dataset = ReplaceMorethanTenthousandQuantity(dataset,'Label')
    dataset.to_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000.csv", index=False)
    dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000.csv")
    print("count\n",dataset['Label'].value_counts())
else:
    dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000.csv")
    print("count\n",dataset['Label'].value_counts())

label_Encoding('srcip')
label_Encoding('sport')
label_Encoding('dstip')
label_Encoding('dsport')
label_Encoding('proto')
label_Encoding('state')

# label_Encoding('dur')
# label_Encoding('sbytes')
# label_Encoding('dbytes')

label_Encoding('sttl')
label_Encoding('dttl')

# label_Encoding('sloss')
# label_Encoding('dloss')
label_Encoding('service')

# label_Encoding('Sload')
# label_Encoding('Dload')
# label_Encoding('Spkts')
# label_Encoding('Dpkts')
# label_Encoding('swin')
# label_Encoding('dwin')

label_Encoding('stcpb')
label_Encoding('dtcpb')

# label_Encoding('smeansz')
# label_Encoding('dmeansz')
# label_Encoding('trans_depth')
# label_Encoding('res_bdy_len')
# label_Encoding('Sjit')
# label_Encoding('Djit')

label_Encoding('Stime')
label_Encoding('Ltime')

# label_Encoding('Sintpkt')
# label_Encoding('Dintpkt')
# label_Encoding('tcprtt')
# label_Encoding('synack')
# label_Encoding('ackdat')
# label_Encoding('is_sm_ips_ports')
# label_Encoding('ct_state_ttl')
# label_Encoding('ct_flw_http_mthd')
# label_Encoding('is_ftp_login')
# label_Encoding('ct_ftp_cmd')
# label_Encoding('ct_srv_src')
# label_Encoding('ct_srv_dst')
# label_Encoding('ct_dst_ltm')
# label_Encoding('ct_src_ltm')
# label_Encoding('ct_src_dport_ltm')
# label_Encoding('ct_dst_sport_ltm')
# label_Encoding('ct_dst_src_ltm')
# label_Encoding('attack_cat')

### extracting features
#除了label外的特徵
crop_dataset=dataset.iloc[:,:-1]
# 列出要排除的列名，這6個以外得特徵做minmax
columns_to_exclude = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state']
# 使用条件选择不等于这些列名的列
doScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col not in columns_to_exclude]]
undoScalerdataset = crop_dataset[[col for col in crop_dataset.columns if col  in columns_to_exclude]]
# print(doScalerdataset.info)
# print(afterprocess_dataset.info)
# print(undoScalerdataset.info)
# 開始minmax
X=doScalerdataset
X=X.values
# scaler = preprocessing.StandardScaler() #資料標準化
scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
scaler.fit(X)
X=scaler.transform(X)
# 将缩放后的值更新到 doScalerdataset 中
doScalerdataset.iloc[:, :] = X
# 将排除的列名和选中的特征和 Label 合并为新的 DataFrame
afterminmax_dataset = pd.concat([undoScalerdataset,doScalerdataset,dataset['Label']], axis = 1)

#minmax完做Label列encode
afterminmax_dataset = LastColumnDoLabelEncode(afterminmax_dataset)

###看檔案是否存在不存在存檔後讀檔
if(CheckFileExists(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000_and_doLabelencode_minmax.csv")!=True):
    SaveDataToCsvfile(afterminmax_dataset, f"./UNSW-NB15 dataset/dataset_AfterProcessed", f"UNSW-NB15_AfterProcessed_updated_10000_and_doLabelencode_minmax")
    afterminmax_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000_and_doLabelencode_minmax.csv")

else:
    afterminmax_dataset = pd.read_csv(filepath + "\\dataset_AfterProcessed\\UNSW-NB15_AfterProcessed_updated_10000_and_doLabelencode_minmax.csv")


