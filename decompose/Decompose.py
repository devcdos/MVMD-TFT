
import os
import numpy as np
import pandas as pd
import spkit as sp
from matplotlib import pyplot as plt
from MVMD import MVMD

# from MEMD_all import memd

sourcepath="./dataset/rawsplit"
def split_train_val_test(data,hour1,hour2,flag="l"):

    index = data['hours_from_start']
    train = data.loc[index < hour1]

    valid=data.loc[(index >= hour1) & (index < hour2)]
    test=data.loc[index >= hour2]
    # train.to_csv("./splitdata/Hue/totalstep{}_tr811.csv".format(totalhourstep))
    # valid.to_csv("./splitdata/Hue/totalstep{}_v811.csv".format(totalhourstep))
    # test.to_csv("./splitdata/Hue/totalstep{}_te811.csv".format(totalhourstep))
    if flag=='l':
        train.to_csv("./dataset/rawsplit/B/totalstep{}_tr811.csv".format(""))
        valid.to_csv("./dataset/rawsplit/B/totalstep{}_v811.csv".format(""))
        test.to_csv("./dataset/rawsplit/B/totalstep{}_te811.csv".format(""))
    else:
        train.to_csv("./dataset/rawsplit/A/totalstep{}_tr811.csv".format(""))
        valid.to_csv("./dataset/rawsplit/A/totalstep{}_v811.csv".format(""))
        test.to_csv("./dataset/rawsplit/A/totalstep{}_te811.csv".format(""))
def mergeMVMD(rawdata,K,alpha,time=None,seperate_path=None):
    MVMDinput=[]
    fres=[]
    for i ,entity in enumerate(rawdata.groupby("building_id")):
        # if i==0:
        #     time=generate_hour_week(len(entity))
        x = entity[1]['energy_kWh'].values
        if len(x)%2:x=x[:-1]
        MVMDinput.append(x)

    u, u_hat, omega = MVMD(np.array(MVMDinput),
                               alpha=alpha, tau=0, K=K, DC=0, init=1, tol=1e-6)
    s=getSumSampleEntropy(np.array(MVMDinput), alpha, K)

    for j, entity in enumerate(rawdata.groupby("building_id")):
        if len(entity[1])%2:entity=entity[1].iloc[:-1,:]
        else: entity=entity[1]
        b1 = list(entity["building_id"].iloc[:1])[0]
        #if b1!="Residential_3": continue
        if seperate_path:
            for k in range(K):
                try:
                    mode=pd.DataFrame(u[k,:,j],columns=['IMF'+str(k+1)])
                except Exception as e:
                    print(b1,e)
                    break
                # time=generate_hour_week(len)
                modeTime = pd.concat([mode,time.reset_index(drop=True)],axis=1)
                # cons = cons.join(time)#

                if not os.path.exists(seperate_path+b1):
                    os.mkdir(seperate_path+b1)
                modeTime.to_csv(seperate_path+b1+"/"+'IMF'+str(k+1)+'.csv')
        else:

            p = pd.DataFrame({"IMF" + str(i + 1): u[i,:, j] for i in range(int(K))},index=entity.index)
            fres.append(entity.join(p))
    return pd.concat(fres)

def getSumSampleEntropy(input,a,k):
    u, u_hat, omega = MVMD(input, a, 0, k, 0, 1, 1e-6)
    s = np.sum(u, axis=0)
    res = input.T - s
    sam_en = sum([sp.entropy_sample(res[:, i], 2, 0.2 * np.std(res[:, i])) for i in range(res.shape[1])])
    return sam_en
def findK_alpha(input,alpha,K,name=None,flag=None):
    def findMVMDsample(K,alpha,input):
        for k in range(1, K + 1):
            maxPE = 0
            bestalpha = -1
    def findMVMD(K,alpha,input):
        bestK = []
        bestAlpha=[]
        bestSE=[]
        ds={}
        for k in range(K[0], K[1]):
            ds[str(k)] = []
            maxPE = 0
            bestalpha = -1
            sam_l1 = []
            for a in range(10, alpha, 20):
                u, u_hat, omega = MVMD(input, a, 0, k, 0, 1, 1e-6)
                s= np.sum(u, axis=0)
                res = input.T - s
                sam_en = sum([sp.entropy_sample(res[:, i], 2, 0.2 * np.std(res[:, i])) for i in range(res.shape[1])])
                ds[str(k)].append(sam_en)
                if sam_en > maxPE: bestalpha = a
                maxPE = max(sam_en, maxPE)
                print(a)
                print(sam_en)
            pd.DataFrame(ds).to_csv('./best_K_alpha/Sample/MVMD/811/finalres/' + 'alpharecord_2i20',mode='w')
            print("MVMD ",k)
            bestK.append(k)
            bestAlpha.append(bestalpha)
            bestSE.append(maxPE)
            pd.DataFrame({'m_K':bestK,'m_alpha':bestAlpha,'m_se':bestSE}).to_csv('./best_K_alpha/Sample/MVMD/811/finalres/' + '2i20',mode='w')
        return pd.DataFrame({'K': bestK, 'se': bestSE}),pd.DataFrame(bestAlpha)

def seperate_decompose(K,alpha,source_path,write_path):
    source=sorted(os.listdir(source_path))
    file=['train','valid','test']
    source.append(source.pop(0))
    for i in range(3):
        sourceinfo = pd.read_csv(source_path + '/' + source[i])
        MVMDres = mergeMVMD(sourceinfo, K, alpha[i])
        MVMDres.to_csv(write_path+"/"+file[i]+".csv")

split_train_val_test(pd.read_csv("./dataset/CaseA.csv"), 72 * 24, 81 * 24, "")
split_train_val_test(pd.read_csv("./dataset/CaseB.csv"), 392 * 24, 441 * 24)

decompose_source_path="./dataset/rawsplit/B"
#decompose_source_path= "./dataset/rawsplit/A"
#decompose_write_path= "./outputs/data/MVMDHue/25/9mode/Nocross"
decompose_write_path= "./outputs/data/MVMDHue/25/21mode/Nocross/longer"

#K, alpha = 9, [170, 1230, 1240] #MVMD parameter for A
K, alpha= 21, [ 4230, 3410,4030]#MVMD parameter for B

seperate_decompose(K,alpha,decompose_source_path,decompose_write_path)

    # time_index+=len(sourceinfo)//len(sourceinfo.groupby("building_id"))
    # print(MVMDdata[0], MVMDdata[1])




