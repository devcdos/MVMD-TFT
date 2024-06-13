import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['figure.figsize'] = (10.0, 5.0)
def plot_importance(past,static,future,path):
    past['cols']=past['cols'][past['importance'].argsort()[::-1]]
    past['importance']=past['importance'][past['importance'].argsort()[::-1]]
    future['cols'] = future['cols'][future['importance'].argsort()[::-1]]
    future['importance'] = future['importance'][future['importance'].argsort()[::-1]]
    static['cols'] = static['cols'][static['importance'].argsort()[::-1]]
    static['importance'] = static['importance'][static['importance'].argsort()[::-1]]
    past['importance'] = [x/np.sum(past["importance"]) for x in past["importance"]]
    static['importance'] = [x/np.sum(static["importance"]) for x in static["importance"]]
    future['importance'] = [x/np.sum(future["importance"]) for x in future["importance"]]
    # past['importance']=np.vectorize(lambda x : x/past["importance"].sum())
    # static['importance']=np.vectorize(lambda x: x / static["importance"].sum())
    # future['importance']=np.vectorize(lambda x: x / future["importance"].sum())
    #path = [str(params[i]) for i in param_name]
    # path = "./variable_importance/" + ''.join((x + '_' for x in path))
    # path = "./variable_importance"
    #path = "./" + ''.join((x + '_' for x in path))
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.clf()
    #sum_imf
    # plt.xticks([i+1 for i in range(len(past["cols"]))],[col for col in past["cols"]])
    # plt.barh([i+1 for i in range(len(past["cols"]))],past["importance"])
    #plt.figure(figsize=(10, 5))
    def make_selection_plot(title, values, labels):
        #fig, ax = plt.subplots(figsize=(7, len(values) * 0.25 + 2))
        fig, ax = plt.subplots(figsize=(7, len(values)* 0.25 + 2))

        # order = np.argsort(values)
        ax.barh(np.arange(len(values)), [i * 100 for i in values], tick_label=np.asarray(labels))
        ax.set_title(title)
        ax.set_xlabel("Importance in %")
        plt.tight_layout()
        plt.show()
        return fig

    # fig, ax = plt.subplots()
    #
    # # order = np.argsort(values)
    # ax.barh(np.arange(len(past['importance'])), past['importance'] * 100, tick_label=np.asarray(past['cols']))
    # ax.set_title("encoder_variables")
    # ax.set_xlabel("Importance in %")
    # plt.tight_layout()
    # plt.show()
    # p=make_selection_plot("encoder_variables",past['importance'],past['cols'])
    # f=make_selection_plot("decoder_variables", future['importance'], future['cols'])
    # s=make_selection_plot("static_variables", static['importance'], static['cols'])

    plt.barh(past['cols'], [i * 100 for i in past['importance']])
    plt.savefig(path + '/' +"test_"+ "past_inputs"+'.png',dpi=600)
    plt.clf()
    # plt.xticks([i + 1 for i in range(len(static["cols"]))], [col for col in static["cols"]])
    # plt.barh([i + 1 for i in range(len(static["cols"]))], static["importance"])
    # plt.figure(figsize=(10, 5))
    plt.barh(static["cols"], [i * 100 for i in static['importance']])
    plt.savefig(path + '/' +"test_"+"static_inputs" + '.png',dpi=600)
    plt.clf()
    # plt.xticks([i + 1 for i in range(len(future["cols"]))], [col for col in future["cols"]])
    #plt.barh(future['cols'], future["importance"])
    # plt.figure(figsize=(10, 5))
    plt.barh(future["cols"], [i * 100 for i in future['importance']])
    plt.savefig(path + '/' + "test_"+ "future_inputs" + '.png',dpi=600)
def individualplotcurve(pred,actual,rootpath):
    fig = plt.figure(figsize=(15, 30))
    # plt.ylim(7)
    # figo, axeso = plt.subplots(7, 2)
    # ax = fig.add_subplot

    indexr=0
    indexc=0
    for i,b in enumerate(pred.keys()):
        if b !='Residential_20':continue

        x = [i + 1 for i in range(actual[b].shape[0])]
        #plt.clf()
        # axt=axeso[indexr][indexc]
        # axt.ylim(0, 7)

        # fig, ax =axt.subplots()
        ax=fig.add_subplot(7,2,i+1)
        ax.title.set_text(b)
        y1 = actual[b][:,0]
        ax.plot(x, y1, label='actual',linewidth=0.6)  # 作y1 = x 图，并标记此线名为linear
        # ax.legend()
        y2 = pred[b][:, 0]
        ax.plot(x, y2, label='pred',linewidth=0.6)
        ax.legend()
        # axt.plot()
        indexc = indexc + 1 if indexc < 1 and indexr == 6 else indexc
        indexr=indexr+1 if indexr<6  else 0
    a=plt.gca()
    # fig.tight_layout()
    fig.savefig(rootpath + "overall" + '.png',dpi=800)#?
def plot_interval(d,r,c,includeB,includeA):
    fig = plt.figure(1, figsize=(20, 10))
    i = 1

    for k in d["A"].keys():
        if k not in includeA: continue
        # ax_list.append((k,fig.add_subplot(r, c, i)))
        ax = fig.add_subplot(r, c, i)
        actual=d["A"][k]['actual']
        TFTp10=d["A"][k]['interval']['TFT']['predictp10']
        TFTp50 = d["A"][k]['TFT']
        MVMDTFTp50=d["A"][k]['MVMD-TFT']
        TFTp90 = d["A"][k]['interval']['TFT']['predictp90']
        MVMDTFTp10 = d["A"][k]['interval']['MVMD-TFT']['predictp10']
        MVMDTFTp90 = d["A"][k]['interval']['MVMD-TFT']['predictp90']
        try:
            ax.fill_between(range(len(TFTp10)),TFTp10,TFTp90,color='lightsalmon',label='80% prediction interval of TFT',alpha=0.6)
            ax.fill_between(range(len(TFTp90)),MVMDTFTp10,MVMDTFTp90,color='lightblue',label='80% prediction interval of MVMD-TFT',alpha=0.6)
            ax.plot(np.array(TFTp50), label='median forecast(TFT)', color='r',linewidth=0.5)
            ax.plot(np.array(MVMDTFTp50), label='median forecast(MVMD-TFT)', color='b',linewidth=0.5)
            ax.plot(np.array(actual), label='actual', color='y',linewidth=0.8)
            ax.set_ylabel("Energy Consumption (kWh)", x=1)
            ax.set_xlabel("Time(h)",x=1)
        except Exception:
            print(k)
            return

        ax.title.set_text('R'+k.split('_')[-1])
        i += 1
    for k in d["B"].keys():
        if k not in includeB: continue
        # ax_list.append((k,fig.add_subplot(r, c, i)))
        ax = fig.add_subplot(r, c, i)
        actual = d["B"][k]['actual']
        TFTp10 = d["B"][k]['interval']['TFT']['predictp10']
        TFTp90 = d["B"][k]['interval']['TFT']['predictp90']
        TFTp50 = d["B"][k]['TFT']
        MVMDTFTp50 = d["B"][k]['MVMD-TFT']
        MVMDTFTp10 = d["B"][k]['interval']['MVMD-TFT']['predictp10']
        MVMDTFTp90 = d["B"][k]['interval']['MVMD-TFT']['predictp90']
        try:
            # ax.plot(np.array(TFTp10),label='forecast(TFT)p10')
            # ax.plot(np.array(TFTp90),label='forecast(TFT)p90')
            ax.fill_between(range(len(TFTp10)), TFTp10, TFTp90, color='lightsalmon',
                            label='80% prediction interval of TFT',alpha=0.6)
            ax.fill_between(range(len(TFTp90)), MVMDTFTp10, MVMDTFTp90, color='lightblue',
                            label='80% prediction interval of MVMD-TFT',alpha=0.6)
            ax.plot(np.array(TFTp50), label='median forecast(TFT)', color='r', linewidth=0.5)
            ax.plot(np.array(MVMDTFTp50), label='median forecast(MVMD-TFT)', color='b', linewidth=0.5)
            ax.plot(np.array(actual), label='actual', color='y', linewidth=0.8)
            ax.set_ylabel("Energy Consumption (kWh)", x=1)
            ax.set_xlabel("Time(h)",x=1)
        except Exception:
            print(k)
            return
        # if i == 4:
        #     ax.legend(loc='center right', bbox_to_anchor=(1.3, 1.2), borderaxespad=0., prop={'size': 7})  # 看一下位置什么意思
        ax.legend(loc='center', bbox_to_anchor=(1.13, 1.15), borderaxespad=0., prop={'size': 7})  # 看一下位置什么意思
        ax.title.set_text('R'+k.split('_')[-1])
        i += 1
    fig.subplots_adjust(
        wspace=0.2, hspace=0.55)
    # plt.tight_layout()
    plt.savefig('./Ye13.pdf', dpi=600,bbox_inches='tight')
    plt.show()

def plot_allpredict(d,r,c):
    # MVarr=d["MVMD-TFT"]
    # Varr = d["VMD-TFT"]
    #Tarr = d["TFT"]
    # CLarr=d["CNN-LSTM"]
    # Larr=d["LSTM"]
    ax_list = []
    #
    # for i in range(1, r * c):
    #     ax_list.append(fig.add_subplot(r, c, i))
    # df_list = []

    fig = plt.figure(1,figsize=(20,10))
    i=1
    for k in d.keys():
       # ax_list.append((k,fig.add_subplot(r, c, i)))
       ax=fig.add_subplot(r, c, i)
       try:
            # if i==5 or i==6:
            #
            #     ax.set_ylabel("Energy Consumption (kWh)")
            ax.set_ylabel("Energy Consumption (kWh)")
            ax.set_xlabel("Time (h)", x=1)
            ax.plot(np.array([d[k][al] for al in d[k].keys()]).T,label=[al for al in d[k].keys()],linewidth=0.3)

            #ax.plot(np.array([d[k][al] for al in d[k].keys() if al!="MVMD-TFT" ]).T,label=[al for al in d[k].keys() if al!="MVMD-TFT" ],linewidth=0.3)

       except Exception:
           print(k)
           return

       ax.title.set_text(k[0]+k.split('_')[-1])
       i+=1
       # for al in d[k].keys():
           # df=np.ndarray(d[k][al])
       # df=pd.DataFrame(np.array([d[k][al] for al in d[k].keys()]).T,columns=[al for al in d[k].keys()])
       # df.plot(ax=ax,title=k)
       if i>r*c:
            #ax.legend(loc='center right',bbox_to_anchor=(1.27,0.2), borderaxespad=0., prop={'size': 7})
            ax.legend(loc='center', bbox_to_anchor=(1.08,1),prop={'size': 7})

            break
       # if i<r*c:
       #     ax.legend_.remove()
       # else:
       #     ax.legend(loc=10, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)
    # plt.subplots_adjust(hspace=0.8,wspace=0.1)
    # plt.tight_layout()
    fig.subplots_adjust(
                        wspace=0, hspace=0.3)
    # fig.set_tight_layout(True)
    plt.margins(0, 0)
    plt.savefig("./Ye11.pdf",dpi=800,bbox_inches='tight')

    #, dpi = 150
    plt.show()

       #df_list.append(pd.DataFrame(np.array([d[k][al] for al in d[k].keys()]).T,columns=[al for al in d[k].keys()]))
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax2 = fig.add_subplot(2, 2, 2)
    # ax3 = fig.add_subplot(2, 2, 3)
    # ax4 = fig.add_subplot(2, 2, 4)
    # df1 = pd.DataFrame(np.random.randn(3, 5), columns=['one', 'two', 'three', 'four', 'five'])
    # df2 = pd.DataFrame(np.random.randn(3, 5), columns=['one', 'two', 'three', 'four', 'five'])
    # df3 = pd.DataFrame(np.random.randn(3, 5), columns=['one', 'two', 'three', 'four', 'five'])
    # df4 = pd.DataFrame(np.random.randn(3, 5), columns=['one', 'two', 'three', 'four', 'five'])
    #
    # df1.plot(ax=ax1, title="df1", grid='on')
    # df2.plot(ax=ax2, title="df1", grid='on')
    # df3.plot(ax=ax3, title="df1", grid='on')
    # df4.plot(ax=ax4, title="df1", grid='on')
    #
    # ax1.legend_.remove()
    # ax2.legend_.remove()
    # ax3.legend_.remove()
    # ax4.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)  ##设置ax4中legend的位置，将其放在图外



def plot_predictcurve(actualpd,predpd,name):
    # path = [str(params[i]) for i in param_name]
    path = "./actual_predict_fig"+name
    if not os.path.isdir(path):
        os.mkdir(path)
    # t=actualpd.groupby('identifier').unique()
    for bn,entity in actualpd.groupby('identifier'):
        entitypath=path+'/'+bn
        if not os.path.isdir(entitypath):
            os.mkdir(entitypath)
        actual=actualpd[actualpd['identifier']==bn].iloc[:,2:].values
        pred=predpd[predpd['identifier']==bn].iloc[:,2:].values
        x = [i + 1 for i in range(actual.shape[0])]
        fignum = actual.shape[1]
        for i in range(fignum):
            plt.clf()
            fig, ax = plt.subplots()  # 创建图实例  # 创建x的取值范围
            y1 = actual[:,i]
            ax.plot(x, y1, label='actual')  # 作y1 = x 图，并标记此线名为linear
            plt.legend()
            y2 = pred[:,i]
            ax.plot(x, y2, label='pred')  # 作y2 = x^2 图，并标记此线名为quadratic
            plt.legend()
            plt.savefig(entitypath + '/' +"timestep"+str(i)+name + '.png')
