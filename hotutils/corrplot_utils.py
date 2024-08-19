import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from sklearn.linear_model import LinearRegression

#drawing function
def count_decimal_places(number):
    # 将数字转换为字符串
    number_str = str(number)
    
    # 判断是否包含小数点
    if '.' in number_str:
        # 获取小数点后的部分，并返回其长度
        return len(number_str.split('.')[1])
    else:
        # 如果没有小数点，则返回 0
        return 0
    
def save_point(number,k):
    number_str = round(float(number), k)
    n = count_decimal_places(number_str)
    if n == k:
        result = str(number_str)
    else:
        result = str(number_str) + '0'
    return result

def ax_corr(fig,pd,pr,xlim,ylim,dx,dy,xc,yc,labelloc,corr,models,xylabel,tlabel,title1,title2,title3,is_legend,ft,x1,x2,y1,y2,sft=100,if_minor=False):
    shapes=['o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P']
    scolors=['#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000',
             '#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF',
             '#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF',
             '#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7',
             'gold','gold','gold','gold','gold','gold','gold','gold','gold','gold',
             '#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3']
    model = LinearRegression()
    X = np.array(pd).reshape(-1, 1)
    Y = np.array(pr).reshape(-1, 1)
    xline = np.arange(pd.min(),pd.max()+0.01,0.01)
    model.fit(X, Y)
    ax1 = fig.add_axes([x1, y1, x2-x1, y2-y1])
    for i in range(len(pd)):
        ax1.scatter(pd[i],pr[i],marker=shapes[i],c=scolors[i],s=sft,edgecolors='none',linewidth=0.1,label=models[i])
        # ax1.scatter(pd[i],pr[i],marker=shapes[i],s=100,edgecolors=scolors[i],facecolors='none',linewidth=1.5,label=models[i])

    ax1.scatter(pd.mean(),pr.mean(),marker='*',s=sft,edgecolors='k',facecolors='none',linewidth=1.5,label='MME')
    ax1.plot(xline,model.predict(xline.reshape(-1,1)),color='k',linewidth=2)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.xaxis.set_major_locator(MultipleLocator(dx))#设置y轴数值间隔
    ax1.yaxis.set_major_locator(MultipleLocator(dy))#设置y轴数值间隔
    if if_minor:
        ax1.minorticks_on()  # 开启左下次刻度
        ax1.xaxis.set_minor_locator(MultipleLocator(dx/5))
        ax1.yaxis.set_minor_locator(MultipleLocator(dy/5))
    ax1.tick_params(length=5, width=1.3, labelsize=ft)  # 调整主刻度字体及轴的长宽
    ax1.tick_params(which='minor', length=3)
    if is_legend:
        # ax1.legend(loc='center',bbox_to_anchor=(0.5, -0.4),ncol = 2,fontsize=ft,frameon=True)
        ax1.legend(loc='center',bbox_to_anchor=(2, 0.5),ncol = 4,fontsize=ft-0.8,frameon=False) #
    ax1.set_title(title1, loc='left', fontsize=ft+3,fontweight='bold')  # 左上title
    ax1.set_title(title2, loc='center', fontsize=ft+3, fontweight='bold')  # 右上title
    ax1.set_title(title3, loc='right', fontsize=ft+3, fontweight='bold')  # 右上title
    ax1.text(xlim[0] - (xlim[1]-xlim[0])*labelloc[0], ylim[1] + (ylim[1]-ylim[0])*labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')
    ax1.axvline(0, linestyle='--', color='k',alpha=0.7, linewidth=1.5)
    ax1.axhline(0, linestyle='--', color='k', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel(xylabel[1], fontsize=ft+1)
    ax1.set_xlabel(xylabel[0], fontsize=ft+1)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    r_value  = save_point(corr[0], 2)
    p_value  = save_point(corr[1], 2)
    dx_range = xlim[1]-xlim[0]
    dy_range = ylim[1]-ylim[0]

    ax1.text(xlim[0] + xc*dx_range,ylim[0] + (yc+0.06)*dy_range,'R = '+ r_value,fontsize=ft+2,fontweight='bold')

    if corr[1] < 0.01:
        ax1.text(xlim[0] + xc*dx_range,ylim[0] + yc*dy_range,'P < 0.01',fontsize=ft+2,fontweight='bold')
    else:
        ax1.text(xlim[0] + xc*dx_range,ylim[0] + yc*dy_range,'P = '+ p_value,fontsize=ft+2,fontweight='bold') 
    
    return ax1

def MME(ds):
    dsm    = ds.mean('model')
    ds_std = ds.std('model')
    return dsm,ds_std 

def ax1_corr(fig,pd,pr,pdo,pro,xlim,ylim,dx,dy,xc,yc,labelloc,corr,models,xylabel,tlabel,title1,title2,title3,is_legend,ft,x1,x2,y1,y2,sft=100):
    shapes=['o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P',
            'o','^','v','*','s','<','p','>','d','P']
    scolors=['#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000','#FF0000',
             '#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF','#00FFFF',
             '#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF','#FF9AFF',
             '#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7','#00C8A7',
             'gold','gold','gold','gold','gold','gold','gold','gold','gold','gold',
             '#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3','#4D85E3']
    model = LinearRegression()
    pdo_mme,pdo_err = MME(pdo)
    pro_mme,pro_err = MME(pro)

    pdm_mme,pdm_err = MME(pd)
    prm_mme,prm_err = MME(pr)

    X = np.array(pd).reshape(-1, 1)
    Y = np.array(pr).reshape(-1, 1)
    xline = np.arange(pd.min(),pd.max()+0.01,0.01)
    model.fit(X, Y)
    ax1 = fig.add_axes([x1, y1, x2-x1, y2-y1])
    for i in range(len(pd)):
        ax1.scatter(pd[i],pr[i],marker=shapes[i],c=scolors[i],s=sft,edgecolors='none',linewidth=0.1,label=models[i])
        # ax1.scatter(pd[i],pr[i],marker=shapes[i],s=100,edgecolors=scolors[i],facecolors='none',linewidth=1.5,label=models[i])

    ax1.scatter(pd.mean(),pr.mean(),marker='*',s=sft+30,edgecolors='k',facecolors='k',linewidth=0.1,label='MME') # ,facecolors='none'
    ax1.scatter(pdo_mme,pro_mme,marker='o',c='k',s=sft-30,edgecolors='k',linewidth=0.1,label='OBS')
    ax1.errorbar(pdm_mme, prm_mme, yerr=prm_err,xerr=pdm_err, ecolor='k', capsize=4, ls='none',capthick=1.3, elinewidth=1.3)
    ax1.errorbar(pdo_mme, pro_mme, yerr=pro_err,xerr=pdo_err, ecolor='k', capsize=4, ls='none',capthick=1.3, elinewidth=1.3)
    ax1.plot(xline,model.predict(xline.reshape(-1,1)),color='k',linewidth=2)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.xaxis.set_major_locator(MultipleLocator(dx))#设置y轴数值间隔
    ax1.yaxis.set_major_locator(MultipleLocator(dy))#设置y轴数值间隔
    ax1.minorticks_on()  # 开启左下次刻度
    ax1.xaxis.set_minor_locator(MultipleLocator(dx/5))
    ax1.yaxis.set_minor_locator(MultipleLocator(dy/5))
    ax1.tick_params(length=5, width=1.3, labelsize=ft)  # 调整主刻度字体及轴的长宽
    ax1.tick_params(which='minor', length=3)
    if is_legend:
        # ax1.legend(loc='center',bbox_to_anchor=(0.5, -0.4),ncol = 2,fontsize=ft,frameon=True)
        ax1.legend(loc='center',bbox_to_anchor=(2, 0.5),ncol = 4,fontsize=ft-0.8,frameon=False) #
    ax1.set_title(title1, loc='left', fontsize=ft+5,fontweight='bold')  # 左上title
    ax1.set_title(title2, loc='center', fontsize=ft+5, fontweight='bold')  # 右上title
    # title.set_position([-0.18,1.02])
    ax1.set_title(title3, loc='right', fontsize=ft+5, fontweight='bold')  # 右上title
    ax1.text(xlim[0] - (xlim[1]-xlim[0])*labelloc[0], ylim[1] + (ylim[1]-ylim[0])*labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')
    ax1.axvline(0, linestyle='--', color='k',alpha=0.7, linewidth=1.5)
    ax1.axhline(0, linestyle='--', color='k', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel(xylabel[1], fontsize=ft)
    ax1.set_xlabel(xylabel[0], fontsize=ft)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    r_value  = save_point(corr[0], 2)
    p_value  = save_point(corr[1], 2)
    dx_range = xlim[1]-xlim[0]
    dy_range = ylim[1]-ylim[0]


    ax1.text(xlim[0] + xc*dx_range,ylim[0] + (yc+0.06)*dy_range,'R = '+ r_value,fontsize=ft+2,fontweight='bold')

    if corr[1] < 0.01:
        ax1.text(xlim[0] + xc*dx_range,ylim[0] + yc*dy_range,'P < 0.01',fontsize=ft+2,fontweight='bold')
    else:
        ax1.text(xlim[0] + xc*dx_range,ylim[0] + yc*dy_range,'P = '+ p_value,fontsize=ft+2,fontweight='bold') 
    
    return ax1