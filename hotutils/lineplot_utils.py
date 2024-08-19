import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


def MME(ds):
    mean = ds.mean('model')
    std  = ds.std('model')
    updw = np.array([mean+std,mean-std])
    return mean,updw

def pt_fig(fig,line1,line2,line1_mn,line2_mn,ylim1,ylim2,dy1,dy2,labelloc,tlabel,title1,title2,title3,ylabel1,ylabel2,ft,x1,x2,y1,y2,labelpad=12):
    ax1 = fig.add_axes([x1, y1, x2-x1, y2-y1])
    X = np.arange(1, 100, 1)
    ax2 = ax1.twinx()
    line1_color='#FF0000'
    line2_color='#D600C4'  
    ax1.plot(X, line1, color=line1_color, linestyle='-', linewidth=2.5, zorder=1)
    ax2.plot(X, line2, color=line2_color, linestyle='-', linewidth=2, zorder=1)
    ax1.fill_between(X, line1_mn[0], line1_mn[1], color='lightpink', zorder=0,alpha=0.3)
    ax2.fill_between(X, line2_mn[0], line2_mn[1], color='#FF9AFF', zorder=0,alpha=0.2)
    ax1.set_xlim(1, 99)
    # ax1.set_ylim(-25, 25)
    ax1.set_xticks([1, 10, 30, 50, 70, 90, 99])
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax1.minorticks_on()  # 开启左下次刻度
    # ax1.xaxis.set_major_locator(MultipleLocator(5))#设置y轴数值间隔
    # ax1.xaxis.set_minor_locator(MultipleLocator(5))
    # ax1.yaxis.set_minor_locator(MultipleLocator(5))
    ax1.tick_params(axis='y', length=5, width=1.3, color=line1_color, labelsize=ft, labelcolor=line1_color)  # 调整主刻度字体及轴的长宽
    ax1.tick_params(axis='y', which='minor', length=4, color=line1_color, labelcolor=line1_color)  # 调整主刻度字体及轴的长宽
    ax1.tick_params(axis='x', length=5, width=1.3, labelsize=ft)  # 调整主刻度字体及轴的长宽
    # ax1.tick_params(axis='x',which='minor', length=4)  # 调整主刻度字体及轴的长宽
    ax1.xaxis.set_tick_params(which='minor', bottom=False)  # 关闭x轴
    # ax1.set_title(r'', loc='left', fontsize=ft)  # 左上title
    ax1.text(X[0] - (X[-1]-X[0])*labelloc[0], ylim1[1] + (ylim1[1]-ylim1[0])*labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold') #
    ax1.set_title(title1, loc='left', fontsize=ft + 3, fontweight='bold')  # 左上title
    ax1.set_title(title2, loc='center', fontsize=ft + 3, fontweight='bold')  # 左上title
    ax1.set_title(title3, loc='right', fontsize=ft + 3, fontweight='bold')  # 左上title
    ax1.axhline(0, linestyle='-', color='k', linewidth=1.2)
    # ax1.axhline(pdv.mean(), linestyle='--', color='r', linewidth=1)
    # ax1.grid(which='major', color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.set_ylabel(ylabel1, c=line1_color, fontsize=ft+1)
    ax1.set_xlabel('Percentile (%)', fontsize=ft+1)
    ax2.spines['left'].set_color(line1_color)
    ax1.set_ylim(ylim1)
    # 设置ax2
    ax2.spines['right'].set_color(line2_color)
    ax2.yaxis.set_major_locator(MultipleLocator(dy2))
    ax2.minorticks_on()  # 打开次坐标
    ax2.xaxis.set_tick_params(which='minor', bottom=False)
    ax2.tick_params(which='minor', length=4, color=line2_color, labelcolor=line2_color)  # 调整主刻度字体及轴的长宽
    ax2.set_ylabel(ylabel2, c=line2_color, fontsize=ft+1, rotation=270, labelpad=labelpad)
    ax2.tick_params(length=5, width=1.3, labelsize=ft, color=line2_color, labelcolor=line2_color)  # 调整主刻度字体及轴的长宽
    ax2.set_ylim(ylim2)
