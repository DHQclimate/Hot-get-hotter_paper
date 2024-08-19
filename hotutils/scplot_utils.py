import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as patches
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
import myfunction as mf
import matplotlib as mpl


def sh_month(ds):
    ds_sh = np.hstack((ds[6:],ds[:6]))
    return ds_sh

def cm_scn(fig,dI,dI0,M,err,y1,y2,dy1,dy2,labelloc,tlabel,title1,title2,title3,unit1,unit2,ft,i,j,k,barlabel='Change',if_var='pr',if_legend=False,direct='out'):
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1 = fig.add_subplot(i,j,k)
    ax2 = ax1.twinx()
    x = np.arange(1, 13)
    if if_var == 'pr':
        colormat = np.where(dI > 0, '#6DBFB4', '#D8B46C')
        ax1.axvspan(3.5, 6.5, alpha=0.18, color='#D8B46C')
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='#6DBFB4', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    elif if_var == 'tx':
        colormat = np.where(dI > 0, 'orange', '#00B2E9')
        ax1.axvspan(3.5, 6.5, alpha=0.13, color='orange')
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='orange', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    else:
        print('error: if_var must be (pr) or (tx)')
    if barlabel == 'Change':
        hatchmat = np.where(M < 0.05, '///', '')
    else:
        hatchmat = np.where(M < 0.1, '///', '')
    ax1.bar(x, dI, width=0.55, alpha=1, color=colormat, align='center', zorder=0,hatch=hatchmat) #, hatch=hatchmat
    ax1.errorbar(x,dI,yerr=err,ecolor='gray',capsize=4,ls='none',elinewidth=1.2)
    ax1.axhline(0, color='k', linewidth=1.5)
    ax2.plot(x, dI0, color='k', marker='.', linewidth=2, zorder=1)
    plt.xticks(x, labels)  # 更换x坐标
    ax1.tick_params(length=4, width=1.1, labelsize=ft,direction=direct)  # 调整主刻度字体及轴的长宽
    ax2.tick_params(length=4, width=1.1, labelsize=ft,direction=direct)  # 调整主刻度字体及轴的长宽
    ax1.set_title(title1, loc='left',fontsize=ft+6,fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title2, loc='center', fontsize=ft+6, fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title3, loc='right', fontsize=ft+6,fontweight='bold')  # 右上title ,fontweight='bold'

    ax1.text(labelloc[0], labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')

    ax1.set_xlim([0.5, 12.5])
    ax1.set_ylim(-y1,y1)
    ax2.set_ylim(y2)
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax2.yaxis.set_major_locator(MultipleLocator(dy2))#设置y轴数值间隔
    ax1.set_ylabel(unit1, fontsize=ft+4)
    ax2.set_ylabel(unit2, fontsize=ft+4, rotation=270, labelpad=23)

def cm_scs(fig,dI,dI0,M,err,y1,y2,dy1,dy2,labelloc,tlabel,title1,title2,title3,unit1,unit2,ft,i,j,k,barlabel='Change',if_var='pr',if_legend=False,direct='out'):
    labels = [ 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    dI  = sh_month(dI)
    dI0 = sh_month(dI0)
    M   = sh_month(M)
    err = [sh_month(err[0]),sh_month(err[1])]
    ax1 = fig.add_subplot(i,j,k)
    ax2 = ax1.twinx()
    x = np.arange(1, 13)
    if if_var == 'pr':
        colormat = np.where(dI > 0, '#6DBFB4', '#D8B46C')
        ax1.axvspan(2.5, 5.5, alpha=0.18, color='#D8B46C',zorder=10)
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='#6DBFB4', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    elif if_var == 'tx':
        colormat = np.where(dI > 0, 'orange', '#00B2E9')
        ax1.axvspan(2.5, 5.5, alpha=0.13, color='orange',zorder=10)
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='orange', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    else:
        print('error: if_var must be (pr) or (tx)')
    if barlabel == 'Change':
        hatchmat = np.where(M < 0.05, '///', '')
    else:
        hatchmat = np.where(M < 0.1, '///', '')
    ax1.bar(x, dI, width=0.55, alpha=1, color=colormat, align='center', zorder=0,hatch=hatchmat) #, hatch=hatchmat
    ax1.errorbar(x,dI,yerr=err,ecolor='gray',capsize=4,ls='none',elinewidth=1.2)
    ax1.axhline(0, color='k', linewidth=1.5)
    ax2.plot(x, dI0, color='k', marker='.', linewidth=2, zorder=1)
    plt.xticks(x, labels)  # 更换x坐标
    ax1.tick_params(length=4, width=1.1, labelsize=ft,direction=direct)  # 调整主刻度字体及轴的长宽
    ax2.tick_params(length=4, width=1.1, labelsize=ft,direction=direct)  # 调整主刻度字体及轴的长宽
    ax1.set_title(title1, loc='left',fontsize=ft+6,fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title2, loc='center', fontsize=ft+6, fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title3, loc='right', fontsize=ft+6,fontweight='bold')  # 右上title ,fontweight='bold'

    ax1.text(labelloc[0], labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')


    ax1.set_xlim([0.5, 12.5])
    ax1.set_ylim(-y1,y1)
    ax2.set_ylim(y2)
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax2.yaxis.set_major_locator(MultipleLocator(dy2))#设置y轴数值间隔
    ax1.set_ylabel(unit1, fontsize=ft+4)
    ax2.set_ylabel(unit2, fontsize=ft+4, rotation=270, labelpad=23)


def ax_scn(fig,dI,dI0,M,err,y1,y2,dy1,dy2,labelloc,tlabel,title1,title2,title3,unit1,unit2,ft,x11,x22,y11,y22,barlabel='Change',if_var='pr',if_legend=False):
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1 = fig.add_axes([x11, y11, x22-x11, y22-y11])
    ax2 = ax1.twinx()
    x = np.arange(1, 13)
    if if_var == 'pr':
        colormat = np.where(dI > 0, '#6DBFB4', '#D8B46C')
        ax1.axvspan(3.5, 6.5, alpha=0.18, color='#D8B46C')
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='#6DBFB4', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    elif if_var == 'tx':
        colormat = np.where(dI > 0, 'orange', '#00B2E9')
        ax1.axvspan(3.5, 6.5, alpha=0.13, color='orange')
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='orange', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    else:
        print('error: if_var must be (pr) or (tx)')
    if barlabel == 'Change':
        hatchmat = np.where(M < 0.05, '///', '')
    else:
        hatchmat = np.where(M < 0.1, '///', '')
    ax1.bar(x, dI, width=0.55, alpha=1, color=colormat, align='center', zorder=0,hatch=hatchmat) #, hatch=hatchmat
    ax1.errorbar(x,dI,yerr=err,ecolor='gray',capsize=4,ls='none',elinewidth=1.2)
    ax1.axhline(0, color='k', linewidth=1.5)
    ax2.plot(x, dI0, color='k', marker='.', linewidth=2, zorder=1)
    plt.xticks(x, labels)  # 更换x坐标
    ax1.tick_params(length=3, width=1.1, labelsize=ft)  # 调整主刻度字体及轴的长宽
    ax2.tick_params(length=3, width=1.1, labelsize=ft)  # 调整主刻度字体及轴的长宽
    ax1.set_title(title1, loc='left',fontsize=ft+6,fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title2, loc='center', fontsize=ft+6, fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title3, loc='right', fontsize=ft+6,fontweight='bold')  # 右上title ,fontweight='bold'

    ax1.text(0.5 - 12*labelloc[0], y1 + 2*y1*labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')

    ax1.set_xlim([0.5, 12.5])
    ax1.set_ylim(-y1,y1)
    ax2.set_ylim(y2)
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax2.yaxis.set_major_locator(MultipleLocator(dy2))#设置y轴数值间隔
    ax1.set_ylabel(unit1, fontsize=ft)
    ax2.set_ylabel(unit2, fontsize=ft, rotation=270, labelpad=15)

def ax_scs(fig,dI,dI0,M,err,y1,y2,dy1,dy2,labelloc,tlabel,title1,title2,title3,unit1,unit2,ft,x11,x22,y11,y22,barlabel='Change',if_var='pr',if_legend=False):
    labels = [ 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    dI  = sh_month(dI)
    dI0 = sh_month(dI0)
    M   = sh_month(M)
    err = [sh_month(err[0]),sh_month(err[1])]
    ax1 = fig.add_axes([x11, y11, x22-x11, y22-y11])
    ax2 = ax1.twinx()
    x = np.arange(1, 13)
    if if_var == 'pr':
        colormat = np.where(dI > 0, '#6DBFB4', '#D8B46C')
        ax1.axvspan(2.5, 5.5, alpha=0.18, color='#D8B46C',zorder=10)
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='#6DBFB4', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    elif if_var == 'tx':
        colormat = np.where(dI > 0, 'orange', '#00B2E9')
        ax1.axvspan(2.5, 5.5, alpha=0.13, color='orange',zorder=10)
        if if_legend :
            ax1.bar(-1, 2, width=0.6, alpha=1, color='orange', align='center', label=barlabel)
            ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
            ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False
    else:
        print('error: if_var must be (pr) or (tx)')
    if barlabel == 'Change':
        hatchmat = np.where(M < 0.05, '///', '')
    else:
        hatchmat = np.where(M < 0.1, '///', '')
    ax1.bar(x, dI, width=0.55, alpha=1, color=colormat, align='center', zorder=0,hatch=hatchmat) #, hatch=hatchmat
    ax1.errorbar(x,dI,yerr=err,ecolor='gray',capsize=4,ls='none',elinewidth=1.2)
    ax1.axhline(0, color='k', linewidth=1.5)
    ax2.plot(x, dI0, color='k', marker='.', linewidth=2, zorder=1)
    plt.xticks(x, labels)  # 更换x坐标
    ax1.tick_params(length=3, width=1.1, labelsize=ft)  # 调整主刻度字体及轴的长宽
    ax2.tick_params(length=3, width=1.1, labelsize=ft)  # 调整主刻度字体及轴的长宽
    ax1.set_title(title1, loc='left',fontsize=ft+6,fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title2, loc='center', fontsize=ft+6, fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title3, loc='right', fontsize=ft+6,fontweight='bold')  # 右上title ,fontweight='bold'

    ax1.text(0.5 - 12*labelloc[0], y1 + 2*y1*labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')


    ax1.set_xlim([0.5, 12.5])
    ax1.set_ylim(-y1,y1)
    ax2.set_ylim(y2)
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax2.yaxis.set_major_locator(MultipleLocator(dy2))#设置y轴数值间隔
    ax1.set_ylabel(unit1, fontsize=ft)
    ax2.set_ylabel(unit2, fontsize=ft, rotation=270, labelpad=15)


def sa_scn(fig,dI,M,err,cte,x_c,y_c,y1,dy1,labelloc,tlabel,title1,title2,title3,unit1,ft,i,j,k,barlabel='Change',if_legend=False,if_cte=False,direct='out'):
    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1 = fig.add_subplot(i,j,k)
    x = np.arange(1, 13)
    colormat = np.where(dI > 0,'#D00000', '#2F7EB0') #  '#F37252', '#6F7EB9' , '#E93E44','#3D78ED'
    ax1.axvspan(3.5, 6.5, alpha=0.18, color='lightgray')
    if if_legend :
        ax1.bar(-1, 2, width=0.6, alpha=1, color='#6DBFB4', align='center', label=barlabel)
        ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
        ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False

    hatchmat = np.where(M < 0.05, '///', '')
    ax1.bar(x, dI, width=0.55, alpha=1, color=colormat, align='center', zorder=0, hatch=hatchmat) #, hatch=hatchmat
    ax1.errorbar(x,dI,yerr=err,ecolor='gray',capsize=4,ls='none',elinewidth=1.2)
    ax1.axhline(0, color='k', linewidth=1.5)
    plt.xticks(x, labels)  # 更换x坐标
    ax1.tick_params(length=4, width=1.1, labelsize=ft,direction=direct)  # 调整主刻度字体及轴的长宽
    ax1.set_title(title1, loc='left',fontsize=ft+6,fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title2, loc='center', fontsize=ft+6, fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title3, loc='right', fontsize=ft+6,fontweight='bold')  # 右上title ,fontweight='bold'

    ax1.text(labelloc[0], labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')
    if if_cte:
        if len(str(round(float(cte), 2))) == 4 :
            ax1.text(x_c, y_c, '' + str(round(float(cte*100))) + '%', fontsize=ft+3,fontweight='bold',horizontalalignment='center')  
        else:
            ax1.text(x_c, y_c, '' + str(round(float(cte*100))) + '%', fontsize=ft+3, fontweight='bold',horizontalalignment='center')

    ax1.set_xlim([0.5, 12.5])
    ax1.set_ylim(-y1,y1)
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax1.set_ylabel(unit1, fontsize=ft+4)


def sa_scs(fig,dI,M,err,cte,x_c,y_c,y1,dy1,labelloc,tlabel,title1,title2,title3,unit1,ft,i,j,k,barlabel='Change',if_legend=False,if_cte=False,direct='out'):
    labels = [ 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
    dI  = sh_month(dI)
    err = [sh_month(err[0]),sh_month(err[1])]
    M   = sh_month(M)
    ax1 = fig.add_subplot(i,j,k)
    x = np.arange(1, 13)
    colormat = np.where(dI > 0,'#D00000','#2F7EB0') # '#F37252', '#6F7EB9' '#E93E44','#3D78ED'
    ax1.axvspan(2.5, 5.5, alpha=0.18, color='lightgray',zorder=10)
    if if_legend :
        ax1.bar(-1, 2, width=0.6, alpha=1, color='#6DBFB4', align='center', label=barlabel)
        ax1.plot(-1, 1, color='k', marker='.', linewidth=2, zorder=1,label='Climatology')
        ax1.legend(fontsize=ft-2, handlelength=2.5,loc='upper right') #,ncol=2 , frameon=False

    hatchmat = np.where(M < 0.05, '///', '')
    ax1.bar(x, dI, width=0.55, alpha=1, color=colormat, align='center', zorder=0, hatch=hatchmat) #, hatch=hatchmat
    ax1.errorbar(x,dI,yerr=err,ecolor='gray',capsize=4,ls='none',elinewidth=1.2)
    ax1.axhline(0, color='k', linewidth=1.5)
    plt.xticks(x, labels)  # 更换x坐标
    ax1.tick_params(length=4, width=1.1, labelsize=ft,direction=direct)  # 调整主刻度字体及轴的长宽
    ax1.set_title(title1, loc='left',fontsize=ft+6,fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title2, loc='center', fontsize=ft+6, fontweight='bold')  # 左上title ,fontweight='bold'
    ax1.set_title(title3, loc='right', fontsize=ft+6,fontweight='bold')  # 右上title ,fontweight='bold'

    ax1.text(labelloc[0], labelloc[1], tlabel, fontsize=ft+8, color='k', fontweight='bold')
    if if_cte:
        if len(str(round(float(cte), 2))) == 4 :
            ax1.text(x_c, y_c, '' + str(round(float(cte*100))) + '%', fontsize=ft+3,fontweight='bold',horizontalalignment='center')  
        else:
            ax1.text(x_c, y_c, '' + str(round(float(cte*100))) + '%', fontsize=ft+3, fontweight='bold',horizontalalignment='center')

    ax1.set_xlim([0.5, 12.5])
    ax1.set_ylim(-y1,y1)
    ax1.yaxis.set_major_locator(MultipleLocator(dy1))#设置y轴数值间隔
    ax1.set_ylabel(unit1, fontsize=ft+4)
