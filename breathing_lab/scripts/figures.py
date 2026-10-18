import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pipeline import ROOT
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120})

def save(fig,name):
 fig.savefig(ROOT/'results'/f'{name}.png',dpi=150,bbox_inches='tight')
 return fig

def table(rows,columns=None):
 from IPython.display import display,HTML
 import html
 columns=columns or list(rows[0])
 def fmt(x):return f'{x:.3f}' if isinstance(x,float) else str(x)
 display(HTML('<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(fmt(r.get(k,'')))+'</td>' for k in columns)+'</tr>' for r in rows)+'</tbody></table>'))

def events():return list(csv.DictReader((ROOT/'annotations/interference-intervals.csv').open()))

def shade(ax):
 for a,b,c in [(62,63.5,'#b2abd2'),(64.5,66.5,'#fdae61'),(128,131,'#f46d43'),(148.5,150.5,'#b2abd2')]:ax.axvspan(a,b,color=c,alpha=.22)
 ax.axvline(65,color='#777',ls='--',lw=.8,label='≈65 с: фонарик по сценарию')

def architecture():
 from matplotlib.patches import FancyBboxPatch
 fig,ax=plt.subplots(figsize=(12,3.2));ax.axis('off')
 boxes=['Видео + PTS\n5 кадров/с','OpenPose COCO\nплечи, ROI: 1 раз/с','SelFlow\nтри кадра, Δx, Δy','Медиана в ROI\nмасштаб → пиксели','Интеграл, тренд\n0,10–0,70 Гц','Максимумы\nRR = 60N / T']
 for i,s in enumerate(boxes):
  x=i*2.05;ax.add_patch(FancyBboxPatch((x,1),1.8,1,boxstyle='round,pad=.08',facecolor='#e6eff9',edgecolor='#41658b'));ax.text(x+.9,1.5,s,ha='center',va='center')
  if i<5:ax.annotate('',(x+2.0,1.5),(x+1.86,1.5),arrowprops={'arrowstyle':'->'})
 ax.set(xlim=(-.2,12.3),ylim=(.8,2.4))
 return save(fig,'architecture')

def frame_at(t):
 cap=cv2.VideoCapture(str(ROOT/'data/raw/ml-breath.mov'),cv2.CAP_FFMPEG);cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();actual=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;cap.release();assert ok
 return cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),actual

def sample_frames():
 fig,axes=plt.subplots(2,3,figsize=(12,5.3))
 for ax,t in zip(axes.flat,[20.,62.5,65.5,90.,129.,150.]):
  im,actual=frame_at(t);ax.imshow(im);ax.set_title(f'{actual:.2f} с');ax.axis('off')
 fig.tight_layout();return save(fig,'scenario_frames')

def pose_frames():
 paths=sorted((ROOT/'results').glob('pose_*.jpg'))
 chosen=[min(paths,key=lambda p:abs(float(p.stem.split('_')[1])-t)) for t in [2,68,132]]
 fig,axes=plt.subplots(1,3,figsize=(12,3))
 for ax,p in zip(axes,chosen):ax.imshow(plt.imread(p));ax.set_title(p.stem.split('_')[1]+' с');ax.axis('off')
 fig.tight_layout();return save(fig,'keypoints_roi')

def overview(d):
 fig,axes=plt.subplots(4,1,figsize=(12,8.7),sharex=True)
 axes[0].plot(d['t'],d['x'],label='x: вправо');axes[0].plot(d['t'],d['y'],label='y: вверх');axes[0].set_ylabel('Интеграл, пикс.');axes[0].legend(loc='upper left',ncol=2)
 axes[1].plot(d['t'],d['xf'],label='x');axes[1].plot(d['t'],d['yf'],label='y вверх');axes[1].set_ylabel('Полоса, пикс.');axes[1].legend(loc='upper left',ncol=2)
 axes[2].plot(d['t'],d['yf'],color='#245b91');axes[2].scatter(d['peaks'],d['yf'][d['indices']],s=14,color='#ce4040',label='максимумы SelFlow');axes[2].set_ylabel('Сигнал, пикс.')
 assist=list(csv.DictReader((ROOT/'annotations/breath-cycle-assisted.csv').open()));at=np.array([float(r['time_seconds']) for r in assist]);axes[2].plot(at,np.full(len(at),np.min(d['yf'])-.2),'|',color='#666',ms=6,label='вспомогательные отметки');axes[2].legend(loc='upper left',ncol=2)
 bt=np.array([[r['time'],r['brightness']] for r in d['records']]);axes[3].plot(bt[:,0],bt[:,1],color='#a37a27');axes[3].set_ylabel('Яркость, 0–255');axes[3].set_xlabel('Время от начала видео, с')
 for ax in axes:shade(ax);ax.grid(alpha=.2);ax.set_xlim(0,153)
 fig.tight_layout();return save(fig,'signals_overview')

def comparisons(d,summary):
 fig,axes=plt.subplots(2,2,figsize=(12,5),sharey=True)
 for ax,row in zip(axes.flat,summary['comparable_windows']):
  a,b=row['start_s'],row['end_s'];sel=(d['t']>=a)&(d['t']<b);p=d['peaks'][(d['peaks']>=a)&(d['peaks']<b)]
  ax.plot(d['t'][sel]-a,d['yf'][sel],color='#245b91');ax.scatter(p-a,np.interp(p,d['t'],d['yf']),s=17,color='#ce4040')
  ax.set_title(f"{row['window']} с: {row['n_peaks']} макс., {row['RR_est_bpm']:.1f} мин⁻¹");ax.set_xlabel('Время внутри окна, с');ax.set_ylabel('Сигнал, пикс.');ax.grid(alpha=.2)
 fig.tight_layout();return save(fig,'comparable_windows')

def diagnostics(d):
 fig,axes=plt.subplots(3,1,figsize=(12,6.5))
 for ax,(a,b) in zip(axes,[(60,70),(124,136),(145,152.8)]):
  sel=(d['t']>=a)&(d['t']<=b);ax.plot(d['t'][sel],d['yf'][sel],label='вертикальный сигнал');ax.plot(d['t'][sel],d['xf'][sel],alpha=.6,label='горизонтальный сигнал')
  p=d['peaks'][(d['peaks']>=a)&(d['peaks']<=b)];ax.scatter(p,np.interp(p,d['t'],d['yf']),color='#ce4040',s=20)
  shade(ax);ax.set(xlim=(a,b),ylabel='Пиксели',xlabel='Время, с');ax.grid(alpha=.2)
 axes[0].legend(loc='upper left');fig.tight_layout();return save(fig,'diagnostic_intervals')

def problem_frames():
 fig,axes=plt.subplots(2,3,figsize=(12,5.3))
 for ax,t in zip(axes.flat,[64.5,128.,148.5,66.5,131.,150.5]):
  im,actual=frame_at(t);ax.imshow(im);ax.set_title(f'{actual:.2f} с');ax.axis('off')
 fig.tight_layout();return save(fig,'problem_frames')
