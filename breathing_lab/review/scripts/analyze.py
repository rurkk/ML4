import json,numpy as np
from PIL import Image,ImageDraw
from pathlib import Path
root=Path(__file__).resolve().parent.parent
meta=json.loads((root/'frames.json').read_text());t=np.array(meta['times']);dt=np.median(np.diff(t));fs=1/dt
v=np.fromfile(root/'green.bin',np.uint8).reshape(-1,180,320).astype(np.float32)
a=(v[:-1]+v[1:])/2
gx=(a[:,:,2:]-a[:,:,:-2])/2;gy=(a[:,2:,:]-a[:,:-2,:])/2
gradx=gx[:,1:-1,:];grady=gy[:,:,1:-1];der=(v[1:]-v[:-1])[:,1:-1,1:-1]
patches=[('neck',126,184,58,84),('chest',122,205,80,133),('left_shoulder',104,143,68,99),('right_shoulder',182,218,70,103),('lower_chest',123,205,115,151)]
signals={}
for name,x0,x1,y0,y1 in patches:
 x=gradx[:,y0:y1,x0:x1].reshape(len(a),-1);y=grady[:,y0:y1,x0:x1].reshape(len(a),-1);z=der[:,y0:y1,x0:x1].reshape(len(a),-1)
 x=x-x.mean(1,keepdims=True);y=y-y.mean(1,keepdims=True);z=z-z.mean(1,keepdims=True)
 xx=(x*x).sum(1);yy=(y*y).sum(1);xy=(x*y).sum(1);xz=(x*z).sum(1);yz=(y*z).sum(1)
 den=xx*yy-xy*xy+1e-6
 vy=-(xx*yz-xy*xz)/den
 signals[name]=np.r_[0,np.cumsum(vy)]
# silhouette width/position by strongest shirt-background boundaries
for name,rows,start,stop in [('left_edge',range(90,145),85,140),('right_edge',range(90,145),200,242)]:
 g=np.diff(v[:,list(rows),:],axis=2)
 q=g[:,:,start:stop]
 pos=np.argmax(np.abs(q),axis=2)+start
 signals[name]=np.median(pos,axis=1)
# PCA on normalized chest intensity, separate overlapping segments to reduce posture drift
crop=v[:,76:150,116:210].reshape(len(v),-1)
crop=crop-crop.mean(1,keepdims=True)
# select pixels with structure using temporal mean gradient
rng=np.random.default_rng(0);sel=rng.choice(crop.shape[1],800,replace=False);X=crop[:,sel];X=X-X.mean(0)
u,s,vh=np.linalg.svd(X,full_matrices=False)
for i in range(4):signals['pca'+str(i)]=u[:,i]*s[i]

def filt(x,lo=.1,hi=.7):
 n=len(x);pad=min(100,n//3);xp=np.pad(x,(pad,pad),mode='reflect');xp=xp-np.mean(xp);f=np.fft.rfftfreq(len(xp),dt);w=(1/(1+(lo/np.maximum(f,1e-9))**8))*(1/(1+(f/hi)**8));return np.fft.irfft(np.fft.rfft(xp)*w,n=len(xp))[pad:-pad]
filtered={k:filt(x) for k,x in signals.items()}
def peaks(y):
 candidates=np.where((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:]))[0]+1
 keep=[]
 for j in candidates[np.argsort(y[candidates])[::-1]]:
  if all(abs(t[j]-t[k])>=1.5 for k in keep):keep.append(j)
 return sorted(keep)
for k,y in filtered.items():
 f=np.fft.rfftfreq(len(y),dt);p=abs(np.fft.rfft(y))**2;eligible=(f>.1)&(f<.7);hz=f[eligible][np.argmax(p[eligible])]
 print(k,'peaks',len(peaks(y)),'dominant_bpm',round(hz*60,1),'std',round(float(y.std()),2))
np.savez(root/'signals.npz',times=t,**filtered,**{'raw_'+k:v for k,v in signals.items()})
# overview plot, normalized each channel
W,H=1700,110*len(filtered)+70
im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
for r,(k,y) in enumerate(filtered.items()):
 cy=55+r*110;scale=np.percentile(abs(y),95)+1e-6
 d.text((8,cy-15),k,fill='black');d.line((120,cy,W-20,cy),fill='#dddddd')
 for sec in range(0,155,10):
  x=120+sec/t[-1]*(W-145);d.line((x,cy-42,x,cy+42),fill='#eeeeee');d.text((x,cy+43),str(sec),fill='black')
 pts=[(120+ti/t[-1]*(W-145),cy-float(np.clip(yi/scale,-1.5,1.5))*28) for ti,yi in zip(t,y)];d.line(pts,fill='#1550a0',width=2)
 for j in peaks(y):
  x=120+t[j]/t[-1]*(W-145);yy=cy-float(np.clip(y[j]/scale,-1.5,1.5))*28;d.ellipse((x-2,yy-2,x+2,yy+2),fill='red')
im.save(root/'signals.png')
