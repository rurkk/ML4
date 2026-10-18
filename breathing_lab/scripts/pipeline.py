"""OpenPose COCO + original SelFlow (TF1 inference via compat.v1), streaming video."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.metadata,resource
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','2')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/mpl'))
os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'cache'))
cv2.setNumThreads(4)
CONFIG=dict(sample_hz=5.,pose_hz=1.,flow_w=384,flow_h=256,image_w=384,image_h=216,
 pose_w=656,pose_h=368,pose_normalization='(BGR-128)/256',confidence=.15,roi_width=.72,roi_top=.18,roi_bottom=.80,
 threads=4)

def sha256(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

def write_json(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2))

def fingerprint():
 paths=[ROOT/'data/raw/ml-breath.mov',*sorted((ROOT/'models').glob('*')),
 *sorted((ROOT/'vendor').glob('*.py')),Path(__file__)]
 return dict(config=CONFIG,files={str(p.relative_to(ROOT)):sha256(p) for p in paths if p.is_file()},
             versions={x:importlib.metadata.version(x) for x in ['numpy','opencv-python-headless','tensorflow','tf_slim']})

def samples(stop=None,hz=None):
 cap=cv2.VideoCapture(str(ROOT/'data/raw/ml-breath.mov'),cv2.CAP_FFMPEG)
 assert cap.isOpened();hz=hz or CONFIG['sample_hz'];k=0;last=-1
 while True:
  ok,bgr=cap.read()
  if not ok:break
  t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
  assert t>last;last=t
  if stop is not None and t>=stop:break
  if t+1e-7<k/hz:continue
  yield t,bgr
  k+=1
 cap.release()

class Pose:
 def __init__(self):
  self.net=cv2.dnn.readNetFromCaffe(str(ROOT/'models/pose_deploy_linevec.prototxt'),str(ROOT/'models/pose_iter_440000.caffemodel'))
  self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
  self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
 def __call__(self,bgr):
  h,w=bgr.shape[:2]
  self.net.setInput(cv2.dnn.blobFromImage(bgr,1/256,(CONFIG['pose_w'],CONFIG['pose_h']),(128,128,128),swapRB=False,crop=False))
  out=self.net.forward()[0,:18]
  # One foreground person; independent heatmap maxima, without multi-person PAF grouping.
  points=[]
  for hm in out:
   _,v,_,loc=cv2.minMaxLoc(hm)
   points.append([loc[0]*w/hm.shape[1],loc[1]*h/hm.shape[0],v])
  return np.array(points)

def chest_roi(points):
 # COCO 2/5: shoulders. Axis-aligned chest box below shoulder midpoint.
 right,left=points[2],points[5]
 if min(right[2],left[2])<CONFIG['confidence']:return None
 width=np.linalg.norm(right[:2]-left[:2]);cx,cy=(right[:2]+left[:2])/2
 if not 100<width<700:return None
 box=np.array([cx-.5*CONFIG['roi_width']*width,cy+CONFIG['roi_top']*width,
               cx+.5*CONFIG['roi_width']*width,cy+CONFIG['roi_bottom']*width])
 box[[0,2]]=np.clip(box[[0,2]],0,1280);box[[1,3]]=np.clip(box[[1,3]],0,720)
 return box

class SelFlow:
 def __init__(self):
  import tensorflow as tf2
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior();tf.logging.set_verbosity(tf.logging.ERROR)
  sys.path.insert(0,str(ROOT/'vendor'))
  from network import pyramid_processing
  self.devices=[str(d) for d in tf2.config.list_physical_devices()]
  self.graph=tf.Graph()
  with self.graph.as_default():
   self.p=[tf.placeholder(tf.float32,[1,CONFIG['flow_h'],CONFIG['flow_w'],3]) for _ in range(3)]
   fw,bw=pyramid_processing(*self.p,train=False,trainable=False)
   self.out=[fw['full_res'],bw['full_res']]
   saver=tf.train.Saver()
   self.session=tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4,inter_op_parallelism_threads=2))
   saver.restore(self.session,str(ROOT/'models/supervise_finetune'))
 def __call__(self,imgs):
  return [x[0] for x in self.session.run(self.out,dict(zip(self.p,[i[None] for i in imgs])))]
 def close(self):self.session.close()

def preprocess(bgr):
 rgb=cv2.cvtColor(cv2.resize(bgr,(CONFIG['image_w'],CONFIG['image_h']),interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2RGB).astype(np.float32)/255
 # Match official supervised checkpoint: normalize each channel per frame before padding.
 rgb=(rgb-rgb.mean(axis=(0,1),keepdims=True))/np.sqrt(rgb.var(axis=(0,1),keepdims=True)+1e-12)
 return np.pad(rgb,((0,CONFIG['flow_h']-CONFIG['image_h']),(0,0),(0,0)),mode='edge')

def flow_summary(fw,bw,box):
 sx=CONFIG['image_w']/1280;sy=CONFIG['image_h']/720
 x0,y0,x1,y1=np.rint(box*[sx,sy,sx,sy]).astype(int)
 assert x1>x0 and y1>y0
 roi=fw[y0:y1,x0:x1].copy();roi[...,0]/=sx;roi[...,1]/=sy
 med=np.median(roi,axis=(0,1));spread=np.median(np.linalg.norm(roi-med,axis=-1))
 # Background reference: top-left wall, static camera. No subtraction in the main signal.
 bg=fw[10:50,10:90].copy();bg[...,0]/=sx;bg[...,1]/=sy
 return dict(u=float(med[0]),v=float(med[1]),flow_mad_px=float(spread),
             bg_u=float(np.median(bg[...,0])),bg_v=float(np.median(bg[...,1])),
             roi=box.tolist())

def infer(stop,pose,flow,save_examples=False):
 records=[];poses=[];buf=[];box=None;pose_next=0.;p_times=[];f_times=[];peak_rss=0
 for t,bgr in samples(stop):
  if t+1e-7>=pose_next:
   st=time.monotonic();pts=pose(bgr);p_times.append(time.monotonic()-st)
   new=chest_roi(pts)
   if new is not None:box=new
   poses.append(dict(time=t,points=pts.tolist(),valid=new is not None,roi=None if new is None else new.tolist()))
   pose_next+=1/CONFIG['pose_hz']
   if save_examples and any(abs(t-z)<.15 for z in [2,20,64,68,90,132,150]):
    render=bgr.copy()
    for i in [0,1,2,5,8,11]:
     x,y,c=pts[i]
     if c>=CONFIG['confidence']:cv2.circle(render,(int(x),int(y)),6,(0,255,255),-1);cv2.putText(render,str(i),(int(x)+6,int(y)),0,.6,(0,255,255),2)
    if new is not None:
     x0,y0,x1,y1=new.astype(int);cv2.rectangle(render,(x0,y0),(x1,y1),(0,220,0),3)
    cv2.imwrite(str(ROOT/f'results/pose_{t:06.2f}.jpg'),render)
  buf.append((t,preprocess(bgr),None if box is None else box.copy(),float(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).mean())))
  if len(buf)<3:continue
  st=time.monotonic();fw,bw=flow([b[1] for b in buf]);f_times.append(time.monotonic()-st)
  prev,center,nxt=buf
  if center[2] is None:rec=dict(u=np.nan,v=np.nan,flow_mad_px=np.nan,bg_u=np.nan,bg_v=np.nan,roi=None)
  else:rec=flow_summary(fw,bw,center[2])
  rec.update(time=center[0],end_time=nxt[0],dt=nxt[0]-center[0],brightness=center[3]);records.append(rec)
  peak_rss=max(peak_rss,resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
  buf.pop(0)
  if len(records)%100==0:print(f'{len(records)} pairs, video {t:.1f} s',flush=True)
 return records,poses,dict(pose_s=sum(p_times),flow_s=sum(f_times),n_pose=len(p_times),n_flow=len(f_times),
                          pose_median_s=float(np.median(p_times)),flow_median_s=float(np.median(f_times)),
                          peak_rss_mib=peak_rss/2**20 if sys.platform=='darwin' else peak_rss/1024)

def run_benchmark():
 st=time.monotonic();pose=Pose();flow=SelFlow();setup=time.monotonic()-st
 first=next(samples())[1]
 warm=time.monotonic();pose(first);flow([preprocess(first)]*3);warm=time.monotonic()-warm
 st=time.monotonic();records,poses,timing=infer(12.,pose,flow);wall=time.monotonic()-st
 duration=152.91805
 projected=timing['pose_s']/12*duration+timing['flow_s']/12*duration
 info=dict(status='completed',test_video_s=12.,wall_s=wall,model_load_s=setup,warmup_s=warm,
           projected_neural_full_s=projected,projected_wall_full_s=wall/12*duration,
           devices=flow.devices,**timing,config=CONFIG,fingerprint=fingerprint())
 write_json(ROOT/'results/benchmark.json',info);write_json(ROOT/'results/test_poses.json',poses)
 write_json(ROOT/'results/test_motion.json',records);flow.close()
 print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
 return info

def run_full():
 manifest=fingerprint();key=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
 cache=ROOT/'cache'/key;cache.mkdir(parents=True,exist_ok=True)
 if (cache/'complete.json').exists():return cache
 bench=json.loads((ROOT/'results/benchmark.json').read_text())
 assert bench['fingerprint']==manifest,'Повторите тест: параметры/зависимости изменились'
 assert max(bench['projected_neural_full_s'],bench['projected_wall_full_s'])<=1800,'STOP: нужен Colab GPU'
 pose=Pose();flow=SelFlow();first=next(samples())[1];pose(first);flow([preprocess(first)]*3)
 st=time.monotonic();records,poses,timing=infer(None,pose,flow,True)
 timing['wall_s']=time.monotonic()-st;flow.close()
 write_json(cache/'motion.json',records);write_json(cache/'poses.json',poses)
 write_json(cache/'manifest.json',manifest);write_json(cache/'complete.json',timing)
 write_json(ROOT/'results/cache_pointer.json',dict(key=key,cache=str(cache.relative_to(ROOT))))
 return cache

if __name__=='__main__':
 if sys.argv[-1]=='benchmark':run_benchmark()
 elif sys.argv[-1]=='full':print(run_full())
