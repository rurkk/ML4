from pipeline import ROOT,write_json,sha256
import cv2,numpy as np,time

def audit():
 p=ROOT/'data/raw/ml-breath.mov';cap=cv2.VideoCapture(str(p),cv2.CAP_FFMPEG)
 assert cap.isOpened()
 meta=dict(width=int(cap.get(3)),height=int(cap.get(4)),fps=cap.get(cv2.CAP_PROP_FPS),frame_count_header=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),fourcc=''.join(chr((int(cap.get(cv2.CAP_PROP_FOURCC))>>(8*i))&255) for i in range(4)),backend=cap.getBackendName())
 start=time.monotonic();ts=[];brightness=[];k=0
 while True:
  ok,bgr=cap.read()
  if not ok:break
  t=cap.get(cv2.CAP_PROP_POS_MSEC)/1000;ts.append(t)
  if t>=k:
   brightness.append([t,float(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).mean()),float(cv2.cvtColor(bgr[260:560,480:780],cv2.COLOR_BGR2GRAY).mean())]);k+=1
 cap.release()
 meta.update(decoded_frames=len(ts),first_pts_s=ts[0],last_pts_s=ts[-1],median_frame_dt_s=float(np.median(np.diff(ts))),duration_pts_s=float(ts[-1]+np.median(np.diff(ts))),decode_s=time.monotonic()-start,bytes=p.stat().st_size,sha256=sha256(p),pts_strictly_increasing=bool(np.all(np.diff(ts)>0)),min_dt_s=float(np.min(np.diff(ts))),max_dt_s=float(np.max(np.diff(ts))))
 assert meta['decoded_frames']==meta['frame_count_header'] and meta['pts_strictly_increasing']
 write_json(ROOT/'results/video_audit.json',meta);np.savez(ROOT/'results/video_timestamps.npz',pts=ts,brightness=brightness)
 return meta
