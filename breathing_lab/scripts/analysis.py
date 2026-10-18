from pathlib import Path
import json,csv
import numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks,periodogram
from scipy.optimize import linear_sum_assignment
from pipeline import ROOT,write_json
LOW,HIGH=.10,.70
FS=5.
MIN_DISTANCE=1.4
PROMINENCE=.35

def linear_detrend(v):
 t=np.linspace(-1,1,len(v));centered=v-np.mean(v)
 return centered-t*np.sum(t*centered)/np.sum(t*t)

def count_window(peaks,a,b):
 return int(np.sum((peaks>=a)&(peaks<b)))

def load_signal(cache):
 records=json.loads((cache/'motion.json').read_text());poses=json.loads((cache/'poses.json').read_text())
 t0=records[0]['time'];ends=np.array([r['end_time'] for r in records]);times=np.r_[t0,ends]
 dx=np.array([r['u'] for r in records]);dy=np.array([r['v'] for r in records])
 assert np.all(np.isfinite(dx)) and np.all(np.isfinite(dy)), 'Нет надёжной ROI; требуется обработка пропусков'
 assert np.all(np.diff(times)>0)
 t=np.round(np.arange(times[0],times[-1]+1e-8,1/FS),6)
 x=np.interp(t,times,np.r_[0,np.cumsum(dx)]);y=np.interp(t,times,np.r_[0,-np.cumsum(dy)])
 sos=butter(3,[LOW,HIGH],btype='bandpass',fs=FS,output='sos')
 xf=sosfiltfilt(sos,linear_detrend(x));yf=sosfiltfilt(sos,linear_detrend(y))
 mad=lambda v:1.4826*np.median(np.abs(v-np.median(v)))
 threshold=PROMINENCE*mad(yf)
 indices,properties=find_peaks(yf,distance=int(np.ceil(MIN_DISTANCE*FS)),prominence=threshold)
 peaks=t[indices]
 assert np.isfinite(xf).all() and np.isfinite(yf).all()
 return dict(records=records,poses=poses,t=t,x=x,y=y,xf=xf,yf=yf,peaks=peaks,indices=indices,threshold=threshold,
             duration=float(times[-1]-times[0]),start=float(times[0]),end=float(times[-1]),sos=sos)

def window_stats(d,a,b,label):
 sel=(d['t']>=a)&(d['t']<b);z=d['yf'][sel];freq,power=periodogram(z,fs=FS,detrend='constant')
 band=(freq>=LOW)&(freq<=HIGH);f=freq[band][np.argmax(power[band])]
 n=count_window(d['peaks'],a,b)
 raw=[r for r in d['records'] if a<=r['time']<b]
 return dict(window=label,start_s=a,end_s=b,duration_s=b-a,n_peaks=n,RR_est_bpm=n*60/(b-a),
  dominant_bpm=float(f*60),band_concentration=float(power[band&(np.abs(freq-f)<=.05)].sum()/power[band].sum()),
  filtered_rms_px=float(np.std(z)),flow_dispersion_px=float(np.median([r['flow_mad_px'] for r in raw])),
  brightness_range=float(np.ptp([r['brightness'] for r in raw])))

def manual_reference(video_sha):
 path=ROOT/'annotations/manual-confirmed.json'
 if not path.exists():return None
 doc=json.loads(path.read_text())
 assert doc['video_sha256']==video_sha and doc.get('independent_review') is True and doc.get('full_video_reviewed') is True
 assert doc['observer'].strip() and doc['event']=='end_inspiration'
 # A full independent review must explicitly resolve any ambiguous intervals before point metrics.
 marks=doc['marks'];assert not any(m['status']=='ambiguous' for m in marks),'Сначала разрешите неоднозначные интервалы эталона'
 assert all(m['status']=='human_confirmed' for m in marks)
 t=np.array([m['time_seconds'] for m in marks]);assert len(t)>1 and np.all(np.diff(t)>0)
 return t

def metrics(d,reference,windows):
 if reference is None:return dict(status='independent_reference_unavailable',RR_MAE=None,RR_bias=None)
 rows=[]
 for a,b,label in windows:
  est=count_window(d['peaks'],a,b)*60/(b-a);ref=count_window(reference,a,b)*60/(b-a)
  rows.append(dict(window=label,RR_est=est,RR_ref=ref,error_bpm=est-ref))
 errors=np.array([r['error_bpm'] for r in rows])
 return dict(status='evaluated',RR_MAE=float(np.mean(abs(errors))),RR_bias=float(errors.mean()),windows=rows)

def write_csv(path,rows):
 with Path(path).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(cache):
 d=load_signal(cache);out=ROOT/'results';out.mkdir(exist_ok=True)
 windows=[(2.,32.,'до: 2–32'),(32.,62.,'до: 32–62'),(72.,102.,'фонарик: 72–102'),(102.,132.,'фонарик: 102–132')]
 rows=[window_stats(d,*w) for w in windows]
 allrow=window_stats(d,d['start'],d['end'],'весь доступный сигнал')
 grid=[]
 for a in [round(d['start']+30*i,6) for i in range(int(np.ceil(d['duration']/30)))]:
  b=round(min(a+30,d['end']),6);grid.append(window_stats(d,float(a),float(b),f'{a:.1f}–{b:.1f}'))
 ref=manual_reference(json.loads((out/'video_audit.json').read_text())['sha256'])
 accuracy=metrics(d,ref,windows)
 sensitivity=[]
 for prom in [.2,.35,.5,.8]:
  ids,_=find_peaks(d['yf'],distance=int(np.ceil(MIN_DISTANCE*FS)),prominence=d['threshold']*prom/PROMINENCE)
  sensitivity.append(dict(prominence_mad=prom,n_peaks=len(ids),RR_est_bpm=len(ids)*60/d['duration']))
 poseconf=np.array([p['points'] for p in d['poses']])[:,[2,5],2]
 summary=dict(full=allrow,comparable_windows=rows,grid=grid,sensitivity=sensitivity,accuracy=accuracy,
  valid_pose_frames=sum(p['valid'] for p in d['poses']),total_pose_frames=len(d['poses']),
  minimum_shoulder_confidence=float(poseconf.min()),raw_pairs=len(d['records']),
  signal_start_s=d['start'],signal_end_s=d['end'],signal_duration_s=d['duration'],
  peak_prominence_px=d['threshold'],config=dict(low_hz=LOW,high_hz=HIGH,fs_hz=FS,min_distance_s=MIN_DISTANCE,prominence_mad=PROMINENCE))
 write_json(out/'summary.json',summary);write_csv(out/'windows.csv',rows);write_csv(out/'all_windows.csv',grid)
 write_csv(out/'detected-cycles.csv',[dict(cycle_number=i+1,time_seconds=float(t),status='neural_motion_peak_unconfirmed_phase') for i,t in enumerate(d['peaks'])])
 np.savez(out/'respiratory_signals.npz',times=d['t'],horizontal_raw_px=d['x'],vertical_up_raw_px=d['y'],horizontal_filtered_px=d['xf'],vertical_filtered_px=d['yf'],peak_times=d['peaks'])
 return d,summary

if __name__=='__main__':
 pointer=json.loads((ROOT/'results/cache_pointer.json').read_text());d,s=analyze(ROOT/pointer['cache']);print(json.dumps(s,ensure_ascii=False,indent=2))
