"""Small checks of units, time boundaries, signal processing and model direction."""
import json,numpy as np
from scipy.signal import butter,sosfiltfilt,find_peaks
from analysis import count_window,linear_detrend
from pipeline import ROOT

def verify(d):
 assert count_window(np.array([0.,30.,60.]),0,30)==1
 assert count_window(np.array([0.,30.,60.]),30,60)==1
 dt=.2;t=np.arange(0,180,dt);known=np.sin(2*np.pi*.3*t)+.02*t
 filtered=sosfiltfilt(butter(3,[.1,.7],btype='bandpass',fs=5,output='sos'),linear_detrend(known))
 ids,_=find_peaks(filtered,distance=7,prominence=.35*1.4826*np.median(abs(filtered-np.median(filtered))))
 rr=60*count_window(t[ids],10,170)/160;assert abs(rr-18)<=.375
 checks=json.loads((ROOT/'results/translation_check.json').read_text())
 max_error=max(max(abs(r['measured_u_px']-r['expected_u_px']),abs(r['measured_v_px']-r['expected_v_px'])) for r in checks)
 assert max_error<1.,checks
 assert d['start']>=0 and d['end']<=152.91806
 assert np.max(np.diff(np.array([r['time'] for r in d['records']])))<.24
 assert np.all(np.isfinite(d['yf'])) and np.all(np.diff(d['peaks'])>=1.4-1e-6)
 return dict(status='passed',half_open_window_boundaries=True,synthetic_RR_expected_bpm=18.,synthetic_RR_measured_bpm=rr,maximum_translation_error_px=max_error,peak_minimum_spacing_s=float(np.min(np.diff(d['peaks']))))
