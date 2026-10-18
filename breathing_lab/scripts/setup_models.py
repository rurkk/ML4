"""Download exact pretrained models; no training, no changes to source video."""
from pathlib import Path
import hashlib,requests,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
OP='5c5d96523ef917bd30301245fdc8343937cae48d'
SF='0525a2a61b07785bb7c9d6c278769c218c57fc35'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def download(url,p,expected=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and (expected is None or sha(p)==expected):return
 r=requests.get(url,timeout=(20,180));r.raise_for_status();tmp=p.with_suffix(p.suffix+'.part');tmp.write_bytes(r.content)
 if expected:assert sha(tmp)==expected
 tmp.replace(p)

def prepare():
 base=f'https://raw.githubusercontent.com/ppliuboy/SelFlow/{SF}'
 for name in ['network.py','warp.py','utils.py','data_augmentation.py']:
  r=requests.get(f'{base}/{name}',timeout=30);r.raise_for_status()
  source=r.text.replace('import tensorflow as tf','import tensorflow.compat.v1 as tf').replace('from tensorflow.contrib import slim','import tf_slim as slim')
  (ROOT/'vendor').mkdir(exist_ok=True);(ROOT/'vendor'/name).write_text(source)
 download(f'{base}/LICENSE',ROOT/'vendor/LICENSE')
 for name,expected in [('supervise_finetune.index','c7295bb34960248cbe2cab2874d5b0e4734e57b736651cee5fbc4f06b0d09460'),('supervise_finetune.data-00000-of-00001','ba5c1ae8f65c477c238f3760140b5122df297ce3736ca3910439c73efc09d332')]:
  download(f'{base}/models/Sintel/{name}',ROOT/'models'/name,expected)
 download(f'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/{OP}/models/pose/coco/pose_deploy_linevec.prototxt',ROOT/'models/pose_deploy_linevec.prototxt','17051b87f709aa094e09c5da7b78e9016a1f37b2b452ed1f190fe74cce70b1ad')
 download(f'https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/{OP}/LICENSE',ROOT/'vendor/OpenPose-LICENSE')
 # Original Caffe weights; mirror pinned by revision and verified by content hash.
 download('https://huggingface.co/gaijingeek/openpose-models/resolve/7b606a341a4dcf057c98ebcb5c17b53b1d7c7cc9/pose/coco/pose_iter_440000.caffemodel',ROOT/'models/pose_iter_440000.caffemodel','b4cf475576abd7b15d5316f1ee65eb492b5c9f5865e70a2e7882ed31fb682549')

if __name__=='__main__':prepare()
