"""DIINA-Universal-v3 synthetic validation. NumPy fallback because PyTorch is unavailable.
Hand-calibrated variance prior, not a trained model and not evidence of generalization.
Real cross-lingual validation requires Common Voice Yoruba, LibriSpeech, and CHiME noise.
"""
import csv,time
from pathlib import Path
import numpy as np
def make_scenarios(seed=2025):
 r=np.random.default_rng(seed); T,D=100,64
 levels=np.repeat(np.array([-.8,-.2,.45,.95,.25],np.float32),20); load=r.normal(0,.20,D).astype(np.float32)
 tonal=levels[:,None]*load[None,:]+r.normal(0,.12,(T,D)).astype(np.float32)
 t=np.arange(T,dtype=np.float32); carrier=.95*np.sin(2*np.pi*t/7)+.55*np.sin(2*np.pi*t/3.5); fl=r.normal(0,.70,D).astype(np.float32)
 non=carrier[:,None]*fl[None,:]+r.normal(0,.55,(T,D)).astype(np.float32)
 noise=r.normal(0,15.,(T,D)).astype(np.float32)
 return {'Tonal (Yoruba/Chinese)':tonal[None], 'Non-tonal fast (English/Spanish)':non[None], 'Extreme ambient noise (std=15)':noise[None]}
def gate(x,knee,ceiling=.4135,scale=2.5):
 v=np.var(x,axis=-1); return np.clip(ceiling*np.exp(-np.maximum(v-knee,0)/scale),0,ceiling),v
def main():
 ss=make_scenarios(); knee=float(np.var(ss['Tonal (Yoruba/Chinese)'],axis=-1).max()+1e-7); rows=[]
 for x in ss.values(): gate(x,knee)
 for name,x in ss.items():
  start=time.perf_counter(); g,v=gate(x,knee); _=((1-g[:,:,None])*x+g[:,:,None]*x).mean(); latency=(time.perf_counter()-start)*1000
  a=float(g.mean()); conf=float(np.clip(.5+.4*np.exp(-v.mean(axis=1)/2),.5,.9).mean())
  rows.append({'scenario':name,'latency_ms':latency,'mean_gate_activation':a,'confidence_proxy':conf,'fast_path_utilization':1-a,'slow_path_utilization':a,'mean_instantaneous_variance':float(v.mean()),'variance_knee':knee,'frames':100,'feature_dim':64,'seed':2025,'gate_method':'hand-calibrated statistical variance prior; not trained'})
 with open('/mnt/data/diina_universal_validation_results.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print('DIINA-Universal-v3 | synthetic calibrated-prior stress test')
 print(f"{'Scenario':37s} {'Latency ms':>11s} {'Gate':>9s} {'Conf*':>9s} {'Fast %':>9s} {'Slow %':>9s} {'Mean var':>11s}")
 for z in rows: print(f"{z['scenario'][:37]:37s} {z['latency_ms']:11.3f} {z['mean_gate_activation']:9.4f} {z['confidence_proxy']:9.4f} {100*z['fast_path_utilization']:8.2f}% {100*z['slow_path_utilization']:8.2f}% {z['mean_instantaneous_variance']:11.3f}")
 print('Gate separation (tonal - extreme noise): %.4f'%(rows[0]['mean_gate_activation']-rows[2]['mean_gate_activation']))
 print('Verdict: synthetic prior separation only; not proof of universal robustness/generalization.')
 print('Conf* is a variance-derived proxy, not ASR confidence. Use Common Voice Yoruba, LibriSpeech, and CHiME noise for real validation.')
if __name__=='__main__': main()
