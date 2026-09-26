
from pathlib import Path
import io
import numpy as np
import pandas as pd

try:
    import mne
except Exception:
    mne = None

BANDS = {"Delta":(0.5,4.0),"Theta":(4.0,8.0),"Alpha":(8.0,13.0),"Beta":(13.0,30.0)}

def load_eeg_file(upload):
    suffix=Path(upload.name).suffix.lower()
    if suffix == ".edf":
        if mne is None:
            raise RuntimeError("MNE-Python is not installed.")
        raw=mne.io.read_raw_edf(io.BytesIO(upload.getvalue()),preload=True,verbose="ERROR")
        return raw.get_data(),float(raw.info["sfreq"]),list(raw.ch_names)
    df=pd.read_csv(upload)
    numeric=df.select_dtypes(include=np.number)
    if numeric.empty:
        raise ValueError("CSV must contain numeric EEG columns.")
    return numeric.to_numpy(dtype=float).T,160.0,list(numeric.columns)

def band_features(signal,sfreq):
    x=np.nan_to_num(np.asarray(signal,dtype=float))
    x=x-np.mean(x)
    n=len(x)
    if n<32: raise ValueError("EEG signal is too short.")
    w=np.hanning(n)
    spec=np.fft.rfft(x*w)
    power=(np.abs(spec)**2)/n
    freqs=np.fft.rfftfreq(n,1/sfreq)
    vals=[]
    for lo,hi in BANDS.values():
        mask=(freqs>=lo)&(freqs<hi)
        vals.append(float(np.mean(power[mask])) if np.any(mask) else 0.0)
    total=sum(vals) or 1.0
    return np.asarray(vals)/total

def extract_band_features(data,sfreq):
    data=np.asarray(data)
    return np.mean(np.vstack([band_features(ch,sfreq) for ch in data[:min(8,len(data))]]),axis=0)
