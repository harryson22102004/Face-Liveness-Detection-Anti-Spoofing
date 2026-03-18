import torch, torch.nn as nn
import torchvision.models as models
import numpy as np
 
class FaceLivenessNet(nn.Module):
    """Binary face liveness detection (real vs spoof)."""
    def __init__(self):
        super().__init__()
        bb=models.mobilenet_v2(pretrained=False)
        bb.classifier=nn.Sequential(nn.Dropout(0.4),nn.Linear(1280,256),nn.ReLU(),nn.Linear(256,1))
        self.model=bb
    def forward(self,x): return torch.sigmoid(self.model(x)).squeeze()
 
class FrequencyAnalyser:
    """Detect spoofing via frequency domain analysis."""
    @staticmethod
    def compute_fft_features(img):
        if len(img.shape)==3: img=img.mean(axis=2)
        fft=np.fft.fft2(img); fft_shift=np.fft.fftshift(fft)
        magnitude=np.log(np.abs(fft_shift)+1)
        h,w=magnitude.shape; cy,cx=h//2,w//2
        radii=np.sqrt((np.arange(h)[:,None]-cy)**2+(np.arange(w)[None,:]-cx)**2)
        low=magnitude[radii<30].mean(); high=magnitude[radii>50].mean()
        return {'low_freq_energy':low,'high_freq_energy':high,'ratio':low/high}
 
def texture_liveness_score(img):
    """LBP-based texture analysis for liveness."""
    from collections import Counter
    if len(img.shape)==3: gray=img.mean(axis=2).astype(np.uint8)
    else: gray=img.astype(np.uint8)
    h,w=gray.shape; codes=[]
    for y in range(1,h-1):
        for x in range(1,w-1):
            center=gray[y,x]; code=0
            for i,(dy,dx) in enumerate([(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]):
               if gray[y+dy,x+dx]>=center: code|=(1<<i)
            codes.append(code)
    hist=Counter(codes); entropy=-sum((c/len(codes))*np.log2(c/len(codes)+1e-8) for c in hist.values())
    return entropy  # Real faces: higher entropy than printed/screen spoofs
 
model=FaceLivenessNet(); fa=FrequencyAnalyser()
x=torch.randn(4,3,224,224)
out=model(x)
print(f"Liveness scores: {out.detach().numpy().round(2)}")
print(f"{'LIVE' if out[0]>0.5 else 'SPOOF'} (score={out[0].item():.2f})")
img=np.random.rand(64,64,3)
fft_f=fa.compute_fft_features(img)
print(f"FFT features: {fft_f}")
lbp=texture_liveness_score(img)
print(f"LBP entropy (live-like if high): {lbp:.2f}")
