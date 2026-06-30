"""
SSIM Extension Paper 
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data as skdata
from skimage.restoration import denoise_nl_means, estimate_sigma
import warnings, os, json
warnings.filterwarnings('ignore')

import torch
import piq

os.makedirs('/home/claude/figures', exist_ok=True)

plt.rcParams.update({
    'font.family':'serif','font.size':9,'axes.titlesize':9,'axes.labelsize':9,
    'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':7.5,
    'lines.linewidth':1.4,'lines.markersize':4.5,'figure.dpi':150,
    'savefig.dpi':300,'savefig.bbox':'tight','axes.grid':True,
    'grid.alpha':0.3,'grid.linewidth':0.5,
})

# Paper constants — [0,255] range
C1 = (0.01 * 255)**2   # 6.5025
C2 = (0.03 * 255)**2   # 58.5225
PATCH = 16
ALPHA = 0.5

COLORS = {'ssim':'#1f77b4','ms_ssim':'#ff7f0e','fsim':'#2ca02c','iw_ssim':'#9467bd',
          'sub_ssim':'#d62728','dir_ssim':'#8c564b','combined':'#e377c2'}
MARKERS = {'ssim':'o','ms_ssim':'s','fsim':'^','iw_ssim':'D','sub_ssim':'v','dir_ssim':'<','combined':'*'}
METRIC_LABELS = {'ssim':'SSIM','ms_ssim':'MS-SSIM','fsim':'FSIM','iw_ssim':'IW-SSIM',
                 'sub_ssim':'Sub-SSIM (Prop.)','dir_ssim':'Dir-SSIM (Prop.)','combined':'Combined (Prop.)'}

# ── test images, [0,255] float ──
def get_images():
    imgs = {}
    for n in ['camera','moon','coins','clock']:
        imgs[n] = getattr(skdata, n)().astype(np.float64)
    imgs['astronaut'] = cv2.cvtColor(skdata.astronaut(), cv2.COLOR_RGB2GRAY).astype(np.float64)
    return imgs

# ── proposed metrics: EXACT paper formulation ──
def ssim_global(x, y):
    x = x.astype(np.float64); y = y.astype(np.float64)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    vxy = np.mean((x-mx)*(y-my))
    return ((2*mx*my+C1)*(2*vxy+C2)) / ((mx**2+my**2+C1)*(vx+vy+C2))

def sub_ssim(x, y, p=PATCH):
    H, W = x.shape; vals=[]
    for i in range(0, H-p+1, p):
        for j in range(0, W-p+1, p):
            vals.append(ssim_global(x[i:i+p,j:j+p], y[i:i+p,j:j+p]))
    return min(vals) if vals else ssim_global(x,y)   # worst patch (lower=worse)

def dir_ssim(x, y, alpha=ALPHA):
    x = x.astype(np.float64); y = y.astype(np.float64)
    # row-wise
    mxr, myr = x.mean(1), y.mean(1)
    vxr, vyr = x.var(1), y.var(1)
    vxyr = np.mean((x-mxr[:,None])*(y-myr[:,None]),axis=1)
    row = np.mean((2*mxr*myr+C1)*(2*vxyr+C2) / ((mxr**2+myr**2+C1)*(vxr+vyr+C2)))
    # col-wise
    mxc, myc = x.mean(0), y.mean(0)
    vxc, vyc = x.var(0), y.var(0)
    vxyc = np.mean((x-mxc[None,:])*(y-myc[None,:]),axis=0)
    col = np.mean((2*mxc*myc+C1)*(2*vxyc+C2) / ((mxc**2+myc**2+C1)*(vxc+vyc+C2)))
    return alpha*row + (1-alpha)*col

def combined_ssim(x, y, p=PATCH, alpha=ALPHA):
    H, W = x.shape; vals=[]
    for i in range(0, H-p+1, p):
        for j in range(0, W-p+1, p):
            vals.append(dir_ssim(x[i:i+p,j:j+p], y[i:i+p,j:j+p], alpha))
    return min(vals) if vals else dir_ssim(x,y,alpha)

# ── baselines via piq (need [0,1] tensors) ──
def t01(img):
    return torch.from_numpy(np.clip(img/255.0,0,1)).float().unsqueeze(0).unsqueeze(0)

def b_ssim(x,y): return float(piq.ssim(t01(x),t01(y),data_range=1.0))
def b_msssim(x,y):
    h,w=x.shape
    if h<161 or w<161:
        ph,pw=max(0,161-h),max(0,161-w)
        x=np.pad(x,((0,ph),(0,pw)),mode='reflect'); y=np.pad(y,((0,ph),(0,pw)),mode='reflect')
    return float(piq.multi_scale_ssim(t01(x),t01(y),data_range=1.0))
def b_fsim(x,y):
    xr=np.stack([x,x,x],-1)/255.0; yr=np.stack([y,y,y],-1)/255.0
    tx=torch.from_numpy(xr).float().permute(2,0,1).unsqueeze(0).clamp(0,1)
    ty=torch.from_numpy(yr).float().permute(2,0,1).unsqueeze(0).clamp(0,1)
    return float(piq.fsim(tx,ty,data_range=1.0))
def b_iwssim(x,y):
    h,w=x.shape
    if h<161 or w<161:
        ph,pw=max(0,161-h),max(0,161-w)
        x=np.pad(x,((0,ph),(0,pw)),mode='reflect'); y=np.pad(y,((0,ph),(0,pw)),mode='reflect')
    return float(piq.information_weighted_ssim(t01(x),t01(y),data_range=1.0))

def all_metrics(ref, dist):
    return {'ssim':b_ssim(ref,dist),'ms_ssim':b_msssim(ref,dist),'fsim':b_fsim(ref,dist),
            'iw_ssim':b_iwssim(ref,dist),'sub_ssim':sub_ssim(ref,dist),
            'dir_ssim':dir_ssim(ref,dist),'combined':combined_ssim(ref,dist)}

# ── degradations ([0,255]) ──
def awgn(img, s):  return np.clip(img + np.random.randn(*img.shape)*s, 0, 255)
def blur(img, k):  k=int(k)|1; return cv2.GaussianBlur(img.astype(np.float32),(k,k),k/3).astype(np.float64)
def jpeg(img, q):
    u8=img.clip(0,255).astype(np.uint8)
    _,e=cv2.imencode('.jpg',u8,[cv2.IMWRITE_JPEG_QUALITY,int(q)])
    return cv2.imdecode(e,cv2.IMREAD_GRAYSCALE).astype(np.float64)
def local_block(img, frac):
    o=img.copy(); H,W=img.shape; bh,bw=int(H*frac),int(W*frac)
    r,c=H//2-bh//2,W//2-bw//2; o[r:r+bh,c:c+bw]=255.0; return o

# ── PART A: sensitivity ──
def sensitivity(images):
    np.random.seed(42)
    exp = {
        'awgn':{'label':'AWGN ($\\sigma$)','lv':[5,12,25,38,51,64,76],'fn':awgn},
        'blur':{'label':'Gaussian Blur ($k$)','lv':[1,3,5,7,9,11,13],'fn':blur},
        'jpeg':{'label':'JPEG Quality','lv':[90,70,50,30,20,10,5],'fn':jpeg},
        'local':{'label':'Local Block (frac.)','lv':[0.05,0.08,0.10,0.12,0.15,0.18,0.20],'fn':local_block},
    }
    res={}
    for name,cfg in exp.items():
        print(f'  sensitivity {name}...',flush=True)
        curves={m:[] for m in METRIC_LABELS}
        for lv in cfg['lv']:
            acc={m:[] for m in METRIC_LABELS}
            for img in images.values():
                d=cfg['fn'](img,lv)
                for k,v in all_metrics(img,d).items(): acc[k].append(v)
            for k in curves: curves[k].append(float(np.mean(acc[k])))
        res[name]={'levels':cfg['lv'],'label':cfg['label'],'curves':curves}
    return res

# ── PART B: denoising ──
def gauss_dn(n,s): k=max(3,int(s/255*6*3)|1); return cv2.GaussianBlur(n.astype(np.float32),(k,k),1).astype(np.float64)
def med_dn(n,s):   return cv2.medianBlur(n.clip(0,255).astype(np.uint8),5).astype(np.float64)
def bil_dn(n,s):   return cv2.bilateralFilter(n.clip(0,255).astype(np.uint8),9,75,75).astype(np.float64)
def nlm_dn(n,s):
    n01=np.clip(n/255.0,0,1)
    sig=estimate_sigma(n01,channel_axis=None)
    out=denoise_nl_means(n01,h=1.15*sig,fast_mode=True,patch_size=5,patch_distance=6,channel_axis=None)
    return np.clip(out*255.0,0,255)

def denoising(images):
    np.random.seed(0)
    levels=[12.75, 25.5, 51.0]   # = 0.05,0.10,0.20 * 255
    dn={'Noisy':None,'Gauss.Filt':gauss_dn,'Median':med_dn,'Bilateral':bil_dn,'NLM':nlm_dn}
    res={}
    for s in levels:
        print(f'  denoising sigma={s:.2f}...',flush=True)
        mr={}
        for name,fn in dn.items():
            acc={m:[] for m in METRIC_LABELS}
            for img in images.values():
                noisy=awgn(img,s)
                d=noisy if fn is None else np.clip(fn(noisy,s),0,255)
                for k,v in all_metrics(img,d).items(): acc[k].append(v)
            mr[name]={k:float(np.mean(v)) for k,v in acc.items()}
        res[round(s,2)]=mr
    return res

# ── alpha ablation ──
def alpha_abl(images):
    np.random.seed(2)
    alphas=np.linspace(0,1,11); s=25.5; out=[]
    for a in alphas:
        vals=[dir_ssim(img,awgn(img,s),alpha=a) for img in images.values()]
        out.append(float(np.mean(vals)))
    return alphas.tolist(), out

# ════════════════════════ PLOTS ════════════════════════
def plot_sensitivity(res, path):
    names=list(res.keys())
    fig,axes=plt.subplots(2,2,figsize=(7.0,5.4)); axes=axes.flatten()
    for ax,nm in zip(axes,names):
        r=res[nm]; lv=r['levels']; cur=r['curves']
        xv = lv if nm!='jpeg' else list(reversed(lv))
        for mk,ml in METRIC_LABELS.items():
            y=cur[mk] if nm!='jpeg' else list(reversed(cur[mk]))
            prop=('Prop' in ml or 'Combined' in ml)
            ax.plot(xv,y,label=ml,color=COLORS[mk],marker=MARKERS[mk],
                    linestyle='-' if prop else '--',linewidth=1.8 if prop else 1.1)
        ax.set_xlabel(r['label']); ax.set_ylabel('Mean Metric Score')
        ax.set_title(f'({chr(97+names.index(nm))}) {nm.upper()}'); ax.set_ylim(-0.15,1.05)
    h,l=axes[0].get_legend_handles_labels()
    fig.legend(h,l,loc='lower center',ncol=4,bbox_to_anchor=(0.5,-0.05),frameon=True)
    plt.tight_layout(); plt.savefig(path,bbox_inches='tight'); plt.close(); print('  saved',path)

def plot_local_failure(images, path):
    np.random.seed(42); img=images['camera']
    bs=[0.03,0.06,0.09,0.12,0.15,0.18,0.21,0.25]
    sv=[b_ssim(img,local_block(img,b)) for b in bs]
    ms=[b_msssim(img,local_block(img,b)) for b in bs]
    su=[sub_ssim(img,local_block(img,b)) for b in bs]
    cv_=[combined_ssim(img,local_block(img,b)) for b in bs]
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,2.8))
    a1.plot(bs,sv,'--o',label='SSIM',color=COLORS['ssim'])
    a1.plot(bs,ms,'--s',label='MS-SSIM',color=COLORS['ms_ssim'])
    a1.plot(bs,su,'-v',label='Sub-SSIM (Prop.)',color=COLORS['sub_ssim'],linewidth=1.8)
    a1.plot(bs,cv_,'-*',label='Combined (Prop.)',color=COLORS['combined'],linewidth=1.8)
    a1.set_xlabel('Corrupted Block Size (fraction)'); a1.set_ylabel('Mean Metric Score')
    a1.set_title('(a) Local Block Corruption'); a1.legend(); a1.set_ylim(-0.15,1.05)
    ex=local_block(img,0.15); diff=np.clip(np.abs(img-ex)*3,0,255)
    a2.imshow(np.hstack([img,ex,diff]),cmap='gray',vmin=0,vmax=255)
    a2.set_title(f'(b) Ref | Corrupted (15%) | 3$\\times$Diff\nSSIM={b_ssim(img,ex):.3f}  '
                 f'Sub={sub_ssim(img,ex):.3f}  Comb={combined_ssim(img,ex):.3f}')
    a2.axis('off')
    plt.tight_layout(); plt.savefig(path,bbox_inches='tight'); plt.close(); print('  saved',path)

def plot_denoise_bars(res, path):
    s=25.5; r=res[round(s,2)]; methods=list(r.keys())
    ms_=['ssim','ms_ssim','fsim','sub_ssim','dir_ssim','combined']
    ml_=['SSIM','MS-SSIM','FSIM','Sub-SSIM\n(Prop.)','Dir-SSIM\n(Prop.)','Combined\n(Prop.)']
    x=np.arange(len(ms_)); w=0.15
    off=np.linspace(-(len(methods)-1)/2,(len(methods)-1)/2,len(methods))*w
    fig,ax=plt.subplots(figsize=(7.0,3.1))
    cols=['#aec7e8','#ffbb78','#98df8a','#ff9896','#c5b0d5']; hatch=['','//','\\\\','xx','..']
    for mth,o,c,h in zip(methods,off,cols,hatch):
        ax.bar(x+o,[r[mth][m] for m in ms_],w,label=mth,color=c,hatch=h,edgecolor='k',linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(ml_); ax.set_ylabel('Mean Score ($\\uparrow$)')
    ax.set_title('Denoising Methods Scored by Each Metric ($\\sigma=25.5$)')
    ax.set_ylim(-0.05,1.02); ax.legend(loc='lower right',ncol=3,fontsize=7)
    plt.tight_layout(); plt.savefig(path,bbox_inches='tight'); plt.close(); print('  saved',path)

def plot_alpha(alphas, vals, path):
    fig,ax=plt.subplots(figsize=(3.4,2.7))
    ax.plot(alphas,vals,'-o',color=COLORS['dir_ssim'])
    ax.axvline(0.5,color='red',linestyle='--',linewidth=1,label='$\\alpha=0.5$')
    ax.set_xlabel('$\\alpha$ (row weight)'); ax.set_ylabel('Mean Dir-SSIM')
    ax.set_title('Effect of $\\alpha$ ($\\sigma=25.5$)'); ax.legend()
    plt.tight_layout(); plt.savefig(path,bbox_inches='tight'); plt.close(); print('  saved',path)

# ════════════════════════ MAIN ════════════════════════
if __name__=='__main__':
    print('constants: C1=%.4f C2=%.4f patch=%d alpha=%.1f'%(C1,C2,PATCH,ALPHA))
    images=get_images()
    print('images:',list(images.keys()))

    print('[A] sensitivity'); sens=sensitivity(images)
    plot_sensitivity(sens,'/home/claude/figures/fig_sensitivity.pdf')
    plot_local_failure(images,'/home/claude/figures/fig_local_block.pdf')

    print('[B] denoising'); dn=denoising(images)
    plot_denoise_bars(dn,'/home/claude/figures/fig_denoise_bars.pdf')

    print('[C] alpha'); al,av=alpha_abl(images)
    plot_alpha(al,av,'/home/claude/figures/fig_alpha.pdf')

    # dump all numbers to JSON for the writeup
    out={'constants':{'C1':C1,'C2':C2,'patch':PATCH,'alpha':ALPHA},
         'sensitivity':sens,'denoising':dn,'alpha':{'alphas':al,'vals':av}}
    with open('/home/claude/results.json','w') as f: json.dump(out,f,indent=2)
    print('\nsaved results.json')

    # print key tables
    print('\n=== SENSITIVITY (representative levels) ===')
    rep={'awgn':25,'blur':7,'jpeg':30,'local':0.15}
    ms_=['ssim','ms_ssim','fsim','iw_ssim','sub_ssim','dir_ssim','combined']
    print(f"{'dist':<14}"+''.join(f'{m:>9}' for m in ms_))
    for nm,lv in rep.items():
        idx=sens[nm]['levels'].index(lv)
        print(f"{nm:<14}"+''.join(f"{sens[nm]['curves'][m][idx]:>9.4f}" for m in ms_))

    print('\n=== DENOISING ===')
    for s,r in dn.items():
        print(f'\n-- sigma={s} --')
        print(f"{'method':<12}"+''.join(f'{m:>9}' for m in ms_))
        for mth,sc in r.items():
            print(f"{mth:<12}"+''.join(f'{sc[m]:>9.4f}' for m in ms_))

    print('\n=== LOCAL BLOCK headline (camera, 15%) ===')
    img=images['camera']; ex=local_block(img,0.15)
    print(f'SSIM={b_ssim(img,ex):.4f}  MS-SSIM={b_msssim(img,ex):.4f}  '
          f'Sub-SSIM={sub_ssim(img,ex):.4f}  Combined={combined_ssim(img,ex):.4f}')
