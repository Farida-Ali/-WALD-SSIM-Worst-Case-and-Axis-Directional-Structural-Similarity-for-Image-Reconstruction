"""
Directional validation + component ablation for WALD-SSIM.
Uses the paper's exact metric formulation on the same 5 test images.
Produces:
  (1) directional table: SSIM_row vs SSIM_col under H/V motion blur
  (2) ablation table: response (1-score) per variant per artifact type
  (3) a directional figure
"""
import numpy as np, cv2, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage import data as skd

C1=(0.01*255)**2; C2=(0.03*255)**2
PATCH=16; ALPHA=0.5

plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':9,
    'savefig.dpi':300,'savefig.bbox':'tight','axes.grid':True,'grid.alpha':0.3})

# ---------- metric components (paper-exact) ----------
def ssim_global(x,y):
    x=x.astype(np.float64); y=y.astype(np.float64)
    mx,my=x.mean(),y.mean(); vx,vy=x.var(),y.var(); vxy=np.mean((x-mx)*(y-my))
    return ((2*mx*my+C1)*(2*vxy+C2))/((mx**2+my**2+C1)*(vx+vy+C2))

def ssim_row(x,y):
    x=x.astype(np.float64); y=y.astype(np.float64)
    mxr,myr=x.mean(1),y.mean(1); vxr,vyr=x.var(1),y.var(1)
    vxyr=np.mean((x-mxr[:,None])*(y-myr[:,None]),axis=1)
    return np.mean((2*mxr*myr+C1)*(2*vxyr+C2)/((mxr**2+myr**2+C1)*(vxr+vyr+C2)))

def ssim_col(x,y):
    x=x.astype(np.float64); y=y.astype(np.float64)
    mxc,myc=x.mean(0),y.mean(0); vxc,vyc=x.var(0),y.var(0)
    vxyc=np.mean((x-mxc[None,:])*(y-myc[None,:]),axis=0)
    return np.mean((2*mxc*myc+C1)*(2*vxyc+C2)/((mxc**2+myc**2+C1)*(vxc+vyc+C2)))

def dir_ssim(x,y,alpha=ALPHA):
    return alpha*ssim_row(x,y)+(1-alpha)*ssim_col(x,y)

def sub_ssim(x,y,p=PATCH):
    H,W=x.shape; v=[]
    for i in range(0,H-p+1,p):
        for j in range(0,W-p+1,p): v.append(ssim_global(x[i:i+p,j:j+p],y[i:i+p,j:j+p]))
    return min(v) if v else ssim_global(x,y)

def combined(x,y,p=PATCH,alpha=ALPHA):
    H,W=x.shape; v=[]
    for i in range(0,H-p+1,p):
        for j in range(0,W-p+1,p): v.append(dir_ssim(x[i:i+p,j:j+p],y[i:i+p,j:j+p],alpha))
    return min(v) if v else dir_ssim(x,y,alpha)

def ms_combined(x,y,scales=3,p=PATCH,alpha=ALPHA):
    vals=[]; xs,ys=x.copy(),y.copy()
    for s in range(scales):
        if xs.shape[0]<p or xs.shape[1]<p: break
        vals.append(combined(xs,ys,p,alpha))
        xs=cv2.resize(xs,(xs.shape[1]//2,xs.shape[0]//2))
        ys=cv2.resize(ys,(ys.shape[1]//2,ys.shape[0]//2))
    return min(vals)   # worst across scales (worst-case MS)

# ---------- images ----------
def get_images():
    d={}
    for n in ['camera','moon','coins','clock']:
        d[n]=getattr(skd,n)().astype(np.float64)
    d['astronaut']=cv2.cvtColor(skd.astronaut(),cv2.COLOR_RGB2GRAY).astype(np.float64)
    # crop to multiple of PATCH
    for k in d: 
        H,W=d[k].shape; d[k]=d[k][:(H//PATCH)*PATCH,:(W//PATCH)*PATCH]
    return d

# ---------- directional artifacts ----------
def motion_blur(img,k,horizontal=True):
    ker=np.zeros((k,k))
    if horizontal: ker[k//2,:]=1.0/k
    else:          ker[:,k//2]=1.0/k
    return cv2.filter2D(img.astype(np.float32),-1,ker).astype(np.float64)

def local_block(img,frac=0.15):
    o=img.copy(); H,W=img.shape; bh,bw=int(H*frac),int(W*frac)
    r,c=H//2-bh//2,W//2-bw//2; o[r:r+bh,c:c+bw]=255.0; return o

def multiscale_block(img):
    """small + large block (different scales) in one image."""
    o=img.copy(); H,W=img.shape
    for frac,(ry,rx) in [(0.06,(0.2,0.2)),(0.20,(0.55,0.55))]:
        bh,bw=int(H*frac),int(W*frac); r,c=int(H*ry),int(W*rx)
        o[r:r+bh,c:c+bw]=255.0
    return o

# ====================================================================
# EXPERIMENT 1 — directional validation
# ====================================================================
def directional_validation(images):
    ks=[5,9,13]
    rows=[]
    for k in ks:
        for direction in ['H','V']:
            sr,sc,sg=[],[],[]
            for img in images.values():
                dist=motion_blur(img,k,horizontal=(direction=='H'))
                sr.append(ssim_row(img,dist)); sc.append(ssim_col(img,dist)); sg.append(ssim_global(img,dist))
            sr,sc,sg=np.mean(sr),np.mean(sc),np.mean(sg)
            rows.append({'k':k,'dir':direction,'global':sg,'row':sr,'col':sc,'gap':sr-sc})
    return rows

# ====================================================================
# EXPERIMENT 2 — component ablation (response = 1 - score)
# ====================================================================
def ablation(images):
    artifacts={
        'local_block': lambda im: local_block(im,0.15),
        'h_blur':      lambda im: motion_blur(im,11,True),
        'v_blur':      lambda im: motion_blur(im,11,False),
        'ms_block':    lambda im: multiscale_block(im),
    }
    variants={
        'SSIM':     lambda x,y: ssim_global(x,y),
        'Sub-SSIM': lambda x,y: sub_ssim(x,y),
        'Dir-SSIM': lambda x,y: dir_ssim(x,y),
        'Combined': lambda x,y: combined(x,y),
        'MS-Comb':  lambda x,y: ms_combined(x,y),
    }
    table={}
    for vname,vf in variants.items():
        table[vname]={}
        for aname,af in artifacts.items():
            resp=[]
            for img in images.values():
                dist=af(img); resp.append(1.0-vf(img,dist))
            table[vname][aname]=float(np.mean(resp))
    return table

# ====================================================================
# FIGURE — directional gap
# ====================================================================
def plot_directional(rows,path):
    ks=sorted(set(r['k'] for r in rows))
    h_gap=[next(r['gap'] for r in rows if r['k']==k and r['dir']=='H') for k in ks]
    v_gap=[next(r['gap'] for r in rows if r['k']==k and r['dir']=='V') for k in ks]
    h_row=[next(r['row'] for r in rows if r['k']==k and r['dir']=='H') for k in ks]
    h_col=[next(r['col'] for r in rows if r['k']==k and r['dir']=='H') for k in ks]
    v_row=[next(r['row'] for r in rows if r['k']==k and r['dir']=='V') for k in ks]
    v_col=[next(r['col'] for r in rows if r['k']==k and r['dir']=='V') for k in ks]

    fig,(a1,a2)=plt.subplots(1,2,figsize=(7.0,2.8))
    # left: row vs col under H and V
    a1.plot(ks,h_row,'-o',color='#1f77b4',label='row | H-blur')
    a1.plot(ks,h_col,'--o',color='#1f77b4',alpha=0.6,label='col | H-blur')
    a1.plot(ks,v_row,'-s',color='#d62728',label='row | V-blur')
    a1.plot(ks,v_col,'--s',color='#d62728',alpha=0.6,label='col | V-blur')
    a1.set_xlabel('motion-blur kernel size $k$'); a1.set_ylabel('SSIM score')
    a1.set_title('(a) Row vs. column response'); a1.legend(fontsize=6.5)
    # right: direction gap
    a2.plot(ks,h_gap,'-o',color='#1f77b4',label='Horizontal blur')
    a2.plot(ks,v_gap,'-s',color='#d62728',label='Vertical blur')
    a2.axhline(0,color='k',lw=0.6)
    a2.set_xlabel('motion-blur kernel size $k$'); a2.set_ylabel(r'gap $=$ SSIM$_{row}-$SSIM$_{col}$')
    a2.set_title('(b) Direction gap reverses sign'); a2.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(path); plt.close(); print('saved',path)

# ====================================================================
if __name__=='__main__':
    imgs=get_images()
    print("Images:",list(imgs.keys()))

    print("\n=== EXPERIMENT 1: Directional validation ===")
    rows=directional_validation(imgs)
    print(f"{'k':>3} {'dir':>4} {'global':>8} {'row':>8} {'col':>8} {'gap(r-c)':>9}")
    for r in rows:
        print(f"{r['k']:>3} {r['dir']:>4} {r['global']:>8.4f} {r['row']:>8.4f} {r['col']:>8.4f} {r['gap']:>9.4f}")
    plot_directional(rows,'/home/claude/figs/fig_directional.pdf')

    print("\n=== EXPERIMENT 2: Component ablation (response = 1 - score, higher=more detected) ===")
    tab=ablation(imgs)
    arts=['local_block','h_blur','v_blur','ms_block']
    print(f"{'variant':<10}"+''.join(f'{a:>13}' for a in arts))
    for v,d in tab.items():
        print(f"{v:<10}"+''.join(f'{d[a]:>13.4f}' for a in arts))

    json.dump({'directional':rows,'ablation':tab},
              open('/home/claude/dir_ablation_results.json','w'),indent=2)
    print("\nsaved dir_ablation_results.json")
