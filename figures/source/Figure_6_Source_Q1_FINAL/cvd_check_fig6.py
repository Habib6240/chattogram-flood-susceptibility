"""Colour-vision / greyscale QA for Figure 6 (Machado et al. 2009, severity 1.0). usage: python3 -I cvd_check_fig6.py FINAL.png OUTDIR"""
import sys, os, numpy as np
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
Image.MAX_IMAGE_PIXELS = None
png, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
M = {'deuteranopia': np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]), 'protanopia': np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]])}
lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
enc = lambda l: (np.where(np.clip(l, 0, 1) <= 0.0031308, 12.92 * np.clip(l, 0, 1), 1.055 * np.clip(l, 0, 1) ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)
im = Image.open(png).convert('RGB'); im = im.resize((im.width // 3, im.height // 3), Image.LANCZOS); a = np.asarray(im) / 255.0; L = lin(a)
for k, m in M.items(): Image.fromarray(enc(L @ m.T)).save(os.path.join(out, f'cvd_{k}.png'))
y = 0.2126 * L[..., 0] + 0.7152 * L[..., 1] + 0.0722 * L[..., 2]; Image.fromarray(enc(np.stack([y] * 3, -1))).save(os.path.join(out, 'cvd_greyscale.png'))
def sim(rgb, mode):
    l = lin(np.array(rgb[:3]))
    if mode in M: l = np.clip(M[mode] @ l, 0, 1)
    if mode == 'greyscale': l = np.array([0.2126 * l[0] + 0.7152 * l[1] + 0.0722 * l[2]] * 3)
    return l
def Lab(l):
    X = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ l / np.array([0.9505, 1, 1.089]); f = np.where(X > 0.008856, np.cbrt(X), 7.787 * X + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])
rd = plt.get_cmap('RdBu_r'); cols = [rd(v) for v in np.linspace(0.08, 0.92, 9)]; cols[4] = (0.93, 0.93, 0.93, 1.0)
ramps = {'(a) YlOrRd': plt.get_cmap('YlOrRd'), '(b) YlGnBu': plt.get_cmap('YlGnBu')}; lines = []
for mode in ('normal', 'deuteranopia', 'protanopia', 'greyscale'):
    for nm, cmap in ramps.items():
        Ls = np.array([Lab(sim(cmap(x), mode))[0] for x in np.linspace(0, 1, 101)]); lines.append(f'{mode} | {nm}: L* {Ls[0]:.0f} -> {Ls[-1]:.0f}, monotonic decrease = {bool(np.all(np.diff(Ls) <= 1e-9))}')
    S = [Lab(sim(c, mode)) for c in cols]; dE = [float(np.linalg.norm(S[i] - S[i + 1])) for i in range(8)]
    lines.append(f'{mode} | (c) diverging 9 classes: L* ' + ', '.join(f'{s[0]:.0f}' for s in S) + f'; min adjacent dE76 = {min(dE):.1f}; zero-class vs +-1 dE76 = {min(float(np.linalg.norm(S[4] - S[3])), float(np.linalg.norm(S[4] - S[5]))):.1f}')
open(os.path.join(out, 'Figure_6_ColourVision_QA.txt'), 'w').write('\n'.join(lines)); print('\n'.join(lines))
