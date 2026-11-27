"""(1) Macro ROC-AUC/AP of the strict out-of-fold ML scores under the ORIGINAL (B03) and URBAN-AWARE (B03U) labels.
(2) ROC-AUC/AP of the final full-data ML maps and of the MCDA surfaces scored on all valid mask cells for each event
    (in-sample for ML: the final maps are fitted on all four events), and MCDA on original versus urban-aware labels.
Usage: python label_version_and_insample_check.py --b07 B07_npz --b07u B07U_npz --masks DIR_with_B03_and_B03U_tifs --maps DIR_B09A --mcda DIR_DU --mask MASKTIF --out PREFIX"""
import argparse, numpy as np, pandas as pd, rasterio
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
ap = argparse.ArgumentParser()
for a in ('b07', 'b07u', 'masks', 'maps', 'mcda', 'mask', 'out'): ap.add_argument('--' + a, required=True)
g = ap.parse_args(); Y = ('2019', '2021', '2023', '2026')
EV = {'2019': 'E03_2019_Jul', '2021': 'E04_2021_Jul', '2023': 'E05_2023_Aug', '2026': 'E07_2026_Jul'}
rows = []
z = np.load(g.b07)
for y in Y:
    o, rf, xg = z['Flood_' + y], z['RF_' + y], z['XGB_' + y]; m = np.isfinite(rf) & np.isfinite(xg) & (o >= 0)
    for n, s in (('RF', rf), ('XGB', xg), ('Consensus', (rf + xg) / 2)): rows.append(('strict_OOF', 'original_B03', y, n, AUC(o[m] == 1, s[m]), AP(o[m] == 1, s[m])))
z = np.load(g.b07u)
for y in Y:
    o, rf, xg = z['Observed_' + y], z['RF_' + y], z['XGB_' + y]; m = np.isfinite(rf) & np.isfinite(xg) & (o >= 0)
    for n, s in (('RF', rf), ('XGB', xg), ('Consensus', (rf + xg) / 2)): rows.append(('strict_OOF', 'urban_aware_B03U', y, n, AUC(o[m] == 1, s[m]), AP(o[m] == 1, s[m])))
mask = rasterio.open(g.mask).read(1) == 1
S = {n: rasterio.open(p).read(1).astype(np.float32) for n, p in (
    ('RF', f'{g.maps}/BUET_B09A_RF_SusceptibilityScore_30m.tif'), ('XGB', f'{g.maps}/BUET_B09A_XGB_SusceptibilityScore_30m.tif'),
    ('Consensus', f'{g.maps}/BUET_B09A_RF_XGB_ConsensusScore_30m.tif'), ('AHP', f'{g.mcda}/DU10_AHP_Susceptibility_Index_01.tif'),
    ('FAHP', f'{g.mcda}/DU11_FAHP_Susceptibility_Index_01.tif'), ('ANP', f'{g.mcda}/DU12_ANP_Susceptibility_Index_01.tif'))}
for y in Y:
    for pre, lab in (('B03', 'original_B03'), ('B03U', 'urban_aware_B03U')):
        l = rasterio.open(f'{g.masks}/BUET_{pre}_{EV[y]}_FloodMask30m.tif').read(1); v = (l != 255) & mask
        for n, s in S.items(): rows.append(('full_valid_cells', lab, y, n, AUC(l[v] == 1, s[v]), AP(l[v] == 1, s[v])))
d = pd.DataFrame(rows, columns=['Evaluation', 'Label_version', 'Event', 'Method', 'ROC_AUC', 'AP']); d.to_csv(g.out + '_by_event.csv', index=False)
mac = d.groupby(['Evaluation', 'Label_version', 'Method'])[['ROC_AUC', 'AP']].mean().reset_index(); mac.to_csv(g.out + '_macro.csv', index=False)
print(mac.round(4).to_string())
