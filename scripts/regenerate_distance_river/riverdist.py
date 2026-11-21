"""Shared helpers: minimal GeoPackage line reader and exact point-to-segment distance (stdlib + numpy only).
Coordinates are in EPSG:32646 (WGS 84 / UTM zone 46N), metres."""
import sqlite3, struct, numpy as np
def read_segments(gpkg, table=None):
    c = sqlite3.connect(gpkg)
    if table is None: table = c.execute("select table_name from gpkg_contents where data_type='features'").fetchone()[0]
    gcol = c.execute("select column_name from gpkg_geometry_columns where table_name=?", (table,)).fetchone()[0]
    cols = [r[1] for r in c.execute(f'pragma table_info("{table}")')]
    segs, attrs = [], []
    for row in c.execute(f'select * from "{table}"'):
        d = dict(zip(cols, row)); b = d[gcol]
        flags = b[3]; env = (flags >> 1) & 7
        off = 8 + {0:0,1:32,2:48,3:48,4:64}[env]
        bo = '<' if b[off] == 1 else '>'; gt = struct.unpack(bo+'I', b[off+1:off+5])[0] % 1000
        has_z = (struct.unpack(bo+'I', b[off+1:off+5])[0] // 1000) in (1, 3); dim = 3 if has_z else 2
        p = off + 5; lines = []
        def line(p):
            n = struct.unpack(bo+'I', b[p:p+4])[0]; p += 4
            a = np.frombuffer(b[p:p+8*dim*n], dtype=bo+'f8').reshape(n, dim)[:, :2]; return a.copy(), p + 8*dim*n
        if gt == 2: a, p = line(p); lines.append(a)
        elif gt == 5:
            m = struct.unpack(bo+'I', b[p:p+4])[0]; p += 4
            for _ in range(m):
                p += 5; a, p = line(p); lines.append(a)
        else: raise ValueError(gt)
        for a in lines:
            segs.append(np.hstack([a[:-1], a[1:]])); attrs += [d] * (len(a) - 1)
    return np.vstack(segs), attrs, cols
def point_seg_dist(px, py, S, chunk=20000):
    """Exact Euclidean distance from points to nearest line segment (vectorised)."""
    out = np.empty(len(px)); x1, y1, x2, y2 = S.T; dx, dy = x2-x1, y2-y1; L2 = np.maximum(dx*dx+dy*dy, 1e-12)
    for i in range(0, len(px), chunk):
        X = px[i:i+chunk, None]; Y = py[i:i+chunk, None]
        t = np.clip(((X-x1)*dx + (Y-y1)*dy)/L2, 0, 1)
        out[i:i+chunk] = np.sqrt(((X-(x1+t*dx))**2 + (Y-(y1+t*dy))**2).min(1))
    return out

def load_segments(path):
    """Return an (n,4) array x1,y1,x2,y2 from a .gpkg (line layer) or a .npz written by 1_prepare_hydrorivers_utm46n.py."""
    if path.lower().endswith(".npz"):
        return np.load(path)["segments"]
    return read_segments(path)[0]
