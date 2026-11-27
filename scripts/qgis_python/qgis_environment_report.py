# Run in QGIS: Plugins > Python Console > Show Editor > open this file > Run.
# Read-only: prints and saves a software-version / processing-history report. Changes nothing in your project.
import sys, os, platform, json, sqlite3, shutil, datetime
from qgis.core import Qgis, QgsApplication, QgsProject
out = []
def add(k, v): out.append((k, str(v)))
add('report_time', datetime.datetime.now().isoformat(timespec='seconds'))
add('QGIS_version', Qgis.QGIS_VERSION)
add('QGIS_release_name', Qgis.QGIS_RELEASE_NAME)
add('Python', sys.version.replace('\n', ' '))
add('OS', platform.platform())
try:
    from osgeo import gdal, osr
    add('GDAL', gdal.__version__)
    add('PROJ', '%d.%d.%d' % osr.GetPROJVersion())
except Exception as e: add('GDAL/PROJ', 'unavailable: %s' % e)
try:
    from qgis.PyQt.QtCore import QT_VERSION_STR; add('Qt', QT_VERSION_STR)
except Exception: pass
for mod in ('numpy', 'scipy', 'pandas', 'sklearn', 'xgboost', 'rasterio', 'shap'):
    try:
        m = __import__(mod); add(mod, getattr(m, '__version__', 'installed (no version attr)'))
    except Exception: add(mod, 'not installed in QGIS Python')
# GRASS
gv = None
for modname in ('grassprovider.grass_utils', 'grassprovider.Grass7Utils', 'processing.algs.grass7.Grass7Utils'):
    try:
        m = __import__(modname, fromlist=['x'])
        cls = getattr(m, 'Grass7Utils', None) or getattr(m, 'GrassUtils', None)
        gv = cls.installedVersion(True) if cls else None
        add('GRASS_path', cls.grassPath() if cls else '')
        if gv: break
    except Exception as e: add('GRASS_probe_' + modname, 'failed: %s' % e)
add('GRASS_version', gv or 'not detected by probe (see Plugins > Processing > Settings > Providers > GRASS)')
# processing providers
reg = QgsApplication.processingRegistry()
add('processing_providers', ', '.join('%s %s' % (p.id(), p.longName()) for p in reg.providers()))
for aid in ('grass7:r.slope.aspect', 'grass:r.slope.aspect', 'grass7:r.watershed', 'grass:r.watershed',
            'grass7:r.stream.extract', 'grass:r.stream.extract', 'native:alignrasters', 'native:linedensity',
            'gdal:proximity', 'gdal:rastercalculator', 'native:rastercalc'):
    add('algorithm_available ' + aid, bool(reg.algorithmById(aid)))
add('project_file', QgsProject.instance().fileName())
# processing history (QGIS keeps it in history.db of the active profile)
prof = QgsApplication.qgisSettingsDirPath()
db = os.path.join(prof, 'processing', 'processing_history.db')
cands = [os.path.join(prof, 'history.db'), db]
base = os.path.join(os.path.expanduser('~'), 'Desktop')
if not os.path.isdir(base): base = os.path.expanduser('~')
txt = os.path.join(base, 'QGIS_environment_report.txt')
hist = os.path.join(base, 'QGIS_processing_history_export.txt')
found = [c for c in cands if os.path.exists(c)]
add('history_db_candidates_found', found or 'none in profile ' + prof)
n = 0
with open(hist, 'w', encoding='utf-8') as fh:
    for c in found:
        try:
            con = sqlite3.connect('file:%s?mode=ro' % c, uri=True)
            tabs = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
            fh.write('### %s tables=%s\n' % (c, tabs))
            for t in tabs:
                cols = [r[1] for r in con.execute('pragma table_info(%s)' % t)]
                fh.write('## table %s columns=%s\n' % (t, cols))
                for row in con.execute('select * from %s' % t):
                    fh.write(repr(row)[:4000] + '\n'); n += 1
            con.close()
        except Exception as e: fh.write('could not read %s: %s\n' % (c, e))
add('history_rows_exported', n)
with open(txt, 'w', encoding='utf-8') as f:
    for k, v in out: f.write('%s: %s\n' % (k, v))
for k, v in out: print('%s: %s' % (k, v))
print('\nSaved:', txt, '\nSaved:', hist, '\nSend me both files.')
