import struct, math, zipfile, sqlite3, numpy as np
def read_shp(shp_path):
    b=open(shp_path,'rb').read(); pos=100; recs=[]
    while pos<len(b):
        cl=struct.unpack('>i',b[pos+4:pos+8])[0]*2; c0=pos+8; pos=c0+cl
        st=struct.unpack('<i',b[c0:c0+4])[0]
        if st==0: recs.append([]); continue
        npart,npts=struct.unpack('<2i',b[c0+36:c0+44]); parts=list(struct.unpack('<%di'%npart,b[c0+44:c0+44+4*npart]))+[npts]; p0=c0+44+4*npart
        pts=np.frombuffer(b[p0:p0+16*npts],dtype='<f8').reshape(-1,2)
        recs.append([pts[parts[i]:parts[i+1]] for i in range(npart)])
    return recs
def read_dbf(path):
    b=open(path,'rb').read(); n=struct.unpack('<I',b[4:8])[0]; hl,rl=struct.unpack('<2H',b[8:12]); fields=[]; p=32
    while b[p]!=0x0D:
        name=b[p:p+11].split(b'\x00')[0].decode('latin1'); ln=b[p+16]; fields.append((name,ln)); p+=32
    rows=[]
    for i in range(n):
        r=b[hl+i*rl+1:hl+(i+1)*rl]; off=0; d={}
        for name,ln in fields: d[name]=r[off:off+ln].decode('utf-8','ignore').strip(); off+=ln
        rows.append(d)
    return rows
_a=6378137.0; _f=1/298.257223563; _k0=0.9996; _lon0=math.radians(93.0); _n=_f/(2-_f); _A=_a/(1+_n)*(1+_n**2/4+_n**4/64)
_al=[_n/2-2*_n**2/3+5*_n**3/16+41*_n**4/180,13*_n**2/48-3*_n**3/5+557*_n**4/1440,61*_n**3/240-103*_n**4/140,49561*_n**4/161280]
_be=[_n/2-2*_n**2/3+37*_n**3/96-_n**4/360,_n**2/48+_n**3/15-437*_n**4/1440,17*_n**3/480-37*_n**4/840,4397*_n**4/161280]
_de=[2*_n-2*_n**2/3-2*_n**3+116*_n**4/45,7*_n**2/3-8*_n**3/5-227*_n**4/45,56*_n**3/15-136*_n**4/35,4279*_n**4/630]
def utm46(lon,lat):
    ph=np.radians(lat); la=np.radians(lon)-_lon0; c=2*math.sqrt(_n)/(1+_n)
    t=np.sinh(np.arctanh(np.sin(ph))-c*np.arctanh(c*np.sin(ph))); xi=np.arctan2(t,np.cos(la)); eta=np.arctanh(np.sin(la)/np.sqrt(1+t**2))
    xs=xi+sum(_al[j]*np.sin(2*(j+1)*xi)*np.cosh(2*(j+1)*eta) for j in range(4)); es=eta+sum(_al[j]*np.cos(2*(j+1)*xi)*np.sinh(2*(j+1)*eta) for j in range(4))
    return 500000+_k0*_A*es,_k0*_A*xs
def utm46_inv(E,N):
    xi=N/(_k0*_A); eta=(E-500000)/(_k0*_A)
    xp=xi-sum(_be[j]*np.sin(2*(j+1)*xi)*np.cosh(2*(j+1)*eta) for j in range(4)); ep=eta-sum(_be[j]*np.cos(2*(j+1)*xi)*np.sinh(2*(j+1)*eta) for j in range(4))
    chi=np.arcsin(np.sin(xp)/np.cosh(ep)); lat=chi+sum(_de[j]*np.sin(2*(j+1)*chi) for j in range(4)); lon=_lon0+np.arctan2(np.sinh(ep),np.cos(xp))
    return np.degrees(lon),np.degrees(lat)
def read_gpkg_lines(path):
    c=sqlite3.connect(path); out=[]
    for fid,blob,ordc,up in c.execute("select fid,geom,ORD_STRA,UPLAND_SKM from '05_River_HydroRIVERS_UPLAND25_UTM46N'"):
        b=bytes(blob); assert b[:2]==b'GP'; fl=b[3]; env={0:0,1:32,2:48,3:48,4:64}[(fl>>1)&7]; w=b[8+env:]
        bo='<' if w[0]==1 else '>'; typ=struct.unpack(bo+'I',w[1:5])[0]; assert typ==5,typ
        ng=struct.unpack(bo+'I',w[5:9])[0]; p=9; lines=[]
        for _ in range(ng):
            bo2='<' if w[p]==1 else '>'; p+=5; npt=struct.unpack(bo2+'I',w[p:p+4])[0]; p+=4
            lines.append(np.frombuffer(w[p:p+16*npt],dtype=bo2+'f8').reshape(-1,2)); p+=16*npt
        out.append((lines,int(ordc),float(up)))
    return out
