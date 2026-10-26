# -*- coding: utf-8 -*-
"""LKH-3 ayar pilotu (alti ayar, iki bahce, k=8, 120 s). Cikti: out/lkh_tuning_pilot.txt"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import instances as I
from mtsp import lkh3, mm
from concurrent.futures import ProcessPoolExecutor
CF={'A_mt3':{'MOVE_TYPE':3},
    'B_init_mt3':{'INITIAL_TOUR_ALGORITHM':'MTSP','MOVE_TYPE':3},
    'C_init_mt2':{'INITIAL_TOUR_ALGORITHM':'MTSP','MOVE_TYPE':2},
    'D_init_mt3_pop':{'INITIAL_TOUR_ALGORITHM':'MTSP','MOVE_TYPE':3,'CANDIDATE_SET_TYPE':'POPMUSIC'},
    'E_init_mt5':{'INITIAL_TOUR_ALGORITHM':'MTSP'},
    'F_init_mt3_nosub':{'INITIAL_TOUR_ALGORITHM':'MTSP','MOVE_TYPE':3,'SUBSEQUENT_MOVE_TYPE':2}}
def job(a):
    cname, rows, cols, k = a
    o=I.orchard(rows,cols)
    r=lkh3.mtsp_minmax(o['X'],o['Y'],o['dx'],o['dy'],k,time_limit=120,extra=CF[cname])
    mk=mm.makespan(r['routes'],o['X'],o['Y'],o['dx'],o['dy']) if r['routes'] else None
    return cname, rows, k, [(round(t,1),round(c)) for t,c in r['trace']][:6], mk
jobs=[(c,40,50,8) for c in CF]+[(c,20,25,8) for c in CF]
if __name__ == '__main__':
  with ProcessPoolExecutor(12) as ex:
    for res in ex.map(job, jobs): print(res, flush=True)
