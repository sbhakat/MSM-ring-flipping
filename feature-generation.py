#msmbuilder imports 
from msmbuilder.featurizer import DihedralFeaturizer
from msmbuilder.utils import verbosedump,verboseload
from msmbuilder.utils import load,dump

#other imports
import os,glob,shutil
import numpy as np
import mdtraj as md
import pandas as pd 
import pickle
#prettier plots

#Loading the trajectory
ref = md.load('prot.pdb')
# Zero indexed. This selectes Tyr73 from PDB
a = ref.top.select("resid 72")

# Uploading PBC fixed XTC file
ds = dataset("*.xtc", topology="prot.pdb", atom_indices=a, stride=20)


#Featurization, sin/cos transformed which uses Chi1 and Chi2 angle of Tyr73
featurizer = DihedralFeaturizer(types=['chi1', 'chi2'])
dump(featurizer,"transformed_raw_featurizer.pkl")

# Sin/cos untransformed featurizer
f=DihedralFeaturizer(types=['chi1', 'chi2'], sincos=False)
dump(f,"raw_featurizer.pkl")

# Fitting the featurizer to the .xtc file
diheds = featurizer.fit_transform(ds)
dump(diheds, "features.pkl")
                            
dihedsraw = f.fit_transform(ds)
dump(dihedsraw, "raw-features.pkl")     

print(diheds[0].shape)


#Robust scaling, scaling featurization
from msmbuilder.preprocessing import RobustScaler
scaler = RobustScaler()
scaled_diheds = scaler.fit_transform(diheds)

dump(scaled_diheds, "scaled-transformed-features.pkl")
