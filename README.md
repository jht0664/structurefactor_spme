# StructureFactor_SPME
This program is to calculate structure factor using atomic coordinates and smooth particle mesh ewald summation (SPME) for better resolution. In this code, I focused on building faster algorithm using Numpy python and utilizing MDTraj I/O interface (http://mdtraj.org) for supporting a number of trajectory files.

## prerequisites in python
I **strongly** recommend to install numpy, linking MKL library (or similar Linear Algebra libraries) and OpenMP library to get fastest performance.
(See details: https://docs.scipy.org/doc/numpy-1.15.1/user/building.html)
Note that in my case I installed numpy package in miniconda (python 3.x) exporting MKL library path.
* numpy (or numpy-mkl): for numerical calculations
* mdtraj (> version 1.9.1): for reading trajectory and structure file

## run program and parameters
Before running this program, you may need to prepare for structure file (.gro or .pdb file). In particular, if you choose to utilize Cromer-mann coefficients for atomic form factor, you should change the name of atomtypes to atom symbol in the structure file. For example, I changed atomtypes from examples/conf.gro
```
    1Cho     C3   14   6.126  -2.495   2.991
    1Cho     H9   15   6.077  -2.570   3.058
    1Cho    H10   16   6.083  -2.399   3.020
    1Cho     C4   17   6.086  -2.530   2.842
```
into examples/cation.gro
```
    1Cho      C   14   6.126  -2.495   2.991
    1Cho      H   15   6.077  -2.570   3.058
    1Cho      H   16   6.083  -2.399   3.020
    1Cho      C   17   6.086  -2.530   2.842
```
If you choose to read atomic form factor files (like "AFF_{atomtype}.out"), you need to make the files which have corresponding atom type names in structure file. (See details in examples/AFF_{}.out files) Note that the first column should be wave vector, q, in unit of angstrom^(-1).

If you want to remove some atoms to get partial structure factor, change atomtype to "X" or "D". (examples/cation.gro)

Once you prepare modified structure file and/or AFF_{atomtype}.out files, you are safe to run this program: for example, in examples/run_cation.sh file,
```
python run_sq.tpy -i traj.trr -s cation.gro -aff cm -nproc 12 -norm YES -start 0 -o cation.avg
```

## settings
you can get help via option -h:
```
python run_sq.py -h
```
### multiprocessing
* -nproc: #processors to use if possible
### reading trajectory
* -i: trajectory file (any file format available in MDtraj (http://mdtraj.org/)
* -s: structure file (see MDtraj manual)
* -step: read every n-th frame in trajectory file
* -start: n-th starting frame to read after you stride trajecotry using option -step.
* -end: n-th ending frame, which is similar with the option -start
### normalization 
* -norm: applying normalization factors (YES/NO)
### particle mesh ewald (PME) and B-spline settings
To get better resolution of structure factor, smooth particle mesh ewals summation can resolve. Conceptually, atom positions are distributed in 3D-mesh grid using B-spline interpolation. You can see the algorithm in:
> A smooth particle mesh Ewald method
> The Journal of Chemical Physics 103, 8577 (1995); https://doi.org/10.1063/1.470117
> Ulrich Essmann, Lalith Perera, and Max L. Berkowitz
* -pme_grid: #knots in each axis (mesh grid size = #knots * knots * knots)
* -spline_grid: grid size of B-spline interpolation in each axis for coefficients of structure factor at pme grid knots
### atomic form factor
As for generating atomic form factors, we only consider non-dispersive part of atomic scattering factor for neutral atoms as defined in Cromer-Mann equation:
> Compton Scattering Factors for Spherically Symmetric Free Atoms, 
> The Journal of Chemical Physics 47, 1892 (1967); https://doi.org/10.1063/1.1712213, 
> Don T. Cromer and Joseph B. Mann
See details https://www.ruppweb.org/Xray/comp/scatfac.htm
* -aff: read atomic form factor files or generate by Cromer and Mann formula (read/cm)
* -cm_dq: spacing between q's (will be dq_form varible in this script)
* -cm_endq: end q for generating atomic form factor. It should be much greater than the option -end_k 
### isotropic structure factor (output)
* -m_dk: the ratio of isotropic structure factor spacing to spacing of atomic form factor
* -end_k: end q for isotropic structure factor
* -o: output surfix for isotropic structure factor

## Tutorial
examples folder has input and out files for partial structure factor of cations and anions.
Either option "-aff read" or "-aff no" works, thus you can understand atomic form factor file format in AFF_{atomtype}.out.
```
sh run_anion.sh
sh run_cation.sh
```
Then, you get output file "cation.avg" and "anion.avg". As you may know when you plot the files, in fact, it is not good idea to read a few frames for figure quality. Please consider at least hundreds or thounsands frames for publishable figures.
Once you have the structure factor, you may want to smooth the plots:
```
python smooth.py -i cation.avg -nw 60
```
Then, plot cation.avg.smooth and anion.avg.smooth. You can get smooth plots, but smoothing would not be a good option when you are interested in sharp peaks (in high q range for solid/crystal)

## Version history
* v0.1: initial coding
* v0.2: made user-friend parameters

## Benchmarking
Linear scaling depending on data size (#atoms * #frames)
Also, a large end q value makes slower for calculations

(based on q range [0:3] A^(-1))
CPU model | #atoms * #frames | #cpus used | CPU time (s) | wall time (s)
--------- | ---------------- | ---------- | ------------ | ------------
Intel Xeon 2.1 GHz | 144,000 | 12 | 185 | 15
Intel Xeon 2.1 GHz | 1,440,000 | 10 | 2892 | 306
Intel Xeon 2.1 GHz | 7,200,000 | 11 | 9186 | 783
 
## Acknowledge
I converted Fortran progem credited by Dr. Jesse G. McDaniel (Georgia Tech)
 and got help for normalization factor from TingTing Weng (UW Madison)

