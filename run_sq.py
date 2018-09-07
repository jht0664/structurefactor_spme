#!/usr/bin/env python3
# ver 0.1 - Coding python program by Hyuntae Jung on 08/31/2018
#           Please email me (jht0664@gmail.com) for any kind of questions on this program.
#           Originally, it was converted from a Fortran-based program written by Jesse McDaniel.
#           Also, TingTing added some lines, employing normalization factor 
#           I would acknowledge Jesse and TingTing.
# ver 0.2 - Edit for public usages by Hyuntae Jung on 09/07/2018

import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='get structure factor reflecting atomic form factors with spline_order 6')
## args
parser.add_argument('-nproc', '--nproc', default=1, nargs='?', type=int,
	help='number of processor to run') 
parser.add_argument('-i', '--input', default='md.dcd', nargs='?', 
	help='trajectory file')
parser.add_argument('-s', '--structure', default='md.pdb', nargs='?', 
	help='structure file (Recommend to modify atom_type in the file to atom symbol for args.aff)')
parser.add_argument('-start', '--start', default=0, nargs='?', type=int,
	help='start frame to read (will be applied AFTER striding trajectory)')
parser.add_argument('-end', '--end', default=-1, nargs='?', type=int,
	help='end frame to read (negative: end frame of the trajectory) (will be applied AFTER striding trajectory)')
parser.add_argument('-step', '--step', default=1, nargs='?', type=int,
	help='stride time frame to read the trajectory file')
parser.add_argument('-norm', '--norm', default="NO", nargs='?',
	help='normaliziation active? (YES/NO)')
parser.add_argument('-pme_grid', '--pme_grid', default=80, nargs='?', type=int,
	help='# pme grid or pme knots (assume isotropic grid system, then total #knots = args.pme_grid**3)')
parser.add_argument('-spline_grid', '--spline_grid', default=100000, nargs='?', type=int,
	help='# spline grid for b-spline')
parser.add_argument('-aff', '--aff', default="read", nargs='?',
	help='read atomic form factor ("AFF_atomtype.out") or generate by Cromer and Mann formula (read/cm)')
parser.add_argument('-cm_dq', '--cm_dq', default=0.01, nargs='?', type=float,
	help='segment of k value for generating atomic form factor (unit = A^-1) when args.aff = "cm"')
parser.add_argument('-cm_endq', '--cm_endq', default=20.0, nargs='?', type=float,
	help='the boundary of k value for generating atomic form factor (unit = A^-1) when args.aff = "cm"')
parser.add_argument('-m_dk', '--m_dk', default=0.1, nargs='?', type=float,
	help='segment of k value for isotroic structure factor, S(k) (iso_dk = args.m_dk * dq_form)')
parser.add_argument('-end_k', '--end_k', default=3.0, nargs='?', type=float,
	help='the boundary of k value for iso. structure factor (unit = A^-1)')
parser.add_argument('-o', '--output', default='.sq', nargs='?', 
	help='output surfix for structure factor')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.2')
## read args
args = parser.parse_args()
## Check arguments for log
print(" input arguments: {0}".format(args))

# setting for multiprocessing
if args.nproc <= 0:
	raise ValueError("wrong #processors")
elif args.nproc == 1:
	print(" Not commend to use single CPU core for this program")
import os
os.environ["MKL_NUM_THREADS"] = str(args.nproc)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.nproc)
os.environ["OMP_NUM_THREADS"] = str(args.nproc)
## import modules
import mdtraj as md
import numpy as np
import copy
import math

## runing parameters 
pme_grid=args.pme_grid # determine pme_max_print, e.g. output s_q size for isotropic s_k 
spline_order=6
spline_grid=args.spline_grid
xyz_scale_small_zero_creteria=0.000001

if spline_order != 6:
	raise ValueError(" supported only for B-spline order 6")

# exclude drudes and virtual atoms "X" from coordinate file
traj = md.load(args.structure)
top = traj.topology 
n_atoms_wo_drudes = top.n_atoms - len(top.select("name =~ 'D'")) # counts #atoms without drude particles
select_atoms = top.select("(not (name =~ 'X')) and (not (name =~ 'D'))") # select all atoms wihout virtual and drdue atoms

# read trajectory
traj = md.load(args.input,top=args.structure,atom_indices=select_atoms,stride=args.step) # need to change nano into anstrom?
if args.end <= 0:
	traj = traj[args.start:]
else:
	if args.end < args.start:
		raise ValueError("args.end should be greater than args.start")
	traj = traj[args.start:args.end]
top = traj.topology 
n_atoms = top.n_atoms
n_frames = len(traj.xyz)
box=traj.unitcell_vectors[0]*10.0 # assume NVT ensemble (no change on cell vector) and convert nm to angstrom.
print(" total # loaded frames from {} to {} = {}".format(args.start,args.end,len(traj.xyz)))
print("Done: reading trajectory")

###############################################################
def construct_reciprocal_lattice_vector(box_3d):
	out = np.empty((3,3))
	a = box_3d[0]
	b = box_3d[1]
	c = box_3d[2]
	# calculate the volume and the reciprocal vectors (notice no 2pi)
	#  assume primitive cell
	vol = np.abs(np.dot(a,np.cross(b,c))) 
	kc = np.cross(a,b)/vol
	ka = np.cross(b,c)/vol
	kb = np.cross(c,a)/vol 
	out[0]=ka
	out[1]=kb
	out[2]=kc
	return copy.copy(out) 
###############################################################

# construct_reciprocal_lattice_vector
kk = construct_reciprocal_lattice_vector(box)
# determine pme_max_print, e.g. output s_q size for isotropic s_k 
pme_max_print = np.int_(args.end_k/(2.0*math.pi*np.amax(kk)))
if pme_max_print >= pme_grid:
	print("pme_max_print {} should be less than pme_grid {}.".format(pme_max_print,pme_grid))
	raise ValueError(" Please reduce args.end_k {} or increase pme_grid.".format(args.end_k))
print(" automatically set pme_max_print {}".format(pme_max_print))

################################################################
# read form factor file, then return form factor array
def read_aff(atom_type_name):
	openfile = np.loadtxt("AFF_"+str(atom_type_name)+".out")
	type_q, type_form = np.transpose(openfile) # unit = angstrom^(-1) or A^(-1)
	dq = type_q[1]-type_q[0]
	return type_form, dq

## Cromer and Mann table
cm_atomlist = np.array(["H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Se","Br","Kr","Rb","I","Xe"])
# ordering is the same as cm_atomlist
#  parameter set(a1, a2, a3, a4, b1, b2, b3, b4, c)
cm_paramters = np.array([[	0.493	,	0.323	,	0.14	,	0.041	,	10.511	,	26.126	,	3.142	,	57.8	,	0.003	],
[	0.873	,	0.631	,	0.311	,	0.178	,	9.104	,	3.357	,	22.928	,	0.982	,	0.006	],
[	1.128	,	0.751	,	0.618	,	0.465	,	3.955	,	1.052	,	85.391	,	168.261	,	0.038	],
[	1.592	,	1.128	,	0.539	,	0.703	,	43.643	,	1.862	,	103.483	,	0.542	,	0.038	],
[	2.055	,	1.333	,	1.098	,	0.707	,	23.219	,	1.021	,	60.35	,	0.14	,	-0.193	],
[	2.31	,	1.02	,	1.589	,	0.865	,	20.844	,	10.208	,	0.569	,	51.651	,	0.216	],
[	12.213	,	3.132	,	2.013	,	1.166	,	0.006	,	9.893	,	28.997	,	0.583	,	-11.529	],
[	3.049	,	2.287	,	1.546	,	0.867	,	13.277	,	5.701	,	0.324	,	32.909	,	0.251	],
[	3.539	,	2.641	,	1.517	,	1.024	,	10.283	,	4.294	,	0.262	,	26.148	,	0.278	],
[	3.955	,	3.112	,	1.455	,	1.125	,	8.404	,	3.426	,	0.231	,	21.718	,	0.352	],
[	4.763	,	3.174	,	1.267	,	1.113	,	3.285	,	8.842	,	0.314	,	129.424	,	0.676	],
[	5.42	,	2.174	,	1.227	,	2.307	,	2.828	,	79.261	,	0.381	,	7.194	,	0.858	],
[	6.42	,	1.9	    ,	1.594	,	1.965	,	3.039	,	0.743	,	31.547	,	85.089	,	1.115	],
[	6.292	,	3.035	,	1.989	,	1.541	,	2.439	,	32.334	,	0.678	,	81.694	,	1.141	],
[	6.435	,	4.179	,	1.78	,	1.491	,	1.907	,	27.157	,	0.526	,	68.164	,	1.115	],
[	6.905	,	5.203	,	1.438	,	1.586	,	1.468	,	22.215	,	0.254	,	56.172	,	0.867	],
[	11.46	,	7.196	,	6.256	,	1.645	,	0.01	,	1.166	,	18.519	,	47.778	,	-9.557	],
[	7.484	,	6.772	,	0.654	,	1.644	,	0.907	,	14.841	,	43.898	,	33.393	,	1.444	],
[	8.219	,	7.44	,	1.052	,	0.866	,	12.795	,	0.775	,	213.187	,	41.684	,	1.423	],
[	8.627	,	7.387	,	1.59	,	1.021	,	10.442	,	0.66	,	85.748	,	178.437	,	1.375	],
[	17.001	,	5.82	,	3.973	,	4.354	,	2.41	,	0.273	,	15.237	,	43.816	,	2.841	],
[	17.179	,	5.236	,	5.638	,	3.985	,	2.172	,	16.58	,	0.261	,	41.433	,	2.956	],
[	17.355	,	6.729	,	5.549	,	3.537	,	1.938	,	16.562	,	0.226	,	39.397	,	2.825	],
[	17.178	,	9.644	,	5.14	,	1.529	,	1.789	,	17.315	,	0.275	,	164.934	,	3.487	],
[	20.147	,	18.995	,	7.514	,	2.273	,	4.347	,	0.381	,	27.766	,	66.878	,	4.071	],
[	20.293	,	19.03	,	8.977	,	1.99	,	3.928	,	0.344	,	26.466	,	64.266	,	3.712	]]
)

def gen_aff_cm(atom_type_name,end_k, dk):
	k_list = np.arange(0,stop=end_k,step=dk) # unit = A^-1
	len_k = len(k_list)
	square_k_over_2pi = np.square(k_list/2.0/math.pi) # unit = A^-2
	if not atom_type_name in cm_atomlist:
		raise ValueError(" atom type name {} is not found in Cromer and Mann table for atomic structure factors".format(atom_type_name))
	a1,a2,a3,a4,b1,b2,b3,b4,c = cm_paramters[np.where(cm_atomlist == atom_type_name)][0]
	out_aff = c+a1*np.exp(-b1*square_k_over_2pi)+a2*np.exp(-b2*square_k_over_2pi)+a3*np.exp(-b3*square_k_over_2pi)+a4*np.exp(-b4*square_k_over_2pi)
	return out_aff, dk

# get atom type, removing mole name "Cho5-O" into "O"
def name_to_atom_type(top_atom):
	a,b=str(top_atom).split('-')
	return b
################################################################

# generate atom type list
atype_name=[]
alist=[]
for i_atom in range(n_atoms):
	get_name = name_to_atom_type(top.atom(i_atom)) # same as "aname" variable
	alist.append(get_name)
	if get_name in atype_name: 
		continue
	else: # not exist for the atom type
		atype_name.append(get_name)

alist = np.array(alist)
atype_name = np.array(atype_name)

# get form factors following ordering of atom type list
n_atom_type = len(atype_name)
print(" found {} atom types in coord file".format(n_atom_type))
for iatype in range(n_atom_type):
	if iatype == 0: # initialize atomtype_form size
		if "read" in args.aff:
			temp_form, dq_form = read_aff(atype_name[iatype])
		elif "cm" in args.aff:
			temp_form, dq_form = gen_aff_cm(atype_name[iatype],args.cm_endq,args.cm_dq)
		n_qs = len(temp_form) # a large value, n_qs, can provide good resolution for long wave vector, q.
		check_dq_form = copy.copy(dq_form)
		print(" dq_form (A^-1) = {}".format(check_dq_form))
		atomtype_form = np.empty((n_atom_type,n_qs))
		atomtype_form[iatype] = copy.copy(temp_form)
	else:
		if "read" in args.aff:
			atomtype_form[iatype], dq_form = read_aff(atype_name[iatype])
		elif "cm" in args.aff:
			atomtype_form[iatype], dq_form = gen_aff_cm(atype_name[iatype],args.cm_endq,args.cm_dq)
		if dq_form != check_dq_form:
			raise IOError(
				" inconsistent form factor dq's {} for {}-th atomtype (name {}) (ref. initial dq's {})"
				.format(dq_form,iatype,str(atype_name[iatype]),check_dq_form))

# count number of atoms with respect to atom types (to do normalization of s(q))
#  writeen by TingTing and edited by Hyuntae
natom_type_kind = np.empty(n_atom_type,dtype=np.int_)
for i_type in range(n_atom_type):
	natom_type_kind[i_type] = len(np.where(alist == atype_name[i_type])[0])
frac_natom_type_kind = np.float_(natom_type_kind)/np.float_(n_atoms_wo_drudes)
print("Done: reading form factor files and make atom type list")

##########################################################################
#***********************************************
# this function calculates B_splines which are used in pme as interpolating
# functions.  B_splines are calculated recursively, and therefore it's a good idea
# to grid them
#************************************************
def b_spline(u,n):
	# define m2 for n-1 values
	mn=np.zeros((n,n))
	for i in range(1,n):
		ui = u - np.float_(i-1)
		if (ui < 0.0) or (ui > 2.0):
			mn[1,i] = 0.0
		else:
			mn[1,i] = 1.0 - np.abs(ui - 1.0)
	# define mj recursively for n-1-(j-1) values
	for j in range(2,n):
		for i in range(1,n-j+1):
			ui = u - np.float_(i-1)
			mn[j,i]=(ui/np.float_(j))*mn[j-1,i]+((np.float_(j+1)-ui)/np.float_(j))*mn[j-1,i+1]
	return mn[n-1,1]

#******************************************************
# this is needed in reciprocal space pme calculation
#******************************************************
def bm(m,n,k):
	sum=np.zeros(1,dtype=np.complex_)
	for i in range(n-1):
		tmp=2.0*math.pi*m*i/np.float_(k)
		sum=sum+b_spline(np.float_(i+1),n)*np.complex(math.cos(tmp),math.sin(tmp))
	tmp=2.0*math.pi*(n-1)*m/np.float_(k)
	return np.complex(math.cos(tmp),math.sin(tmp))/sum

# construct b_array
def b_array_init(k,n):
	data = np.empty((k,k,k),dtype=np.complex_)
	basis_bm = np.empty(k,dtype=np.complex_)
	for i in range(k):
		basis_bm[i]=bm(i,n,k)
	for i in range(k):
		for j in range(k):
			for l in range(k):
				data[i,j,l]=basis_bm[i]*basis_bm[j]*basis_bm[l]
	return copy.copy(data)
##########################################################################	

## initialize bspline interpolation
b_array = b_array_init(pme_grid,spline_order) # same as "B" variable in fortran
# grid B_splines
b6_spline = np.empty(spline_grid)
for i in range(spline_grid):
	b6_spline[i] = b_spline(6.0/np.float_(spline_grid)*np.float_(i+1),6)
print("Done: calculation of b_spline coefficients")

######################################################################
#********************************************
# this subroutine creates direct coordinates, scaled
# by input integer "K" (pme_grid), using the
# reciprocal lattice vectors
# 
# note, only coordinates are returned for atoms of type "i_type"
#********************************************
def create_scaled_direct_coordinates(xyz, resiprocal_k, k):
	out = np.float_(k)*np.einsum('ijk,lk->ijl',xyz*10.0,resiprocal_k) # "10" means to convert nm to angstrom 
	return np.mod(out,np.float_(k))
######################################################################

## calculate structure factors
# trim trajectories and computing full structure factor 
xyz_scale = create_scaled_direct_coordinates(traj.xyz, kk, pme_grid)
# make sure scaled coordinates are not numerically equal to zero, otherwise this will screw up Q grid routine
xyz_scale[np.where(xyz_scale < xyz_scale_small_zero_creteria)] = xyz_scale_small_zero_creteria
print("Done: make (scaled) resiprocal xyz coordinates for pme")

#########################################################################
#*****************************************************************
# This subroutine interpolates charges onto Q grid to be used in pme reciprocal space
# routines
#*****************************************************************
def grid_Q(xyz,K,n,n_grid,bo_spline):
	out_q = np.zeros((K,K,K))
	n_atom=np.shape(xyz)[0]
	u = np.empty(3)
	arg = np.empty(3) # distance between (original pt - nearpt), adding one element from range(spline_order)
	for j in range(n_atom):
		u=xyz[j]
		nearpt=np.int_(np.floor(u))
		# only need to go to k=0,n-1, for k=n, arg > n, so don't consider this
		for k1 in range(n):
			n1 = nearpt[0]-k1
			arg[0] = u[0]-np.float_(n1)
			# shift index of array storage if < 0
			n1 = np.mod(n1,K)
			for k2 in range(n):
				n2 = nearpt[1]-k2
				arg[1] = u[1]-np.float_(n2)
				n2 = np.mod(n2,K)
				for k3 in range(n):
					n3 = nearpt[2]-k3
					arg[2] = u[2]-np.float_(n3)
					n3 = np.mod(n3,K)
					# find spline index
					splindex = np.ceil(arg/6.0*np.float_(n_grid))
					splindex = np.int_(splindex) 
					# note 0<arg<n , so arg should always be within bounds of gridded spline
					# we assume spline_order is 6
					sum = bo_spline[splindex[0]]*bo_spline[splindex[1]]*bo_spline[splindex[2]]
					out_q[n1,n2,n3] = out_q[n1,n2,n3] + sum
	return copy.copy(out_q)

#######################################################
# Fast 1st edition for grid_Q function by Hyuntae, instead of grid_Q function
# 1. due to the periodicity of numpy array, no need to redefine n1, n2, and n3
# 2. all combinations of n1, n2, n3 are generated by np.meshgrid
# 3. remove for-loops if possible
# This module is faster by three times than grid_Q in python,
#   which shows similar performace compared with Fortran program using grid_Q 
#######################################################
def grid_Q_1ed(xyz,K,n,n_grid,bo_spline):
	out_q = np.zeros((K,K,K))
	n_atom=np.shape(xyz)[0]
	for j in range(n_atom):
		u=xyz[j]
		nearpt=np.int_(np.floor(u))
		# generate all possible n vectors based on nearpt via np.meshgrid
		# indexing='ij' -> change x value at first, then y and z.
		# indexing='xy' -> change y value at first, then x and z.
		n_vectors = np.array(np.meshgrid(np.arange(nearpt[0],nearpt[0]-n,-1),
			np.arange(nearpt[1],nearpt[1]-n,-1),
			np.arange(nearpt[2],nearpt[2]-n,-1),indexing='xy')).T.reshape(-1,3)
		arg=np.float_(u-n_vectors)
		splindex = np.int_(np.floor(arg/6.0*np.float_(n_grid)))
		sum_coeff = b6_spline[splindex[:,0]]*b6_spline[splindex[:,1]]*b6_spline[splindex[:,2]]
		out_q[n_vectors[:,0],n_vectors[:,1],n_vectors[:,2]] = \
			out_q[n_vectors[:,0],n_vectors[:,1],n_vectors[:,2]] + sum_coeff
	return copy.copy(out_q)

def combine_partial_structure_factors(sq_store,n_atom_type,pme_grid):
	out = np.zeros((n_atom_type,n_atom_type,pme_grid,pme_grid,pme_grid))
	for i_type in range(n_atom_type):
		for j_type in range(i_type,n_atom_type):
			if i_type == j_type:
				out[i_type,j_type] = (np.real(sq_store[i_type])**2) \
					+ (np.imag(sq_store[i_type])**2)
			else:
				# here we have unlike types, so we want SQi(c.c.) * SQj + SQi * SQj(c.c.) , where
				# (c.c.) is complex conjugate, which equals 2 * real ( SQi * SQj(c.c.) )
				out[i_type,j_type] = 2.0 * np.real(sq_store[i_type] * np.conjugate(sq_store[j_type]))
	return out
########################################################################

## calculate structure factor for all time frames
sq_store  = np.empty((n_atom_type,pme_grid,pme_grid,pme_grid),dtype=np.complex_) # same as "SQ_store" variable in fortran
sq2_store = np.zeros((n_atom_type,n_atom_type,pme_grid,pme_grid,pme_grid))

for i_frame in range(n_frames):
	if i_frame%10 == 0:
		print(" ... calculating {} th frame ... ".format(i_frame))
	for i_type in range(n_atom_type):
		q_grid = grid_Q_1ed(xyz_scale[i_frame][np.where(alist == atype_name[i_type])], 
			pme_grid, spline_order, spline_grid, b6_spline)
		q_1r = q_grid.reshape(-1)
		q_1d = np.fft.fft(np.complex_(q_1r))
		fq   = q_1d.reshape((pme_grid,pme_grid,pme_grid))
		# structure factor = b_array * FQ
		sq = fq*b_array
		sq_store[i_type] = copy.copy(sq)
	# now create all the cross SQ2 structure factors for this snapshot
	sq2_store = sq2_store + combine_partial_structure_factors(sq_store,n_atom_type,pme_grid)
print("Done: partial sq calculations for each frame and each atom type")

################################################################################
# add normalizatino factor, f2_norm, by TingTing
def add_partial_structure_factors(sq2_store, kk, natom_type_kind, pme_max_print, atomtype_form, dq_form):
	out = np.zeros((pme_max_print,pme_max_print,pme_max_print)) 
	k_mag_out = np.zeros((pme_max_print,pme_max_print,pme_max_print)) 
	# here we add only those structure factors that will be printed, to save time
	for l_k in range(pme_max_print):
		for j_k in range(pme_max_print):
			for i_k in range(pme_max_print):
				# convert wavevector, note the reciprocal lattice vectors kk don't have the 2*pi factor
				k_vec = 2.0 * math.pi * ( np.float_(i_k) * kk[0] + np.float_(j_k) * kk[1] + np.float_(l_k) * kk[2])
				#k_vec_out[i_k,j_k,l_k,:] = copy.copy(k_vec)
				k_mag = np.sqrt(np.dot(k_vec,k_vec))
				k_mag_out[i_k,j_k,l_k] = k_mag
				# normalization factor written by TingTing
				f2_norm = np.zeros(1)
				n_atom_type = len(natom_type_kind)
				for i_type in range(n_atom_type):
					#xi = frac_natom_type_kind[i_type]
					fi = get_form_fac(i_type, k_mag, atomtype_form, dq_form)
					for j_type in range(i_type,n_atom_type):
						#xj = frac_natom_type_kind[j_type]
						fj = get_form_fac( j_type, k_mag, atomtype_form, dq_form)
						out[i_k,j_k,l_k] = out[i_k,j_k,l_k] \
							+ fi * fj * sq2_store[i_type,j_type,i_k,j_k,l_k]
					f2_norm = f2_norm + fi * fi * natom_type_kind[i_type]
				if (f2_norm == np.inf) or (f2_norm == 0.0):
					raise ValueError(" Need to increase atomic form factor range [0,{}]".format(f2_norm))
				if args.norm == "YES":
					out[i_k,j_k,l_k] = out[i_k,j_k,l_k]/f2_norm
	return copy.copy(k_mag_out), copy.copy(out)

#*************************
# if q_mag is greater than the longest qvec for which we
# have the form factor, then return 0
#*************************
def get_form_fac(i_index, q_mag, atomtype_form, dq_form):
	try:
		out = atomtype_form[i_index,np.int(np.floor(q_mag/dq_form))]
	except IndexError:
		out = 0.0
	return out
################################################################################

# average structure factor but within (pme_max_print,pme_max_print,pme_max_print) range
sq2_store = sq2_store/np.float_(n_frames) 
kk_avg = kk # we assume no volume change
# apply form factor
k_mag, sq2 = add_partial_structure_factors(sq2_store, kk_avg, natom_type_kind, pme_max_print, atomtype_form, dq_form)

## write raw_data
#raw_data = np.column_stack((k_vec.flatten(), sq2.flatten()))
#np.savetxt(args.output+'.raw', raw_data, 
#	header='flat k_vec ({}), flat sq2 ({})'.format(k_vec.shape,sq2.shape), 
#	fmt='%f', comments='# ')
#np.save(args.output+'.raw', raw_data)

## write isotropic s_k
# sums up all contributions to S(k) at |kvec|=k over S(kvec)
iso_dk = np.float_(args.m_dk) * dq_form
print(" segment k of iso. s_k (A^-1) = {}".format(iso_dk))
#max_i = np.int_(np.floor(np.argmax(k_mag)/iso_dk))
max_i = np.int_(np.ceil(np.float(args.end_k)/iso_dk))
sk = np.zeros(max_i)
countk = np.zeros(max_i,dtype=np.int_)
for l_k in range(pme_max_print):
	for j_k in range(pme_max_print):
		for i_k in range(pme_max_print):
			iso_i = np.int_(np.floor(k_mag[i_k,j_k,l_k]/iso_dk))
			if iso_i >= max_i:
				continue
			sk[iso_i] = sk[iso_i] + sq2[i_k,j_k,l_k]
			countk[iso_i] = countk[iso_i] + 1

with np.errstate(divide='ignore', invalid='ignore'):
	avg_sk = np.true_divide(sk,np.float_(countk))
	avg_sk[avg_sk == np.inf] = 0
	avg_sk = np.nan_to_num(avg_sk)

# reduce array size using args.end_k
list_dk = np.arange(start=0,stop=args.end_k,step=iso_dk)
out_avg_sk = np.column_stack((list_dk[1:],avg_sk[1:len(list_dk)]))

# write text file
np.savetxt(args.output, out_avg_sk,
	header='iso k (A^-1), iso s(k) (arbitary)', fmt='%f', comments='# ')
#np.save(args.output, out_avg_sk)
print("Done: writing iso sq {}".format(args.output))
