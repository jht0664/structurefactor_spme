import argparse
parser = argparse.ArgumentParser(
	formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
	description='simply smooth 1d line with n windows')
parser.add_argument('-i', '--input', default='sq_tot', nargs='?', 
	help='the text file name with two columns (x,y)')
parser.add_argument('-nw', '--nw', default=4, nargs='?', type=int,
	help='number of windows to average')
parser.add_argument('-o', '--output', default='.smooth', nargs='?', 
	help='output surfix for smooth data')
parser.add_argument('args', nargs=argparse.REMAINDER)
parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')
## read args
args = parser.parse_args()
## Check arguments for log
print(" input arguments: {0}".format(args))

import numpy as np

# from https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
# smooth the data using a window with requested size.
#  This method is based on the convolution of a scaled window with the signal.
#  The signal is prepared by introducing reflected copies of the signal 
#  (with the window size) in both ends so that transient parts are minimized
#  in the begining and end part of the output signal.
def smooth(x,window_len=11,window='flat'):
	if x.ndim != 1:
		raise ValueError("smooth only accepts 1 dimension arrays.")
	if x.size < window_len:
		raise ValueError("Input vector needs to be bigger than window size.")
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y[int(window_len/2.0):len(x)+int(window_len/2.0)]

data=np.loadtxt(args.input)
rem_zero = np.where(data[:,1] == 0)
skim_out = np.delete(data,rem_zero,axis=0)
skim_out_interp = np.interp(data[:,0],skim_out[:,0],skim_out[:,1])
skim_out_interp_smooth = smooth(skim_out_interp,args.nw)
np.savetxt(args.input+args.output,np.column_stack((data[:,0],skim_out_interp_smooth)))

