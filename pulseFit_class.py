import numpy as np
import matplotlib.pyplot as pl
#from iniData_class import iniData
from scipy.optimize import curve_fit
import sys
from scipy.signal import savgol_filter

class pulseFit():

	def __init__(self, xdata, ydata, fitNum=3):

		self.xdata = xdata
		self.ydata = ydata
		self.fitNum = fitNum
#		self.dt = dt
#		self.time = np.array(range(len(self.ctrl)))*self.dt

	def funcFit(self,xdata,a1, s1):#a1,a2,a3,w1,w2,w3):
		
		y = a1 * np.exp(-xdata**2/s1**2) #(a1*np.sin(w1*xdata) + a2*np.sin(w2*xdata) + a3*np.sin(w2*xdata)) * np.exp(-xdata**2/25)
		return y

	def fit(self,startVal=None):
		pass
	#	popt, pcov = curve_fit(self.funcFit, self.xdata, self.ydata, p0 = startVal)
	#	print popt, pcov#, np.sqrt(np.diag(pcov))

if __name__=='__main__':

	"""
	kMul = 6
	dtime = .01
	iter = 10000

	kGoal = int(sys.argv[1])
	Tpulse = np.double(sys.argv[2])
	kWidGoal = int(sys.argv[3])

	init = iniData(kWidGoal = kWidGoal, dt = dtime, LoadFileIterNum=iter, knumload = 12*kMul+1, 
				knum=12*kMul+1, kPosGoal=kGoal, kPosGoalLoad=kGoal,
			 		T_pulse = Tpulse, ChooseFile = 'LOAD')

	#init.plot_At(init.ctrl_c, init.ctrl_s, tin = 0)
	"""
	
	
	loadname1 = 'xy.dat'
	data1 = np.loadtxt(loadname1)
	x = data1[:,0]
	y = data1[:,1]
#	print y
#	print sum(y)/len(y), sum(x)/len(x)
#	x -= sum(x)/len(x)
#	y -= sum(y)/len(y)

	#window = 21
	#poly = 2
	#y = savgol_filter(y, window, poly)

	lon = len(y)
	print lon
	pl.plot(x,y)
#	pl.plot(x,ynew, '--')
	pl.show()
	
#!!!	y = np.append(y,np.zeros(lon))

	dt = .01
	xf = np.fft.rfftfreq(lon, dt)#*2*np.pi
	yf = (np.fft.rfft(y))*dt/2
#	print len(y), len(yf)
	yf2 = (abs(yf))**2
	hbd = max(yf2)
	#yff = [yf[i] if yf2[i]>.01*hbd else 0 for i in range(len(yf))]
	#yff = [yf[i] for i in range(len(yf)) if yf2[i]>.01*hbd]
	#xff = [xf[i] for i in range(len(yf)) if yf2[i]>.01*hbd]
	window = 51 
	poly = 3 
	yffabs = savgol_filter(abs(yf), window, poly)
	yff = yffabs * yf/abs(yf)
#	yff = yf
	
	#yffr = savgol_filter(yf.real, window, poly)
	#yffi = savgol_filter(yf.imag, window, poly)
	#yff = yffr+1j*yffi
	pl.plot(xf[:], np.absolute(yf[:]),'--')
	pl.plot(xf[:], np.absolute(yff[:]))
	#pl.plot(xf[:], np.imag(yff[:]))
#	pl.semilogx(xf[:],yf[:])
	pl.xlabel('$\omega$')
	pl.ylabel('$F(\omega)$')
#	pl.xlim((-15,15))
	pl.show()	


 	func2 = lambda xdata, a, s : a * np.exp(-abs(xdata/s)**1)
	popt2, pcov2 = curve_fit(func2, xf, np.double(abs(yf)))
	print popt2, sum(pcov2)
	y2 = func2(xf, popt2[0], popt2[1])
	pl.plot(xf,yf,'--')
	pl.plot(xf,y2)# *yff/abs(yff))
	pl.show()
	"""
 	func3 = lambda xdata, a1, a2, s1, s2 : a1 * np.exp(-abs(xdata/s1)**1) + a2 * np.exp(-abs(xdata/s2)**1)
	popt3, pcov3 = curve_fit(func3, xf, np.double(abs(yf)))
	print popt3, sum(pcov3)
	y3 = func3(xf, popt3[0],popt3[1],popt3[2],popt3[3])
	pl.plot(xf,yf,'--')
#	pl.plot(xf,yffabs,'k')
	pl.plot(xf,y3)# *yff/abs(yff))
	pl.show()
	"""	
	
	xi = np.fft.rfftfreq(2*lon-1, (xf[1]-xf[0])/2)
	yi = np.fft.irfft(yff)*2/dt
	y2i = np.fft.irfft(y2*yff/abs(yff))*2/dt
	print len(xi), len(yi)
	pl.plot(x,y,'--')
	pl.plot(x, yi, 'k',linewidth = 2.0)
	pl.plot(x, y2i, 'g', linewidth = 2.)
	pl.show()

#	print [i for i in y2i/abs(y2i)]
#	print sum([1 for i in y2i if i>0 ])
#	print sum([-1 for i in y2i if i<=0 ])

	savename = 'xiyi.dat'
	np.savetxt(savename,np.column_stack((x,y2i)))

	loadname2 = 'xiyi.dat'
	data2 = np.loadtxt(loadname2)

	xl = data2[:,0]
	yl = data2[:,1]

	pl.plot(xl,yl)
	pl.show()
	"""
	xlf = np.fft.rfftfreq(lon, dt)#*2*np.pi
	ylf = (np.fft.rfft(yl))*dt
	pl.plot(xlf,abs(ylf))
	pl.show()
	"""
	ycos, ysin = np.cos(yl), np.sin(yl)
	yexp = ycos + 1j * ysin	
	yy  = np.real(-1j*np.log(yexp))
	pl.plot(xl, yl, '--')
	pl.plot(xl, yy) 
	pl.show()	

	

	xxf = np.fft.rfftfreq(lon, dt)#*2*np.pi
	yyf = (np.fft.rfft(yy))*dt
	pl.plot(xf, abs(y2),"--")
	pl.plot(xxf,abs(yyf))
	pl.show()


#	print len(y), len(yf)
#	pl.plot(x, np.cos(yl))
#	pl.show()
	#print np.size(data), np.size(xf), np.size(yf)
	"""
	# test 1
 	func = lambda xdata, a, s : a * np.exp(-(xdata/s)**2)
	popt, pcov = curve_fit(func, xf, yf)
	print popt, sum(pcov)
	y1 = func(xf, popt[0], popt[1])
	

	# test 2
 	func2 = lambda xdata, a, s : a * np.exp(-abs(xdata/s)**1)
	popt2, pcov2 = curve_fit(func2, xf, yf)
	print popt2, sum(pcov2)
	y2 = func2(xf, popt2[0], popt2[1])
	#plot 
	pl.plot(xf, yf, 'x')
	pl.plot(xf, y1, 'k',linewidth = 2.0)
	pl.plot(xf, y2, 'r', linewidth = 2.0)
	pl.show()
	#test 1 cutoff
	xnew  = [xf[i] for i in range(len(y1)) if y1[i]<yf[i]]
	ynew  = [yf[i] for i in range(len(y1)) if y1[i]<yf[i]]

	popt3, pcov3 = curve_fit(func, xnew, ynew, p0 = popt)
	print popt3, sum(pcov3)
	y3 = func(xnew, popt3[0], popt3[1])


	#test 2 cutoff
	xnew2  = [xf[i] for i in range(len(y2)) if y2[i]<yf[i]]
	ynew2  = [yf[i] for i in range(len(y2)) if y2[i]<yf[i]]

	popt4, pcov4 = curve_fit(func2, xnew2, ynew2, p0 = popt2)
	print popt4, sum(pcov4)
	y4 = func2(xnew2, popt4[0], popt4[1])
	#plot cutoff
	pl.plot(xnew, ynew, 'x')
	pl.plot(xnew, y3, 'k',linewidth = 2.0)
	pl.plot(xnew2, y4, 'r',linewidth = 2.0)

	pl.show()

	
	#test 3 savgol_filter:
	window = 11 
	poly = 4

	ynew3 = savgol_filter(yf, window, poly)

	bd = 2.* (sum(yf)/ len(yf))
	print bd
	xnew5  = [xf[i] for i in range(len(ynew2)) if bd<yf[i]]
	ynew5  = [yf[i] for i in range(len(ynew2)) if bd<yf[i]]
	ynew5 = savgol_filter(ynew5, window, poly)	
	wid = 10000
	Ni = lon/2 - wid
	Nf = lon/2 + wid
	pl.plot(xf[Ni:Nf], yf[Ni:Nf], 'x')
	pl.plot(xf[Ni:Nf], ynew3[Ni:Nf], 'k',linewidth = 2.0)
	xwid = 20
	pl.xlim((-xwid,xwid))
#	pl.plot(xnew2, ynew2, 'x')
#	ynew6 = savgol_filter(ynew2,window,poly)
#	pl.plot(xnew2, ynew6, 'r',linewidth = 2.0)

	pl.show()


	#test 4 cutoff + cut_bd:

	bd = 1.* (sum(ynew2)/ len(ynew2))
	print bd
	xnew4  = [xnew2[i] for i in range(len(ynew2)) if bd<ynew2[i]]
	ynew4  = [ynew2[i] for i in range(len(ynew2)) if bd<ynew2[i]]

	popt6, pcov6 = curve_fit(func2, xnew4, ynew4, p0 = popt4)
	print popt6, sum(pcov6)
	y6 = func2(xnew4, popt6[0], popt6[1])
	#plot cutoff
	pl.plot(xnew4, ynew4, 'x')
	pl.plot(xnew4, y6, 'k',linewidth = 2.0)
#	pl.plot(xnew2, y4, 'r',linewidth = 2.0)

	pl.show()
	"""
