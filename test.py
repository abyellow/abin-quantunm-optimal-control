import numpy as np
import matplotlib.pyplot as pl

#Consider function f(t)=1/(t^2+1)
#We want to compute the Fourier transform g(w)

#Discretize time t
t0=-200.
dt=0.001
t = np.arange(t0 ,-t0, dt)

#Define function
f=1./(t**2+1.)

#Compute Fourier transform by numpy's FFT function
g = np.fft.fft(f)
#frequency normalization factor is 2*np.pi/dt
w = np.fft.fftfreq(f.size)*2*np.pi/dt

#print min(g)

#In order to get a discretisation of the continuous Fourier transform
#we need to multiply g by a phase factor
gf = dt * np.exp(-complex(0,1)*w*t0)/(np.sqrt(2*np.pi))
#pl.plot(gf)
#pl.show()
g *= gf

print sum(np.imag(g))
#Plot Result
pl.scatter(w,g,color="r")
#For comparison we plot the analytical solution
pl.plot(w,np.exp(-np.abs(w))*np.sqrt(np.pi/2),color="g")

pl.gca().set_xlim(-10,10)
pl.show()
pl.close()
