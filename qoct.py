"""
 Editor Bin H.
 Quantum Optimal Control Example
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

class QH:
	"""
	Initial data/conditions of Quantum Hamiltonian and initial states.
	"""
	def __init__(self, H0, Hctrl, ctrl_i, phi_i, dt=.01):

		self.H0 = H0          #Hamiltonian with no control/laser
		self.Hctrl = Hctrl    #Hamiltonian of control/laser term
		self.ctrl_i = ctrl_i  #initial control/laser
		self.phi_i = phi_i    #initial quantum states
		self.dt = dt          #time step size
		self.t_ini = 0.       #start time

		self.dim = np.shape(self.H0)[0]  #dimension of Hamiltonian
		self.tim_all = np.shape(self.ctrl_i)[0] #time length of ctrl/laser
		self.tim_real = np.array(range(self.tim_all+1)) * self.dt +\
				 self.t_ini	#real time of time length 
	
	def __u_next(self,H,u_now):
		"""Derive U at next time step"""
		return np.dot(expm(-1j*H*self.dt), u_now)

	def u_t(self):
		"""Evolve propergator for given time period"""
		dim = self.dim 
		tim_all = self.tim_all
		ctrl = self.ctrl_i
		H0 = self.H0
		Hctrl = self.Hctrl
		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[0,:,:] = np.eye(dim)

		for tim in xrange(tim_all):
			H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl) )
			u_all[tim+1,:,:] = self.__u_next(H, u_all[tim,:,:])

		return u_all

	def phi_t(self):
		"""Evolve state for given time period"""
		dim = self.dim
		tim_all = self.tim_all 
		phi_all = np.zeros((tim_all+1,dim,1),dtype = complex)
		phi_all[0,:,:] = self.phi_i[:]
		u_all = self.u_t()

		for tim in xrange(tim_all):
			phi_all[tim+1,:,:] = np.dot(u_all[tim+1,:,:], phi_all[0,:,:])
		
		return phi_all



class QOCT:
	"""
	Quantum optimal control codes
	"""

	def __init__(self, QH):
		
		self.errorbd = 10**-4 # error bound of convergence 
		self.QH = QH          #class QH for all i.c. and EoM
		self.phi_g            #goal quantum states we expect
	
	def u_t_rev(self, H, phi_g):
		pass	

	def new_ctrl(self, phi_now, psi_now, Hctr):
		pass
		
	def fidelity(phi_i_fi, phi_g):
		pass

	def run(self):
		pass
		"""
		ctrl = self.ctrl_i	
		phi_t  = EoM(H, phi_i,ctrl)
		
		for it in range(iter_time):
			
			psi_t = revEoM(H, phi_g, ctrl)
			fi = fidelity(phi_t, phi_g)
			print 'Error:', 1-fi
			
			if 1-fi < self.errorbd:
				break
		
			for tim in range():
				new_ctrl = renewCtrl()
				phi_t[]  = EoM_next()

		return ctrl 	
		"""

if __name__ == '__main__':

	H0 = np.matrix([[1,0],[0,-1]])
	Hctr = [[0,1],[1,0]]
	ctrl = 1.*np.ones(1000)
	#phi = [[1],[1]]/np.sqrt(2)
	phi = [[0],[1]]
		
	qh_test = QH(H0, Hctr, ctrl, phi)
	time = qh_test.tim_real
	phi = qh_test.phi_t()
	prob = phi*np.conjugate(phi)
	print np.sum(prob,axis = 1)[:20]
	
	#plt.plot(time, phi[:,1,:].real)
	#plt.plot(time, phi[:,1,:].imag)
	plt.plot(time, prob[:,0,:])
	plt.plot(time, prob[:,1,:])
	plt.show()



