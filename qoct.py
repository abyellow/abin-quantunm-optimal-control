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
		self.ctrl = ctrl_i  #initial control/laser
		self.phi_i = phi_i    #initial quantum states
		self.dt = dt          #time step size
		self.t_ini = 0.       #start time

		self.dim = np.shape(self.H0)[0]  #dimension of Hamiltonian
		self.tim_all = np.shape(self.ctrl)[0] #time length of ctrl/laser
		self.tim_real = np.array(range(self.tim_all+1)) * self.dt +\
				 self.t_ini	#real time of time length 
	
	def u_next(self,H,u_now):
		"""Derive U at next time step"""
		return np.dot(expm(-1j*H*self.dt), u_now)

	def u_t(self):
		"""Evolve propergator for given time period"""
		dim = self.dim 
		tim_all = self.tim_all
		ctrl = self.ctrl
		H0 = self.H0
		Hctrl = self.Hctrl

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[0,:,:] = np.eye(dim)

		for tim in xrange(tim_all):
			H = H0 + np.matrix( ctrl[tim] * np.array(Hctrl) )
			u_all[tim+1,:,:] = self.u_next(H, u_all[tim,:,:])

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

	def __init__(self, qh_input, phi_g):
		
		self.error_bd = 10**-4 # error bound of convergence 
		self.qh_in = qh_input   #class QH for all i.c. and EoM
		self.phi_g            #goal quantum states we expect
	
	def u_prev(self,H,u_now):
		"""Derive U at next time step"""
		return np.dot(u_now, expm(-1j*H*self.dt))

	def u_t_back(self):
		"""Evolve propergator backward for given time period"""
		
		dim = self.qh_in.dim 
		tim_all = self.qh_in.tim_all
		ctrl = self.qh_in.ctrl
		H0 = self.qh_in.H0
		Hctrl = self.qh_in.Hctrl

		u_all = np.zeros((tim_all+1,dim,dim),dtype = complex)
		u_all[-1,:,:] = np.eye(dim)

		for tim in xrange(tim_all,0,-1):
			H = H0 + np.matrix( ctrl[tim-1] * np.array(Hctrl) )
			u_all[tim,:,:] = self.u_prev(H, u_all[tim-1,:,:])

		return u_all

	def psi_t(self):
		"""backward state start from time T with goal state"""
		dim = self.qh_in.dim
		tim_all = self.qh_in.tim_all 
		psi_all = np.zeros((tim_all+1,dim,1),dtype = complex)
		psi_all[-1,:,:] = self.phi_g[:]
		u_all = self.u_t_back()

		for tim in xrange(tim_all,0,-1):
			phi_all[tim,:,:] = np.dot(u_all[tim,:,:], phi_all[0,:,:])
		
		return phi_all

	def d_ctrl(self, phi_now, psi_now):
		"""calculate new control/laser variation"""
		return np.real(np.dot(psi_now,np.dot(self.qh_in.ctrl,phi_now)))

	def norm_ctrl(self, *ctrls):
		"""normalize control"""
		ctrl_norm = 0
		for ctrl in ctrls:
			ctrl_norm += ctrl**2

		return ctrl_norm	
	
	def fidelity(phi_T, phi_g):
		"""fidelity of phi at final time T """
		return np.dot(phi_g,phi_T)/np.dot(phi_g,phi_g)

	def run(self):
		
		phi_t  = qh_in.phi_t() 
		tim_all = self.qh_in.tim_all		
		for it in range(iter_time):
			
			psi_t = self.psi_t()
			fi = fidelity(phi_t[-1,:,:], phi_g[:])
			print 'Error:', 1-fi
			
			if 1-fi < self.error_bd:
				break
		
			for tim in range(tim_all):
				dctrl = self.d_ctrl(phi_now, psi_t[tim])
				new_ctrl = ctrl + dctrl 
				nctrl =  self.norm_ctrl(new_ctrl)
				phi_t[]  = self.qh_in.u_next(H,u_now)

		return ctrl 	
		

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



