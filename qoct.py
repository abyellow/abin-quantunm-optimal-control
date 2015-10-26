import numpy as np
# Editor Bin H.
# Quantum Optimal Control Example
# Two level system

class qoct:

	def __init__(self, H0, Hctr, ctrl_i, phi_i, phi_g):
		
		self.H0 = H0
		self.Hctr = Hctr
		self.ctrl_i = ctrl_i
		self.phi_i = phi_i
		self.phi_g = phi_g
		self.errorbd = errorbd
		
	def EoM(self,H,phi_i):	
	
	def EoM_next(H, phi_now):
	
	def revEoM(self, H, phi_g):
	
	def renewCtrl(self, phi_now, psi_now, Hctr):

	def fidelity(phi_i_fi, phi_g):


	def run(self):

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


if __name__ == '__main__':

	
		
