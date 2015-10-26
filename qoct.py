import numpy as np
# Editor Bin H.
# Quantum Optimal Control Example
# Two level system

class ini_data:

	def __init__(self):
		
		self.H0 
		self.Hctr
		self.ctrl_i
		self.phi_i
		self.phi_g



class qoct:

	def __init__(self, iniData, errorbd = 10**-4):
		
		self.H0 = iniData.H0
		self.Hctr = iniData.Hctr
		self.ctrl_i = iniData.ctrl_i
		self.phi_i = iniData.phi_i
		self.phi_g = iniData.phi_g
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

	
		
