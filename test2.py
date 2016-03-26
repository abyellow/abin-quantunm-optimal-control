import numpy as np

class SOMECLASS:

	def __init__(self, x0, y0):
		
		self.x = x0
		self.y = y0

	def sum(self):
		return self.x + self.y

	def product(self):
		return self.x * self.y


if __name__ == '__main__':

	x0 = 1
	y0 = 2
	some = SOMECLASS(x0,y0)

	print some.sum()
	print some.product()
