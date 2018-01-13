import numpy as np
from test1 import optimize
from multiprocessing import Process
from mnist import mnist
train1= np.array(mnist.train_images)
class node:
	def __init__(self,learner=1000,vector=np.empty(0),problist= np.empty(0),input=np.empty(0),igain=1000):
		self.learner= learner
		self.input= input
		self.problist=problist
		self.igain= igain
		self.vector= vector

class tree:
	def __init__(self,depth):
		self.depth= depth
		self.array= [node() for i in range(2**depth-1)]

	def train(self,input):
		i=0
		self.array[0].input= input
		self.array[0].igain=-100
		m=2**self.depth-1
		while 2*i+2<m:
			if self.array[i].igain<0:
				vector1,learner1,output1,output2,problist1,problist2,igain1,igain2=optimize(self.array[i].input, self.array[i].igain)
				self.array[i].learner= learner1
				self.array[i].vector= vector1
				self.array[2*i+1]= node(problist= problist1, input= output1, igain= igain1)
				self.array[2*i+2]= node(problist= problist2, input= output2, igain= igain2)

			i+=1

	def test(self, input):
		i=0
		while i<len(self.array):
			Node=self.array[i]
			if np.array_equal(Node.vector,np.empty(0)):
				return np.array(self.array[i].problist)
			else:
				if np.dot(input,Node.vector)>Node.learner:
					i= 2*i+1
				else:
					i= 2*i+2

Tree= [tree(5) for i in range(2)]

# if __name__ == "__main__":
for i in Tree:
	i.train(range(20000))
	# p.start()



mnist.load_testing()
for i in range(10000):
	input=mnist.test_images[i]
	prob=np.zeros(10)
	for j in Tree:
		prob= np.add(prob,j.test(input))
	
	a=np.argmax(prob)
	# print(a== mnist.test_labels[i])
		