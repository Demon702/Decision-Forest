from mnist import mnist
import random
import numpy as np
import math
mnist.load_training()
train= np.array(mnist.train_images)
label= np.array(mnist.train_labels)
a= np.zeros(10)
# for i in range(60000):
# 	a[mnist.train_labels[i-1]]+=1


def randomgenerator(m):
	d= np.random.randint(-100,100,size=m)
	return d/math.sqrt(np.dot(d,d))

# def optimize(input):


def optimize(input,igain3):
	
	mingain=-10000
	
	d1=np.zeros(1)
	d2= np.zeros(1)
	output1= np.zeros(1)
	output2= np.zeros(1)
	maxim=-100000000
	minim=10000000
	igain1act=0
	igain2act=0
	while True:
		randvec= randomgenerator(784)
	
		for i in input:
			l= np.dot(randvec, train[i])
			maxim= max(l,maxim)
			minim= min(l,minim) 
		if maxim!=minim:
			break
	
	

	for k in range(49):
		igain1,igain2=[0,0]
		point= minim+ k*(maxim-minim)/50
		
		b1= np.empty(0,dtype="int32")
		b2= np.empty(0, dtype="int32")
# print(len(randomgenerator(784)))
		for j in input:
			a1= train[j]
	
	
			if (np.dot(a1,randvec)-point) > 0:	
				b1= np.append(b1,j)
		
			else:
				b2= np.append(b2,j)
		if len(b1)==0 or len(b2)==0:
			continue
		
		c1= np.zeros(10,dtype="float64")
		for i in b1:
			c1[label[i]]+=1
		c2= np.zeros(10,dtype="float64")
		for i in b2:
			c2[label[i]]+=1

		c1= np.divide(c1,len(b1))
		c2= np.divide(c2,len(b2))
		
		size= len(b1)+ len(b2)
		for i in range(10):
			m1,m2=[c1[i],c2[i]]
			if m1>0:
				igain1+= m1* math.log(m1)
			if m2>0:
				igain2+= m2* math.log(m2)
		igain=igain3
		igain= igain-(len(b1)/size* igain1+ len(b2)/size*igain2)
		
		if -igain>mingain:
			mingain=-igain
			d1=c1
			d2=c2
			output1=b1
			output2=b2
			igain1act=igain1
			igain2act=igain2
			point1=point

	if mingain<0.09:
		igain1,igain2=[2,2]
	
	return randvec,point1,output1,output2,d1,d2,igain1act,igain2act




