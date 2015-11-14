'''
Created on Jul semi_layer, 2015

@author: kashefy
'''
import os
import sys
import caffe
# import cv2 as cv2
# import cv2.cv as cv
import cPickle as pkl
from pylab import *
import numpy.random as rnd
from caffe import layers as L
from caffe import params as P
from scipy.linalg import *
from sklearn.neighbors import *
from scipy.sparse import *
rnd.seed(100)
RANDOM_VEC=rnd.randn(5)
FEATURE_INDEX=array([30,60,90,120,300])
def create_labelset(data_num,prototxt,label_ratio=0.1):
	solver = caffe.SGDSolver(prototxt)
	net=solver.net
	labelset=zeros(data_num,dtype=int)
	cifar_table=dict()
	batchsz=500

	for it in xrange(data_num/batchsz):
		start_index=it* batchsz
		end_index=(it+1)* batchsz 
		net.forward()
		if end_index>data_num:
			labelset[start_index:data_num]=net.blobs['label'].data[:data_num- start_index]
			break
		else:
			labelset[start_index:end_index]=net.blobs['label'].data
		dat=net.blobs['data'].data
		if it%10==0:
			print "iteration ",it
		# solver.step_backward()
		
		# solver.apply_update()
		
		# solver.step_extra()
		# for n in xrange(dat.shape[0]):
		# 	key=str(RANDOM_VEC.dot(dat[n,FEATURE_INDEX]))
		# 	cifar_table[key]=n+ start_index
	pkl.dump((labelset,cifar_table),open('examples/cifar100/labelset.pkl','wb'))
	
	return (labelset,cifar_table)
def generate_use_data(use_data):
	f=open('use_data.txt',"wb")
	for i in xrange(use_data.shape[0]):
		if use_data[i]==True:
			f.write('1')
		else:
			f.write('0')
	f.close()
class KNN():
	def __init__(self,N,dim,yd):
		self.N=N
		self.D=dim
		self.yd=yd
		self.feature_mat=zeros([N,dim])
		self.label=zeros(N,dtype=int)
		self.capacity=0
		self.index=0
		self.testset=0
		self.index2=0
		self.capacity2=0
	def clear(self):
		self.feature_mat.fill(0.0)
		self.index=0
		self.capacity=0
		self.label.fill(0)
		
	def make_testset(self,U):
		self.U=U
		self.testset=zeros([U,self.D])
		self.yp_cnn=zeros([U,self.yd])
		self.y=zeros(U,dtype=int)
		self.index2=0
		self.capacity2=0
	def update(self,x,y):
		end=min(x.shape[0]+self.index, self.N)
		rest=end - self.index
		self.feature_mat[self.index:end]=x[:rest]
		self.label[self.index:end]=y.astype(int)[:rest]
		self.index=x.shape[0]+ self.index
		if self.index>self.N:
			self.feature_mat[:self.index - self.N]=x[rest:]
			self.label[:self.index - self.N]=y[rest:]
			self.index=self.index% self.N
		self.capacity=min(self.capacity+x.shape[0],self.N)
	def update_testset(self,x,yp_cnn,y=None):
		y=y.astype(int)
		end=min(x.shape[0]+self.index2,self.U)
		rest=end - self.index2
		self.testset[self.index2:end]=x[:rest]
		self.y[self.index2:end]=y[:rest]
		self.yp_cnn[self.index2:end]=yp_cnn[:rest]
		self.index2 = x.shape[0]+ self.index2
		if self.index2>self.U:
			self.testset[:self.index2 - self.U]=x[rest:]
			self.y[:self.index2 - self.U]=y[rest:]
			self.yp_cnn[:self.index2 - self.U]=yp_cnn[rest:]
			self.index2 = self.index2 % self.U
		self.capacity2=min(self.capacity2+x.shape[0],self.U)
	def make_tree(self):
		self.feature_mat=self.feature_mat[:self.capacity]
		# self.label=self.label_mat[:self.capacity]
		self.tree= BallTree(self.feature_mat,leaf_size=10)
	def predict(self,thh=0.9,print_=True):
		print "capacity",self.capacity,"capacity2",self.capacity2
		x=self.testset[:self.capacity2]
		y=self.y[:self.capacity2]
		yp_cnn=self.yp_cnn[:self.capacity2]
		label=self.label[:self.capacity]
		label_mat= zeros([self.capacity, self.yd])
		label_mat[arange(self.capacity),label]=1.0
		U=x.shape[0]
		sz=200 
		K, K2=30,10  # for K neignrest neigbers
		min_value= 0.00000001
		num_batch=U/sz		
		
		dist,index = self.tree.query(x,k=K)
		row=repeat(arange(U),K)
		affmat = csr_matrix((dist.reshape(-1),
			(row,index.reshape(-1))),shape=(U,self.capacity))
		affmat.data=affmat.data**2
		# sigma=(affmat.mean())/((U+ self.capacity)* K)
		sigma=  affmat.data.mean()
		affmat.data /= sigma*4.0

		# tree_U=BallTree(x,leaf_size=10)
		# dist2,index2 = tree_U.query(x,k=K2)
		# row=repeat(arange(U),K2)
		# affmat2 = csr_matrix((dist2.reshape(-1),
		# 	(row,index2.reshape(-1))),shape=(U,U))
		# affmat2.data=affmat2.data**2
		# affmat2.data /= sigma
		for i in xrange(U):
			sigma= affmat[i].data.mean()
			affmat[i].data -= sigma
			# affmat2[i].data -= sigma
		affmat.data=exp(- affmat.data)
		# affmat2.data=exp(- affmat2.data)
		yp_np=affmat.dot(label_mat)
		tmp=yp_np.sum(1)
		tmp [absolute(tmp)<min_value]+=min_value*2
		yp_np=yp_np/(tmp[:,newaxis])
		cp=yp_np.copy()
		y2p=exp(yp_cnn/5.0)
		y2p=y2p/(y2p.sum(1)[:,newaxis])
		alpha=0.3
		yp_np+=alpha*(y2p- yp_np)
		# for iters in xrange(5):
		# 	yp_np=affmat.dot(label_mat)+ affmat2.dot(yp_np)*0.5
		# 	tmp=yp_np.sum(1)
		# 	tmp [absolute(tmp)<min_value]+=min_value*2
		# 	yp_np=yp_np/(tmp[:,newaxis])
		# 	yp_np+=alpha*(y2p- yp_np)
			# print "yp_np change:", norm(yp_np - cp,2)
			# cp= yp_np.copy()
		yp=yp_np.argmax(1)
		confidence=yp_np.max(1)
		pivot=argsort(confidence)[U * thh]
		ind=confidence>(confidence[pivot])
		accuracy=0.0
		if ind.sum()>0:
				# print "yp shape",yp.shape,y.shape,ind.min(),ind.max()
				accuracy=nan_to_num(np.mean(yp[ind]==y[ind])*100)
		if print_:
			# print "confidence mean",yp_np.mean(),yp_np.max(),yp_np.min()
			print "using ",ind.sum()," unlabeled data "," of accuracy ",accuracy, " %"
		tree_U=0
		# self.tree=0
		return yp,ind
class Indexing:
	def __init__(self,N,batchsz,step=1):
		self.arrangeN=arange(N)
		self.N=N
		self.batchsz=batchsz
		self.start_index=0
		self.stepsz=step * batchsz
		self.tmp_index=zeros(self.stepsz,dtype=int)
	def step(self):
		end_index=self.start_index + self.stepsz
		if end_index>self.N:
			self.tmp_index[:self.N- self.start_index] = self.arrangeN[self.start_index:self.N] 
			self.tmp_index[self.N- self.start_index:] = self.arrangeN[:end_index - self.N]
		else:
			self.tmp_index=self.arrangeN[self.start_index:end_index]
		self.start_index = end_index % self.N
		return self.tmp_index
	def fix(self):
		return self.tmp_index
if __name__ == '__main__':
	# # os.chdir('/home/kashefy/src/caffe/')
	# os.chdir('/home/jianqiao/Caffe/caffe-master/')
	
	# test_batchsz=1000
	batchsz=125
	caffe.set_mode_gpu()
	caffe.set_device(1) # for gpu mode
	# caffe.set_mode_cpu()
	TOTAL_NUM=50000
	prototxt='examples/cifar100/lenet_cifar100_solver.prototxt'
	# (labelset,cifar_table )=create_labelset(TOTAL_NUM, prototxt)
	(labelset,cifar_table )=pkl.load(open('examples/cifar100/labelset.pkl','rb'))
	labelset= labelset.astype(int)
	filt_hist = []
	unlabeled=rnd.rand(batchsz)>0
	yd=100
	label_ratio=0.1
	rnd.seed(100)
	use_data=rnd.rand(TOTAL_NUM)<label_ratio
	print "python use data :",use_data.sum()
	generate_use_data(use_data)
	import subprocess
	print "start"
	# subprocess.call("./create_cifar_subset.sh")
	print "end"
	labeled_num=use_data.sum()
	unlabel_glb=use_data==0
	knn=KNN(5000,640,yd)
	niter = 201
	semi_start=500
	semi_=1
	test_interval = 4
	train_loss = zeros(niter)
	test_acc = zeros(int(np.ceil(niter / test_interval))+1)
	label=zeros(batchsz,dtype=int)
	# prototxt='models/bvlc_alexnet/solver.prototxt'
	prototxt='models/bvlc_alexnet/gpu2_full_solver.prototxt'
	solverstate='models/bvlc_alexnet/caffe_alexnet_train_iter_1500.solverstate'
	solverstate='models/bvlc_alexnet/solver_state/alexnet_semi_train_iter_2000.solverstate'
	solver = caffe.SGDSolver(prototxt)
	# caffemodel='bvlc_alexnet.caffemodel'
	caffemodel='models/bvlc_alexnet/solver_state/alex_cifar_sub_train_iter_1000.caffemodel'
	# solver.restore(solverstate)
	solver.net.copy_from(caffemodel)
	net=solver.net
	semi_layer=23
	for layer_id in [1,5,9,11,13]:
		net.layers[layer_id].use_data.reshape(batchsz,1,1,1)
	net.layers[semi_layer].use_data.reshape(1,1,1,batchsz)
	print list(net._layer_names).index('loss')
	print list(net._layer_names)
	sub_niter_semi,sub_niter_super=50,100
	add_data = zeros(TOTAL_NUM)>1.0
	false_label=zeros(TOTAL_NUM,dtype=int)
	index1,index3=Indexing(TOTAL_NUM,batchsz,1),Indexing(TOTAL_NUM,batchsz,1)
	index2=Indexing(TOTAL_NUM,batchsz,sub_niter_semi)

	prototxt2='models/bvlc_alexnet/net2_full_solver.prototxt'
	solver2=caffe.SGDSolver(prototxt2)
	net2= solver2.net
	testnet2=solver2.test_nets[0]
	caffemodel_2='models/bvlc_alexnet/solver_state/alex_cifar_sub_train_iter_1000.caffemodel'
	net2.copy_from(caffemodel_2)
	testnet2.copy_from(caffemodel_2)
	knn.make_testset(30*125)

	for it in xrange(0,niter):
		if it %8==0:
			caffemodel_2='models/bvlc_alexnet/solver_state2/alexnet_semi_train_iter_'+str(it*sub_niter_semi)+'.caffemodel'
			if it<20:
				caffemodel_2='models/bvlc_alexnet/solver_state2/alex_cifar_sub_train_iter_1000.caffemodel'
			# caffemodel_2='models/bvlc_alexnet/solver_state2/alexnet_full_train_iter_2000.caffemodel'
			net2.copy_from(caffemodel_2)
			testnet2.copy_from(caffemodel_2)
			knn.clear()
			for sit in xrange(sub_niter_super):
				testnet2.forward()
				knn.update(testnet2.blobs['fc7'].data,testnet2.blobs['label'].data)
			print "labeled set feature calculated!"
			knn.make_tree()
			print "make tree done!"
		
		knn.make_testset(sub_niter_semi*125)
		# index3.start_index=index1.start_index
		for sit in xrange(sub_niter_semi):
			unlabeled=unlabel_glb[index3.step()]
			net2.forward()
			knn.update_testset(net2.blobs['fc7'].data[unlabeled],
				net2.blobs['fc8'].data[unlabeled], net2.blobs['label'].data[unlabeled])
		print "unlabeled set feature calculated!"
		# pkl.dump(knn,open('knn.pkl','wb'))
		thh=0.85
		if it>30:thh=0.75
		if it>50:thh=0.7
		if it>100:thh=0.65
		(yp,confident)=knn.predict(thh=thh)
		tmp_index=index2.step()
		unlabeled= unlabel_glb[tmp_index]
		tmp_index = tmp_index[unlabeled]
		false_label[tmp_index] =yp
		add_data[tmp_index] =confident
		acc_num=0
		add_num=0
		for sit in xrange(sub_niter_semi):
			solver.step_forward()
			unlabeled=unlabel_glb[index1.step()]
			new_add_data=logical_and(unlabeled,add_data[index1.fix()])
			add_num += new_add_data.sum()
			acc_num += (net.blobs['label'].data[new_add_data]==false_label[index1.fix()][new_add_data]).sum()
			net.blobs['label'].data[unlabeled]=0.0
			net.blobs['label'].data[new_add_data]=false_label[index1.fix()][new_add_data]
			unlabeled[new_add_data]=False
			labeled=logical_not(unlabeled)
			net.layers[semi_layer].use_data.data[0,0,labeled]=1.0
			net.layers[semi_layer].use_data.data[0,0,unlabeled]=0.0
			for layer_id in [1,5,9,11,13]:
				net.layers[layer_id].use_data.data[labeled]=1.0
				net.layers[layer_id].use_data.data[unlabeled]=0.0
			# 	net.layers[layer_id].use_data.data.fill(1)
			# net.layers[semi_layer].use_data.data.fill(1)
			solver.step_backward()
			solver.apply_update()
			solver.step_extra()
			train_loss[it] += solver.net.blobs['loss'].data
			# if sit%10==0:
			# 	print net.params['conv1'][0].data[0,0,0,:5]
			# 	print net.layers[1].blobs[0].diff.sum()
		# print "unlabeled ",(float(acc_num)/add_num)*100," %"
		if it % test_interval == 0:
			correct = 0.0
			test_iters=200 # should be 200
			for test_it in range(test_iters):
				solver.test_nets[0].forward()
				correct+=solver.test_nets[0].blobs['accuracy'].data
			test_acc[it // test_interval] = float(correct) / (test_iters)
			print 'Iteration', it*sub_niter_semi, 'testing accuracy:',float(correct) / (test_iters)
			# filt_hist.append(solver.test_nets[0].params['conv1'][0].data)
		
	# print "confmat,",confmat
	# f=open("alexnet_cifar100_result2.txt","ab")
	# f.write(str('-'*80)+"\n")
	# f.write("iteration "+str(niter)+"\n")
	# f.write("train accuracy "+str(test_acc)+"\n")
	# f.write("loss "+str(train_loss[::50])+"\n \n")
	# f.close()
	# # f=open("alexnet_cifar100_confmat.txt","ab")
	# # f.write(str('-'*80)+"\n")
	# # for i in xrange(confmat.shape[0]):
	# # 	f.write(confmat[i])
	# # 	f.write('\n')
	# # f.close()
	fig = figure(1)
		
	_, ax1 = subplots()
	ax2 = ax1.twinx()
	ax1.plot(arange(niter), train_loss)
	ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('test accuracy')
	
	show()