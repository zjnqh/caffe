
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
def create_labelset(data_num,prototxt,batchsz):
	solver = caffe.SGDSolver(prototxt)
	net=solver.net
	datalayer = net.layers[0]
	labelset=zeros(data_num,dtype=int)
	# cifar_table=dict()
	datalayer.use_data.data[0,0,0,0]=1
	for it in xrange(data_num/batchsz):
		start_index=it* batchsz
		end_index=(it+1)* batchsz 
		datalayer.use_data.data[0,0,0,1:]=arange(start_index,end_index)
		net.forward()
		if end_index>data_num:
			labelset[start_index:data_num]=net.blobs['label'].data[:data_num- start_index]
			break
		else:
			labelset[start_index:end_index]=net.blobs['label'].data
		# dat=net.blobs['data'].data
		if it%50==0:
			print "iteration ",it
	net=solver.test_nets[0]
	batchsz=50
	data_num=10000
	test_labelset=zeros(data_num,dtype=int)
	for it in xrange(data_num/batchsz):
		start_index=it* batchsz
		end_index=(it+1)* batchsz 
		net.forward()
		if end_index>data_num:
			test_labelset[start_index:data_num]=net.blobs['label'].data[:data_num- start_index]
			break
		else:
			test_labelset[start_index:end_index]=net.blobs['label'].data
		# dat=net.blobs['data'].data
		if it%50==0:
			print "iteration ",it
		# solver.step_backward()
		
		# solver.apply_update()
		
		# solver.step_extra()
		# for n in xrange(dat.shape[0]):
		# 	key=str(RANDOM_VEC.dot(dat[n,FEATURE_INDEX]))
		# 	cifar_table[key]=n+ start_index
	pkl.dump((labelset,test_labelset),open('examples/cifar100/labelset.pkl','wb'))
	
	return (labelset,test_labelset)
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
		x=x.reshape(x.shape[0],x.shape[1])
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
	def update_testset(self,x,y=None, yp_cnn=None):
		y=y.astype(int)
		x=x.reshape(x.shape[0],x.shape[1])
		yp_cnn= yp_cnn.reshape(yp_cnn.shape[0],yp_cnn.shape[1])
		end=min(x.shape[0]+self.index2,self.U)
		# print "end,index2",end,self.index2,self.U
		rest=end - self.index2
		self.testset[self.index2:end]=x[:rest]
		self.y[self.index2:end]=y[:rest]
		# print "rest",rest,self.yp_cnn[self.index2:end].shape,yp_cnn[:rest].shape
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
		self.tree= BallTree(self.feature_mat,leaf_size=400)
	def predict(self,thh_list=None,print_=True):
		print "capacity",self.capacity,"capacity2",self.capacity2
		x=self.testset[:self.capacity2]
		U=x.shape[0]
		y=self.y[:self.capacity2]
		yp_cnn=self.yp_cnn[:self.capacity2]
		label=self.label[:self.capacity]
		label_mat= zeros([self.capacity, self.yd])
		label_mat[arange(self.capacity),label]=1.0
		
		sz=200 
		K, K2=20,20  # for K neignrest neigbers
		min_value= 0.00000001
		num_batch=U/sz		
		
		dist,index = self.tree.query(x,k=K)
		row=repeat(arange(U),K)
		affmat = csr_matrix((dist.reshape(-1),
			(row,index.reshape(-1))),shape=(U,self.capacity))
		affmat.data=affmat.data**2
		# sigma=(affmat.mean())/((U+ self.capacity)* K)
		sigma=  affmat.data.mean()
		affmat.data /= sigma

		tree_U=BallTree(x,leaf_size=400)
		dist2,index2 = tree_U.query(x,k=K2)
		row=repeat(arange(U),K2)
		affmat2 = csr_matrix((dist2.reshape(-1),
			(row,index2.reshape(-1))),shape=(U,U))
		affmat2.data=affmat2.data**2
		affmat2.data /= sigma
		for i in xrange(U):
			sigma= affmat[i].data.mean()
			affmat[i].data -= sigma
			affmat2[i].data -= sigma
		affmat.data=exp(- affmat.data)
		affmat2.data=exp(- affmat2.data)
		yp_np=affmat.dot(label_mat)
		tmp=yp_np.sum(1)
		tmp [absolute(tmp)<min_value]+=min_value*2
		yp_np=yp_np/(tmp[:,newaxis])
		
		cp=yp_np.copy()
		# yp_np= y2p
		for iters in xrange(5):
			yp_np=affmat.dot(label_mat)+ affmat2.dot(yp_np)*0.5
			tmp=yp_np.sum(1)
			tmp [absolute(tmp)<min_value]+=min_value*2
			yp_np=yp_np/(tmp[:,newaxis])
			# yp_np+=alpha*(y2p- yp_np)
			print "yp_np change:", norm(yp_np - cp,2)
			cp= yp_np.copy()
		yp=yp_np.argmax(1)
		accuracy=nan_to_num(np.mean(yp==y)*100)
		print "label propagation accuracy ",accuracy, " %"
		sigma=absolute(yp_cnn).mean()/3.0
		y2p=exp(yp_cnn/sigma)
		y2p=y2p/(y2p.sum(1)[:,newaxis])
		alpha=0.2
		yp_np+=alpha*(y2p- yp_np)
		yp=yp_np.argmax(1)
		top1=yp_np.max(1)
		yp_np[arange(self.capacity2),yp]=-exp(20)
		top2=yp_np.max(1)
		confidence=top1
		argsorted=argsort(confidence)
		ind=0
		for thh in thh_list:
			pivot=argsorted[U * thh]
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

class Semi_indexing:
	def __init__(self,if_valid,batchsz,labelled=1,step=1):
		self.arrangeN=if_valid
		self.valid_vec= if_valid==labelled
		self.valid_num=self.valid_vec.sum()
		print "valid_num",self.valid_num
		self.N=if_valid.shape[0]
		self.batchsz=batchsz
		self.start=0
		self.global_start=0
		self.stepsz=step * batchsz
		self.tmp_index=zeros(self.stepsz,dtype=int)
		if self.valid_num>0:
			self.valid_index=argwhere(if_valid ==labelled).T[0]
		# self.valid_index_step=zeros(self.valid_num,dtype=int)
		# self.valid_index_step[:-1]=self.valid_index[1:] -self.valid_index[:-1]
		# self.valid_index_step[-1]=self.valid_index[1] + self.N -self.valid_index[-1]
		# self.valid_index_step = self.valid_index_step - 1

	def step(self,global_step=0):
		if global_step>0:
			self.global_start= (self.start + global_step)% self.N
			self.start = argwhere((self.valid_index- global_start_index)>=0).T[0,0]
			
		end_index=self.start + self.stepsz

		# if end_index>self.valid_num:
		# 	self.tmp_index[:self.valid_num- self.start] = self.valid_index_step[self.start:self.valid_num] 
		# 	self.tmp_index[self.valid_num- self.start:] = self.valid_index_step[:end_index - self.valid_num]
		# else:
		# 	self.tmp_index=self.valid_index_step[self.start:end_index]
		if end_index>self.valid_num:
			self.tmp_index[:self.valid_num- self.start] = self.valid_index[self.start:self.valid_num] 
			self.tmp_index[self.valid_num- self.start:] = self.valid_index[:end_index - self.valid_num]
		else:
			self.tmp_index=self.valid_index[self.start:end_index]
		self.start = end_index % self.valid_num
		self.global_start = self.valid_index[self.start]
		return self.tmp_index
	def fix(self):
		return self.tmp_index

class Indexing:
	def __init__(self,N,batchsz,TOTAL_NUM=50000):
		self.arrangeN=arange(N)+1
		self.N=N
		self.batchsz=batchsz
		self.start=0
		self.stepsz=batchsz
		self.tmp_index=zeros(self.stepsz,dtype=int)
	def step(self,step=1):
		end_index=self.start + self.stepsz * step
		if end_index>self.N:
			self.tmp_index[:self.N- self.start] = self.arrangeN[self.start:self.N] 
			self.tmp_index[self.N- self.start:] = self.arrangeN[:end_index - self.N]
		else:
			self.tmp_index=self.arrangeN[self.start:end_index]
		self.start = end_index % self.N
		return self.tmp_index
	def fix(self):
		return self.tmp_index
class Partial_Indexing:
	def __init__(self,arrangeN,batchsz):
		self.arrangeN=arrangeN
		self.N=arrangeN.shape[0]
		self.batchsz=batchsz
		self.start=0
		self.stepsz=batchsz
		self.tmp_index=zeros(self.stepsz,dtype=int)
	def step(self,step=1):
		end_index=self.start + self.stepsz * step
		if end_index>self.N:
			self.tmp_index[:self.N- self.start] = self.arrangeN[self.start:self.N] 
			self.tmp_index[self.N- self.start:] = self.arrangeN[:end_index - self.N]
		else:
			self.tmp_index=self.arrangeN[self.start:end_index]
		self.start = end_index % self.N
		return self.tmp_index
	def fix(self):
		return self.tmp_index

if __name__ == '__main__':
	# # os.chdir('/home/kashefy/src/caffe/')
	# os.chdir('/home/jianqiao/Caffe/caffe-master/')
	
	caffe.set_mode_gpu()
	caffe.set_device(0) # for gpu mode
	# caffe.set_mode_cpu()
	TOTAL_NUM=50000
	prototxt='examples/cifar100/lenet_cifar100_solver.prototxt'
	# (labelset,cifar_table )=create_labelset(TOTAL_NUM, prototxt)
	# (labelset,cifar_table )=pkl.load(open('examples/cifar100/labelset.pkl','rb'))
	labelset=list()
	f=open('label_file.txt','rb')
	labelset=f.readline().split()
	print len(labelset)
	labelset= array(labelset).astype(int)
	filt_hist = []
	yd=61
	label_ratio=0.1
	rnd.seed(100)
	use_data=rnd.rand(TOTAL_NUM)<label_ratio
	unlabeled=logical_not(use_data)
	UNLABEL_NUM=use_data.shape[0]-use_data.sum()
	print "python use data :",use_data.sum()
	# generate_use_data(use_data)
	# import subprocess
	# print "start"
	# # subprocess.call("./create_cifar_subset.sh")
	# print "end"
	# labeled_num=use_data.sum()
	# unlabel_glb=use_data==0
	test_batchsz=50
	batchsz=100
	niter = 2
	test_interval = 20
	train_loss = zeros(niter)
	test_acc = zeros(int(np.ceil(niter / test_interval))+1)
	# label=zeros(batchsz,dtype=int)
	prototxt='examples/cifar_gnet/prototxt/quick_solver1.prototxt'
	solverstate='models/bvlc_googlenet/caffe_alexnet_train_iter_1500.solverstate'
	solverstate='models/bvlc_googlenet/solver_state/alexnet_semi_train_iter_2000.solverstate'
	# (labelset,test_labelset )=create_labelset(TOTAL_NUM, prototxt,batchsz)
	(labelset2,test_labelset)=pkl.load(open('examples/cifar100/labelset.pkl','rb'))
	print "labelset",labelset.shape,labelset[-50:]
	# print "labelset2",labelset2.shape,labelset2[-50:]

	solver = caffe.SGDSolver(prototxt)
	net=solver.net
	testnet=solver.test_nets[0]
	# caffemodel='models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	# # caffemodel='models/bvlc_googlenet/solver_state/alex_cifar_sub_train_iter_1000.caffemodel'
	# # solver.restore(solverstate)
	# # solver.net.copy_from(caffemodel)
	add_data = zeros(TOTAL_NUM)>1.0
	false_label=zeros(TOTAL_NUM,dtype=int)
	index1,index2=Indexing(TOTAL_NUM,batchsz),Indexing(TOTAL_NUM,batchsz)
	# index2=Indexing(TOTAL_NUM,batchsz,sub_niter_semi)
	index4=Indexing(10000,test_batchsz)
	caffemodel_2='models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	caffemodel_2='/data1/jianqiao/cifar100_224/solver_state/googlenet_quick3_iter_8000.caffemodel'
	net.copy_from(caffemodel_2)
	testnet.copy_from(caffemodel_2)
	# knn=KNN(sub_niter_super*50,1024,yd)
	# thh_list=[0.5,0.6,0.7,0.8,0.9]
	datalayer = net.layers[0]
	test_datalayer = testnet.layers[0]
	infer_iters,sub_niter_super,test_iters=4,40,2
	score_blob=net.blobs['loss3/classifier']
	test_score_blob=testnet.blobs['loss3/classifier']
	label_blob=net.blobs['label']
	test_label_blob=testnet.blobs['label']
	conf_vec,y_vec=zeros(batchsz * infer_iters),zeros(batchsz * infer_iters,dtype=int)
	yp_vec=y_vec.copy()
	labelset,test_labelset=labelset%1000,test_labelset%1000
	infer_label=labelset.copy()
	infer_label[unlabeled]=-1
	score_layer=list()
	# use_U=unlabeled.copy()
	use_U=zeros(TOTAL_NUM,dtype=bool)
	snapshot=20
	snapshot_prefix= "/data1/jianqiao/cifar100_224/solver_state/googlenet_semi1_iter_"
	# index3=Partial_Indexing(arange(TOTAL_NUM)[logical_or(use_U,use_data)],batchsz)
	index3=Indexing(TOTAL_NUM,batchsz)
	for i in xrange(0):
		print "index:",index3.step()
	for i in xrange(3):
		layer_name='loss'+str(i+1)+'/loss'
		if i==2:layer_name='loss'+str(i+1)+'/loss3'
		score_layerid=list(net._layer_names).index(layer_name)
		score_layer.append(net.layers[score_layerid])
	for it in xrange(0,niter):
		print "\n","*"*40
		print "network 1"
		if it>=0:
			datalayer.use_data.data[0,0,0,0]=1
			# for i in xrange(3):
			# 	score_layer[i].use_data.data[0,0,0,0]=0
			for sit in xrange(infer_iters):
				if sit%50==0:print "sit:",sit
				tmp_index=index1.step()
				datalayer.use_data.data[0,0,0,1:1+batchsz]=tmp_index
				datalayer.use_data.data[0,0,0,1+batchsz:]=labelset[tmp_index]
				# print "input:",labelset[tmp_index]
				# print "key:",index1.fix()
				net.forward()
				# solver.step_forward()
				# print "True: ",labelset[tmp_index]
				# print "output:",label_blob.data.astype(int)%1000
				score=score_blob.data
				yp=score.argmax(1)
				score_cp=score.copy()
				score_cp[arange(batchsz),yp]=-100000.0
				confidence=score.max(1)- score_cp.max(1)
				# mean_to_sub+= score.mean()
				# mean_to_div2+= median(absolute(score.max(1)))

				y=label_blob.data.astype(int)
				conf_vec[sit* batchsz:(sit+1)*batchsz]=confidence
				yp_vec[sit* batchsz:(sit+1)*batchsz]=yp
				y_vec[sit* batchsz:(sit+1)*batchsz]=label_blob.data.astype(int)%1000
				# print "yp:",yp
				# print "y :",label_blob.data.astype(int)
				# print "python",labelset[tmp_index]

				# solver.step_backward()
				# solver.apply_update()
				# solver.step_extra()
				# accurate=(y==yp)
				# accu_num=accurate.sum()
			print "accuracy:",(y_vec==yp_vec).mean()*100.0
			select_ratio=0.7
			sort_index=conf_vec.argsort()
			tmp_index=index2.step(infer_iters)
			print "misalignment:",(labelset[tmp_index]!=y_vec).mean()*100.0
			margin=conf_vec[sort_index[int(select_ratio* batchsz* infer_iters)]]
			selected=logical_and(conf_vec>=margin,unlabeled[tmp_index])
			use_U[tmp_index[selected]]=True
			infer_label[tmp_index[selected]]=yp_vec[selected]
			use_U[tmp_index[logical_not(selected)]]=False
		# print "use_U:",use_U[:50]
		# print "use_data",use_data[:50]
		# print "logical_or:",logical_or(use_U,use_data)[:50]
		# print "arange:",arange(TOTAL_NUM)[logical_or(use_U,use_data)][:50]
		# index3=Partial_Indexing(arange(TOTAL_NUM)[logical_or(use_U,use_data)],batchsz)
		datalayer.use_data.data[0,0,0,0]=1
		datalayer.use_data.data[0,0,0,0]=1
		tmp_label=zeros(1+batchsz,dtype=int)
		tmp_label[0]=1
		print "\n","*"*40
		print "network 2"
		# for i in xrange(3):
		# 	score_layer[i].use_data.data[0,0,0,0]=1
		for sit in xrange(sub_niter_super):
			datalayer.use_data.data[0,0,0,0]=1
			if sit%50==0:print "sit:",sit
			tmp_index=index3.step()
			# print "tmp_index:",tmp_index
			# tmp_label[1:]=labelset[tmp_index]
			# print "tmp_index:",tmp_index
			# print "infer_label ",infer_label[tmp_index]
			# print "True: ",labelset[tmp_index]
			# print "input:",label_blob.data
			# for i in xrange(3):
			# 	score_layer[i].use_data.data[1:,0,0,0]=infer_label[tmp_index]
			datalayer.use_data.data[0,0,0,1:1+batchsz]=tmp_index
			datalayer.use_data.data[0,0,0,1+batchsz:]=labelset[tmp_index]
			# print "key:",tmp_index
			solver.step_forward()
			# print "input:",label_blob.data.astype(int)%1000
			if (sit+1)%10==0:
				print "accuracy:",(score_blob.data.argmax(1)==label_blob.data).mean()*100.0
				print "misalignment:",(labelset[tmp_index]!=label_blob.data).mean()*100.0
			solver.step_backward()
			solver.apply_update()
			solver.step_extra()

			# print "labelset", labelset[tmp_index]
		
		# (yp,confident)=knn.predict(thh_list=thh_list)
		# sub_niter_semi = int(yp[confident].shape[0]/ batchsz)
		# step_over -= sub_niter_semi * batchsz
		print "\n","*"*40
		print "network 3"
		# if it % test_interval == 0:
		if it>=0:
			total_iter=(it+1)*sub_niter_super
			if total_iter>snapshot:
				snapshot_it= int(total_iter/snapshot)*snapshot
				snapshot_file=snapshot_prefix+str(snapshot_it)+".caffemodel"
				testnet.copy_from(snapshot_file)
				print "copy_from(snapshot_file)"
			correct31, correct35 = 0.0,0.0
			test_datalayer.use_data.data[0,0,0,0]=1
			for sit in range(test_iters):
				if sit%100==0:print "sit:",sit
				tmp_index=index4.step()
				test_datalayer.use_data.data[0,0,0,1:1+test_batchsz]=tmp_index
				test_datalayer.use_data.data[0,0,0,1+test_batchsz:]=test_labelset[tmp_index]
				# print "index:",tmp_index
				# print "python:",test_labelset[tmp_index]
				testnet.forward()
				# print "cpp:",test_label_blob.data
				correct31+=testnet.blobs['loss3/top-1'].data
				correct35+=testnet.blobs['loss3/top-5'].data
			test_acc[it // test_interval] = float(correct31) / (test_iters)
			ac1,ac5=float(correct31) / (test_iters),float(correct35) / (test_iters)
			print 'Iteration', total_iter, 'testing accuracy:',ac1,ac5
			# filt_hist.append(solver.test_nets[0].params['conv1'][0].data)

	# print labelset[:50]+1000
	# print "saveint",saveint,"  ",saveint2	
	# print "python label \n"
	# print labelset+ 1000
	# print "confmat,",confmat
	# f=open("semi_cifar_googlenet.txt","ab")
	# f.write(str('-'*80)+"\n")
	# f.write("iteration "+str(niter)+"\n")
	# f.write("train accuracy "+str(test_acc)+"\n")
	# f.write("loss "+str(train_loss[::20])+"\n \n")
	# f.close()
	# f=open("alexnet_cifar100_confmat.txt","ab")
	# f.write(str('-'*80)+"\n")
	# for i in xrange(confmat.shape[0]):
	# 	f.write(confmat[i])
	# 	f.write('\n')
	# f.close()
	# fig = figure(1)
		
	# _, ax1 = subplots()
	# ax2 = ax1.twinx()
	# ax1.plot(arange(niter), train_loss)
	# ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
	# ax1.set_xlabel('iteration')
	# ax1.set_ylabel('train loss')
	# ax2.set_ylabel('test accuracy')

	# # ax3= ax1.twinx()
	# # ax3.plot(arange(niter), train_loss)
	
	# show()



	# select_ratio = 0.9- 0.2*float(it)/niter
		# if total_iter >= caffemodel_iter* ss_freq and it >0:
		# 	datalayer.use_data.data[0,0,0,0]=0
		# 	caffemodel_2='models/bvlc_googlenet/solver_state2/googlenet_semi_iter_'+str(caffemodel_iter*ss_freq)+'.caffemodel'
		# 	if it<2:
		# 		caffemodel_2='models/bvlc_googlenet/solver_state2/googlenet_quick3_iter_8000.caffemodel'
		# 	net.copy_from(caffemodel_2)
		# 	feature=zeros([50,2048])
		# 	knn.clear()
		# 	for sit in xrange(sub_niter_super):
		# 		net.forward()
		# 		# feature[:,:1024] = net.blobs['pool5/7x7_s1'].data.reshape(50,1024)
		# 		# feature[:,1024:] = net.blobs['loss2/fc'].data
		# 		# knn.update(net.blobs['pool5/7x7_s1'].data,net.blobs['label'].data)
		# 		# knn.update(feature, testnet2.blobs['label'].data)
		# 		# if sit%10==0 and sit>0:
		# 		# 	print "inner iteration:", sit
		# 		# 	print testnet2.blobs['pool5/7x7_s1'].data[0:20]
		# 		if sit%100==0:print "iteration ",sit
		# 	print "labeled set feature calculated!"
		# 	knn.make_tree()
		# 	print "make tree done!"
		# 	caffemodel_iter +=1
		# knn.make_testset(unlabel_iter*50)
		# index3.start_index=index1.start_index
		# correct31, correct35, correct21, correct25 = 0.0,0.0, 0.0, 0.0




		# tmp = arange(tmp_index[0],tmp_index[1])
			# print "python labelset:",labelset[tmp_index]
			# print "python step over:",tmp_index[-1]- tmp_index[0]
			# if tmp_index[0]>=TOTAL_NUM and datalayer.use_data.data[0,0,0,4]>=TOTAL_NUM:
			# 	tmp_index= tmp_index% TOTAL_NUM
			# 	datalayer.use_data.data[0,0,0,4]-=TOTAL_NUM
			# datalayer.use_data.data[0,0,0,5:]=tmp_index
			# net2.forward()
			# tmpIdx[sit * batchsz:(sit+1)*batchsz]=datalayer2.use_data.data[0,0,0,5:]
			# save_label[sit * batchsz:(sit+1)*batchsz]=net2.blobs['label'].data
			# print "tmp_index ",datalayer2.use_data.data[0,0,0,4:].astype(int)
			# # feature[:,:1024] = net2.blobs['pool5/7x7_s1'].data.reshape(100,1024)
			# # feature[:,1024:] = net2.blobs['loss2/fc'].data
			# # knn.update_testset(net2.blobs['pool5/7x7_s1'].data,
			# # 	net2.blobs['label'].data,net2.blobs['loss3/classifier'].data)
			# # knn.update_testset(net2.blobs['pool5/7x7_s1'].data[unlabeled],
			# # 	net2.blobs['fc8'].data[unlabeled], net2.blobs['label'].data[unlabeled])
			# correct31+=net2.blobs['loss3/top-1'].data
			# correct35+=net2.blobs['loss3/top-5'].data
			# correct21+=net2.blobs['loss2/top-1'].data
			# correct25+=net2.blobs['loss2/top-5'].data
			# step_over += datalayer.use_data.data[0,0,0,2]
			# step_vec = (datalayer.use_data.data[0,0,0,5:batchsz+5]).astype(int)
			# print "labelset ",labelset[step_vec][:5]
			# print " "
			# print "net2 ",net2.blobs['label'].data
			# if it==0 and sit==0:
			# 	datalayer.use_data.data[0,0,0,4]=batchsz
		# print "unlabeled set feature calculated!"
		# print 'Iteration', (it+1)*sub_niter_semi, 'testing accuracy:',correct31/sub_niter_semi,correct35/sub_niter_semi
		# print 'inception 2 testing accuracy:',correct21/sub_niter_semi,correct25/sub_niter_semi
		# # pkl.dump(knn,open('knn.pkl','wb'))