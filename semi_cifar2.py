
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
	def __init__(self,N,batchsz):
		self.arrangeN=arange(N)
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

if __name__ == '__main__':
	# # os.chdir('/home/kashefy/src/caffe/')
	# os.chdir('/home/jianqiao/Caffe/caffe-master/')
	
	# test_batchsz=1000
	batchsz=25
	test_batchsz=25
	caffe.set_mode_gpu()
	caffe.set_device(1) # for gpu mode
	# caffe.set_mode_cpu()
	TOTAL_NUM=50000
	# prototxt='examples/cifar100/lenet_cifar100_solver.prototxt'
	# (labelset,cifar_table )=create_labelset(TOTAL_NUM, prototxt)
	# (labelset,cifar_table )=pkl.load(open('examples/cifar100/labelset.pkl','rb'))
	labelset=list()
	f=open('label_file.txt','rb')
	labelset=f.readline().split()
	print len(labelset)
	labelset= array(labelset).astype(int)
	filt_hist = []
	unlabeled=rnd.rand(batchsz)>=0
	yd=61
	label_ratio=0.1
	rnd.seed(100)
	use_data=rnd.rand(TOTAL_NUM)<label_ratio
	UNLABEL_NUM=use_data.shape[0]-use_data.sum()
	print "python use data :",use_data.sum()
	generate_use_data(use_data)
	import subprocess
	print "start"
	# subprocess.call("./create_cifar_subset.sh")
	print "end"
	labeled_num=use_data.sum()
	unlabel_glb=use_data==0
	# TOTAL_NUM=50
	# labelset=labelset[:TOTAL_NUM]
	
	niter = 36011
	semi_start=500
	semi_=1
	test_interval = 800
	test_iters=400 # should be 200
	train_loss = zeros(niter)
	test_acc = zeros(int(np.ceil(niter / test_interval))+1)
	label=zeros(batchsz,dtype=int)
	prototxt='examples/cifar224/prototxt/quick_solver12.prototxt'
	solver = caffe.SGDSolver(prototxt)
	net=solver.net
	testnet=solver.test_nets[0]
	caffemodel='examples/cifar224/googlenet_sub_pretrain.caffemodel'
	# caffemodel='models/bvlc_googlenet/bvlc_googlenet.caffemodel'
	# # caffemodel='models/bvlc_googlenet/solver_state/alex_cifar_sub_train_iter_1000.caffemodel'
	# # solver.restore(solverstate)
	print "net.copy_from(caffemodel)"
	net.copy_from(caffemodel)
	testnet.copy_from(caffemodel)
	# print net.blobs.keys()
	# semi_layer=23
	# for layer_id in [1,5,9,11,13]:
	# 	net.layers[layer_id].use_data.reshape(batchsz,1,1,1)
	# net.layers[semi_layer].use_data.reshape(1,1,1,batchsz)
	score_layer=list()
	for i in xrange(3):
		layer_name='loss'+str(i+1)+'/loss'
		if i==2:layer_name='loss'+str(i+1)+'/loss3'
		score_layerid=list(net._layer_names).index(layer_name)
		score_layer.append(net.layers[score_layerid])
	score_blob=net.blobs['loss3/classifier']
	test_score_blob=testnet.blobs['loss3/classifier']
	# print len(list(net._layer_names))
	# print (list(net._layer_names))
	UNLABEL=1000
	unlabel_iter,sub_niter_super=100,100
	add_data = zeros(TOTAL_NUM)>1.0
	false_label=zeros(TOTAL_NUM,dtype=int)
	index1,index3=Indexing(TOTAL_NUM,batchsz),Indexing(TOTAL_NUM,batchsz)
	step_over =0
	caffemodel_iter=1
	total_iter=0
	tmp_index=zeros(unlabel_iter * batchsz,dtype=int)
	sub_niter_super=2
	# tmpIdx=zeros(batchsz*sub_niter_super,int)
	# save_label=zeros(batchsz*sub_niter_super)
	label_blob=net.blobs['label']
	test_label_blob=testnet.blobs['label']
	
	min_thh,max_thh=0.000001,30.0
	conf_info=zeros([6,test_iters])
	acc_mean,acc_dev,wrg_mean,wrg_dev=0.0,0.0,0.0,0.0
	print "shape",score_layer[i].use_data.shape[0]
	# for i in xrange(3):
	# 	score_layer[i].use_data.data[0]=1
	use_data=zeros([batchsz+1,1,1,1])
	use_data[0]=1
	total_infer,total_true,total_proc=1.0,1.0,1.0
	f=open("semi_cifar_googlenet2.txt","ab")
	mean_to_sub,mean_to_div=0.0,3.0
	margin =0.7
	momentum=0.9
	sort_conf_vec=zeros(test_batchsz * test_iters)
	snapshot=test_interval
	snapshot_prefix="../cifar224/solver_state/googlenet_semi12_iter_"
	for it in xrange(0,niter):
		mean_to_sub2,mean_to_div2=0.0,0.1
		
		if it % test_interval == 1 and it>0:
			if it>snapshot:
				snapshot_it= int(it/snapshot)*snapshot
				snapshot_file=snapshot_prefix+str(snapshot_it)+".caffemodel"
				testnet.copy_from(snapshot_file)
				print "net.copy_from(caffemodel)"
			correct31, correct35 = 0.0,0.0
			# sort_conf_vec=zeros(test_batchsz)
			for test_it in xrange(test_iters):
				solver.test_nets[0].forward()
				score=test_score_blob.data
				yp=score.argmax(1)
				score_cp=score.copy()
				score_cp[arange(test_batchsz),yp]=-100000.0
				confidence=score.max(1)- score_cp.max(1)
				# mean_to_sub+= score.mean()
				mean_to_div2+= median(absolute(score.max(1)))
				# score/=mean_to_div
				# score-=score.mean(1)[:,newaxis]
				# # sigma=absolute(score).mean()/3.0
				# # nan_to_num(score)
				# # score[score>max_thh]=max_thh
				# # score[score<-max_thh]=-max_thh
				# exp_score=exp(score)
				# exp_score=nan_to_num(exp_score)
				# exp_score_sum=exp_score.sum(1)
				# exp_score_sum=nan_to_num(exp_score_sum)
				# exp_score_sum[absolute(exp_score_sum)<min_thh]=min_thh
				# score=exp_score/(exp_score_sum[:,newaxis])
				
				sort_conf_vec[test_it* test_batchsz:(test_it+1)*test_batchsz]=confidence
				# confidence=sort(confidence,0)
				# # print confidence[:5]
				# sort_conf_vec+=confidence
				y=test_label_blob.data.astype(int)
				# accurate=(y==yp)
				# accu_num=accurate.sum()
				# if accu_num<=test_batchsz and accu_num>0:
				# 	wrong=confidence[logical_not(accurate)]
				# 	conf_info[0,test_it]=accurate.sum()
				# 	conf_info[1,test_it]=((accurate- accurate.mean())**2).sum()
				# 	conf_info[2,                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ]=accu_num
				# 	conf_info[3,test_it]=wrong.sum()
				# 	conf_info[4,test_it]=((wrong- wrong.mean())**2).sum()
				# 	conf_info[5,test_it]=test_batchsz- accu_num
				correct31+=solver.test_nets[0].blobs['loss3/top-1'].data
				correct35+=solver.test_nets[0].blobs['loss3/top-5'].data
			# select_ratio = 0.9- 0.2*float(it)/niter
			select_ratio=0.01
			margin=sort_conf_vec[int(select_ratio*test_batchsz* test_iters)]
			test_acc[it // test_interval] = (correct31) / (test_iters)
			print 'Iteration', (it+1),' testing accuracy:',correct31/test_iters, correct35/test_iters
			f.write(" iter:"+str(it)+" "+str(correct31/test_iters)+" ")
			# acc_mean= conf_info[0].sum()/(conf_info[2].sum())
			# acc_dev= sqrt(conf_info[1].sum()/(conf_info[2].sum()))
			# wrg_mean= conf_info[3].sum()/(conf_info[5].sum())
			# wrg_dev= sqrt(conf_info[4].sum()/(conf_info[5].sum()))
			# print "conf_info",acc_mean,acc_dev,wrg_mean,wrg_dev
			mean_to_sub = mean_to_sub2/test_iters
			mean_to_div = mean_to_div*momentum+ (1- momentum)* mean_to_div2/test_iters/10.0
			# filt_hist.append(solver.test_nets[0].params['conv1'][0].data)	

		solver.step_forward()

		# margin=(acc_mean * acc_dev + wrg_mean *wrg_dev)/(acc_dev + wrg_dev)
		# margin=(acc_mean * wrg_dev + wrg_mean *acc_dev)/(acc_dev + wrg_dev)
		# margin=0.5
		# margin+= (1 - margin)*0.6
		# print "margin ",margin
		score=score_blob.data
		y=label_blob.data.astype(int)
		yp=score.argmax(1)
		score_cp=score.copy()
		score_cp[arange(test_batchsz),yp]=-100000.0
		confidence=score.max(1)- score_cp.max(1)
		# score-=mean_to_sub
		# sigma=absolute(score).mean()/100.0
		# print "sigma:",sigma
		sigma=1.0
		if it%test_interval==0:
			print "confidence median:",median(score.max(1)),mean_to_div
		# score/=mean_to_div
		# score-=score.mean(1)[:,newaxis]
		# score[score>max_thh]=max_thh
		# score[score<-max_thh]=-max_thh
		# # print "score:",score.mean(),score.max(),score.min()
		# exp_score=exp(score)
		# exp_score=nan_to_num(exp_score)
		# exp_score_sum=exp_score.sum(1)
		# exp_score_sum=nan_to_num(exp_score_sum)
		# exp_score_sum[absolute(exp_score_sum)<min_thh]=min_thh
		# score=exp_score/(exp_score_sum[:,newaxis])
		
		# sort_conf_vec=sort(confidence)
		# print "confidence:",confidence.mean(),confidence.max(),confidence.min()
		# margin = sort_conf_vec[int(0.8* batchsz)]
		order= confidence.argsort()
		picked=confidence[order[int(0.7* batchsz)]]
		# print "sort_conf_vec:",sort_conf_vec[:5]
		# print "confidence:",confidence[:5]
		unlabeled=(y >=UNLABEL)
		total_proc+= unlabeled.sum()
		select= logical_and(confidence> margin,unlabeled)
		select= logical_and(select,confidence > picked)
		# select= ones(batchsz)<0
		use_data[1:,0,0,0]=y
		# use_data[arange(batchsz)[unlabeled]+1,0,0,0]=1
		# print "select num",select.sum()
		# use_data[arange(batchsz)[select]+1,0,0,0]= yp[select]+2
		# use_data[arange(batchsz)[select]+1,0,0,0]= y[select] -1000
		use_data[arange(batchsz)[select]+1,0,0,0]= yp[select]
		for i in xrange(3):
			score_layer[i].use_data.data[...]=use_data
		# print "step_backward()"
		solver.step_backward()
		solver.apply_update()
		solver.step_extra()
		# print "param: ",net.params['conv1/7x7_s2'][0].data[0,0,0,:5]
		if select.sum()>0:
			# print " unlabeled num ", unlabeled.sum()," select num ", select.sum(),
			total_infer+= select.sum()
			total_true+= (yp[select]==(y[select]-UNLABEL)).sum()
			# print "label ",y[unlabeled]
			# print "yp ",yp[unlabeled]
		if it %50 ==0:
			
			print "margin:",margin," infer accuracy:", total_true*100.0/total_infer,"% of",total_infer,"/",total_proc
			f.write(" margin:"+str(margin)+" infer:"+str(total_true*100.0/total_infer)+"% of"+str(total_infer)+"/"+str(total_proc))
			total_infer,total_true,total_proc=1.0,1.0,1.0
		# if unlabeled.sum()>0:
		# 	accuracy=(yp[unlabeled]==(label[unlabeled]-UNLABEL)).mean()*100.0
		# 	print "label ",label[unlabeled]
		# 	print "yp ",yp[unlabeled]
		# 	print "accuracy:", accuracy," % of ",unlabeled.sum()
		
		# if sit%20==0:print net.blobs['label'].data[:10]
		train_loss[it] += solver.net.blobs['loss3/loss3'].data
		# if mean_to_div2!=0.1:
		# 	mean_to_div = mean_to_div2/test_iters
		# if sit%10==0:
		# 	print net.params['conv1/7x7_s2'][0].data[0,0,0,:5]
			# print net.layers[1].blobs[0].diff.sum()
		# print "unlabeled ",(float(acc_num)/add_num)*100," %"
		# print "dsadas"
		
	# print "python label \n"
	# print labelset+ UNLABEL
	# print "confmat,",confmat
	
	f.write(str('-'*80)+"\n")
	f.write("iteration "+str(niter)+"\n")
	f.write("train accuracy "+str(test_acc)+"\n")
	f.write("loss "+str(train_loss[::test_interval])+"\n \n")
	f.close()
	f=open("alexnet_cifar100_confmat.txt","ab")
	f.write(str('-'*80)+"\n")
	# for i in xrange(confmat.shape[0]):
	# 	f.write(confmat[i])
	# 	f.write('\n')
	f.close()
	# fig = figure(1)
		
	_, ax1 = subplots()
	ax2 = ax1.twinx()
	ax1.plot(arange(niter), train_loss)
	ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('test accuracy')
	savefig('foo1.jpg')
	# ax3= ax1.twinx()
	# ax3.plot(arange(niter), train_loss)
	
	# show()