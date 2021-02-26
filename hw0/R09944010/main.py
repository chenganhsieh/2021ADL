import torch
import numpy as np
import random
import pandas as pd
import argparse
import torch.nn as nn
import sys
from os import path
from tqdm import trange,tqdm
from preprocess import Preprocess
from model import LinearModel
from dataset import MyDataset,ConcatDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("-d","--data_dir",help="data root directory including train.csv / dev.csv / test.csv",default="data")
parser.add_argument("-p","--preprocess",help="do the preprocess",action="store_true")
parser.add_argument("-t","--train",help="train with preprocess data",action="store_true")
parser.add_argument("-pt","--predict",help="eval with checkpoint",action="store_true")
parser.add_argument("--all",help="run all the tasks",action="store_true")
args = parser.parse_args()
if not args.preprocess and not args.train and not args.predict and not args.all:
	sys.exit("No parameters found, please read README.md first")
if not args.data_dir:
	sys.exit("No file directory given")
if args.predict and not args.all and not args.preprocess and not args.train:
	if (not path.exists("model_positive.pkl")) or (not path.exists("model_negative.pkl")):
		sys.exit("No model checkpoint found...Please run training first")
		

torch.manual_seed(1)
np.random.seed(7)  
num_epochs = 100
freq_threshold = 10
BATCH_SIZE = 64
labels = 2  # label = negative / positive
lr = 0.0001


# read file and create bag of words
print("Reading file...")
filepath = args.data_dir.strip('/')+"/"
preprocess = Preprocess(filepath+'train.csv',filepath+'dev.csv',filepath+'test.csv',freq_threshold,args)

# create dataset and dataloader
if args.train or args.all:
	print('Create dataset and dataloader...')
	dataset_train=ConcatDataset(posdata=preprocess.train_pos_data,negdata = preprocess.train_neg_data,target=preprocess.train_category)
	dataset_val=ConcatDataset(posdata=preprocess.val_pos_data,negdata=preprocess.val_neg_data,target=preprocess.val_category)
	dataset_test=ConcatDataset(posdata=preprocess.test_pos_data,negdata=preprocess.test_neg_data)

	train_loader=DataLoader(dataset=dataset_train,batch_size=BATCH_SIZE,shuffle=True,num_workers=8)
	val_loader=DataLoader(dataset=dataset_val,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)
	test_loader=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)

	# create model and loss function
	print('Create model and loss function...')
	linearposmodel = LinearModel(len(preprocess.train_pos_data[0]),labels) 
	linearnegmodel = LinearModel(len(preprocess.train_neg_data[0]),labels) 
	if torch.cuda.is_available():
		linearposmodel = linearposmodel.cuda()
		linearnegmodel = linearnegmodel.cuda()

	loss_function = nn.CrossEntropyLoss()
	pos_optimizer = torch.optim.Adam(linearposmodel.parameters(), lr=lr)
	neg_optimizer = torch.optim.Adam(linearnegmodel.parameters(), lr=lr)

	# start training
	print('Start training...')
	best_acc = 0
	for i in trange(num_epochs):
		linearposmodel.train()
		linearnegmodel.train()
		train_pos_loss =  0
		train_neg_loss =  0
		train_correct = 0 
		train_acc, val_acc = 0, 0
		totalTrain, totalVal = 0, 0
		m,n=0,0
		for pos_x,neg_x, label in train_loader:
			m += 1
			pos_x = pos_x.cuda()
			neg_x = neg_x.cuda()
			label = label.cuda()
			
			totalTrain += label.size(0)
			pos_output = linearposmodel(pos_x)
			neg_output = linearnegmodel(neg_x)
			combine_output = torch.add(pos_output.cpu().data,neg_output.cpu().data)
			combine_output = torch.div(combine_output,2)
			pos_loss = loss_function(pos_output, label)
			pos_loss.backward()
			neg_loss = loss_function(neg_output, label)
			neg_loss.backward()
			pos_optimizer.step()
			neg_optimizer.step()
			predicted =torch.argmax(combine_output,dim=1)
			train_correct += (predicted==label.cpu().data).sum()
			train_pos_loss+= pos_loss
			train_neg_loss+= neg_loss
			tqdm.write("===Batch:"+str(m)+"Pos Loss:"+str(float(train_pos_loss.data/m))+" ===")
			tqdm.write("===Batch:"+str(m)+"Neg Loss:"+str(float(train_neg_loss.data/m))+" ===")
			tqdm.write("\n")
		
		train_acc=float(train_correct.item()/totalTrain) 
		linearposmodel.eval()
		linearnegmodel.eval()
		val_correct= 0
		with torch.no_grad():
			checklabel = []
			for pos_x,neg_x, label in val_loader:
				n=n+1
				pos_x = pos_x.cuda()
				neg_x = neg_x.cuda()
				label = label.cuda()

				totalVal += label.size(0)
				pos_output = linearposmodel(pos_x)
				neg_output = linearnegmodel(neg_x)
				combine_output = torch.add(pos_output.cpu().data,neg_output.cpu().data)
				combine_output = torch.div(combine_output,2)
				
				predicted =torch.argmax(combine_output,dim=1)
				checklabel+=torch.argmax(combine_output,dim=1).tolist()
				val_correct += (predicted==label.cpu().data).sum()
			# check_sentences = []
			# for i in range(len(checklabel)):
			# 	if checklabel[i]!= preprocess.val_category[i]:
			# 		check_sentences.append(preprocess.val_text_data[i])
			# df = pd.DataFrame({'Text': check_sentences})
			# df.to_csv('check_sentences'+str(train_acc)+'.csv',index=False) 

			val_acc=float(val_correct.item()/totalVal)
			tqdm.write("Train Acc:"+str(train_acc))  
			tqdm.write("Val Acc:"+str(val_acc))
			if best_acc < val_acc:
				torch.save(linearposmodel.state_dict(), "model_positive.pkl")
				torch.save(linearnegmodel.state_dict(), "model_negative.pkl")
				best_acc = val_acc
				outputlabel = []
				for pos_x,neg_x in test_loader:
					pos_x = pos_x.cuda()
					neg_x = neg_x.cuda()
					pos_output = linearposmodel(pos_x)
					neg_output = linearnegmodel(neg_x)
					combine_output = torch.add(pos_output.cpu().data,neg_output.cpu().data)
					combine_output = torch.div(combine_output,2)
					outputlabel+= torch.argmax(combine_output,dim=1).tolist() 
				df = pd.DataFrame({'Id': preprocess.test_id,'Category': outputlabel})
				df.to_csv('answer_'+str(best_acc)+'.csv',index=False)  

if args.predict and not args.preprocess and not args.train and not args.all:
	# create model and loss function
	print('Create model...')
	linearposmodel = LinearModel(len(preprocess.test_pos_data[0]),labels) 
	linearnegmodel = LinearModel(len(preprocess.test_neg_data[0]),labels) 
	linearposmodel.load_state_dict(torch.load("model_positive.pkl"))
	linearnegmodel.load_state_dict(torch.load("model_negative.pkl"))
	if torch.cuda.is_available():
		linearposmodel = linearposmodel.cuda()
		linearnegmodel = linearnegmodel.cuda()
	
	dataset_test=ConcatDataset(posdata=preprocess.test_pos_data,negdata=preprocess.test_neg_data)
	test_loader=DataLoader(dataset=dataset_test,batch_size=BATCH_SIZE,shuffle=False,num_workers=8)
	linearposmodel.eval()
	linearnegmodel.eval()
	outputlabel = []
	for pos_x,neg_x in test_loader:
		pos_x = pos_x.cuda()
		neg_x = neg_x.cuda()
		pos_output = linearposmodel(pos_x)
		neg_output = linearnegmodel(neg_x)
		combine_output = torch.add(pos_output.cpu().data,neg_output.cpu().data)
		combine_output = torch.div(combine_output,2)
		outputlabel+= torch.argmax(combine_output,dim=1).tolist() 
	df = pd.DataFrame({'Id': preprocess.test_id,'Category': outputlabel})
	df.to_csv('answer.csv',index=False)  




















