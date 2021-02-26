import pandas as pd
import pickle
import re,string
import multiprocessing
from multiprocessing import Pool
from os import path
pos_finish = 0
neg_finish = 0
finish = 0

class Preprocess():
	def __init__(self,data_dir,val_dir,test_dir,freq_threshold,args):
		self.freq_threshold = freq_threshold
		self.pos_freq_dict = {}
		self.neg_freq_dict = {}
		self.pos_word2idx = {}
		self.neg_word2idx = {}
		if args.predict and not args.all:
			train = pd.read_csv(data_dir)
			test = pd.read_csv(test_dir)
			train_data= train['text'].values.tolist()
			train_label = train['Category'].values.tolist()
			self.cal_posneg_freq(train_data,train_label)
			test_data = test['text'].values.tolist()
			all_data = test_data
			print(f"CPU Count:{multiprocessing.cpu_count()}")
			P = Pool(processes= multiprocessing.cpu_count()-1) 
			pos_data = P.map(self.bag_of_pos_word, all_data)
			print(f"Positive data {len(pos_data)}")
			P.close()
			P.join()
			negP = Pool(processes= multiprocessing.cpu_count()-1) 
			neg_data = negP.map(self.bag_of_neg_word, all_data)
			print(f"Negative data {len(neg_data)}")		
			negP.close()
			negP.join()
			self.test_pos_data = pos_data
			self.test_neg_data = neg_data
			self.test_id = test['Id'].values.tolist()
			return
		elif data_dir!=None and (not path.exists("positive.pkl") or args.preprocess):
			print("Not found preprocessed data... doing preprocessing now....")
			global finish
			train = pd.read_csv(data_dir)
			val = pd.read_csv(val_dir)
			test = pd.read_csv(test_dir)

			print(f"total train data size: {len(train)}")
			print(f"total val data size: {len(val)}")
			print(f"total test data size: {len(test)}")

			train_data= train['text'].values.tolist()
			train_label = train['Category'].values.tolist()
			self.cal_posneg_freq(train_data,train_label)

			val_data = val['text'].values.tolist()
			self.val_text_data = val['text']
			test_data = test['text'].values.tolist()

			all_data = train_data + val_data + test_data
			# multi processing to get bag of words
			print(f"CPU Count:{multiprocessing.cpu_count()}")
			P = Pool(processes= multiprocessing.cpu_count()-1) 
			pos_data = P.map(self.bag_of_pos_word, all_data)
			print(f"Positive data {len(pos_data)}")
			P.close()
			P.join()
			negP = Pool(processes= multiprocessing.cpu_count()-1) 
			neg_data = negP.map(self.bag_of_neg_word, all_data)
			print(f"Negative data {len(neg_data)}")		
			negP.close()
			negP.join()
			self.train_pos_data = pos_data[:len(train)]
			self.val_pos_data = pos_data[len(train):len(train)+len(val)]
			self.test_pos_data = pos_data[len(train)+len(val):]

			self.train_neg_data = neg_data[:len(train)]
			self.val_neg_data = neg_data[len(train):len(train)+len(val)]
			self.test_neg_data = neg_data[len(train)+len(val):]
			with open('positive.pkl', 'wb') as f:
				pickle.dump(pos_data, f)
			with open('negative.pkl', 'wb') as f:
				pickle.dump(neg_data, f)
			self.train_id = train['Id'].values.tolist()
			self.train_category = train['Category'].values.tolist()
			self.val_id = val['Id'].values.tolist()
			self.val_category = val['Category'].values.tolist()
			self.test_id = test['Id'].values.tolist()
			print(f"Positive Vector size:{len(self.train_pos_data[0])}")
			print(f"Negative Vector size:{len(self.train_neg_data[0])}")
		else:
			print("Found preprocessed data...reading now....")
			train = pd.read_csv(data_dir)
			val = pd.read_csv(val_dir)
			test = pd.read_csv(test_dir)
			print(f"total train data size: {len(train)}")
			train_data= train['text']
			train_label = train['Category'].values.tolist()
			self.cal_posneg_freq(train_data,train_label)
			val_data = val['text']
			test_data = test['text']
			self.val_text_data = val['text']

			with open('positive.pkl', 'rb') as f:
				data = pickle.load(f)
				self.train_pos_data = data[:len(train)]
				self.val_pos_data = data[len(train):len(train)+len(val)]
				self.test_pos_data = data[len(train)+len(val):]
			with open('negative.pkl', 'rb') as f:
				data = pickle.load(f)
				self.train_neg_data = data[:len(train)]
				self.val_neg_data = data[len(train):len(train)+len(val)]
				self.test_neg_data = data[len(train)+len(val):]
			print(f"train data {len(self.train_pos_data )}")
			print(f"val data {len(self.val_pos_data )}")
			print(f"test data {len(self.test_pos_data )}")
			self.train_id = train['Id'].values.tolist()
			self.train_category = train['Category'].values.tolist()
			self.val_id = val['Id'].values.tolist()
			self.val_category = val['Category'].values.tolist()
			self.test_id = test['Id'].values.tolist()
	def cal_posneg_freq(self,data,labels):
		m = -1
		for sentence in data:
			m+=1
			sentence=sentence.strip('\n')
			sentence = re.sub('，', '', sentence)
			sentence = re.sub(':', '', sentence)
			sentence = re.sub('。', '', sentence)
			sentence = re.sub('~', '', sentence)
			sentence = re.sub('的', '', sentence)
			sentence = re.sub('和', '', sentence)
			sentence = re.sub('還', '', sentence)
			sentence = re.sub('在', '', sentence)
			sentence = re.sub('還是', '', sentence)
			sentence = re.sub('應該', '', sentence)
			sentence = re.sub('我', '', sentence)
			sentence = re.sub('你', '', sentence)
			sentence = re.sub('他', '', sentence)
			sentence = re.sub('妳', '', sentence)
			sentence = re.sub('><', '', sentence)
			sentence = re.sub('^^', '', sentence)
			sentence = " ".join(sentence.split())
			sentence = sentence.split(" ")
			for word in sentence:
				if labels[m] == 0:
					if word not in self.neg_freq_dict:
						self.neg_freq_dict[word] = 1
					else:
						self.neg_freq_dict[word] += 1
				else:
					if word not in self.pos_freq_dict:
						self.pos_freq_dict[word] = 1
					else:
						self.pos_freq_dict[word] +=1
		# sort freq and filter word by threshold
		self.neg_freq_dict = {k: v for k, v in sorted(self.neg_freq_dict.items(), key=lambda item: item[1]) if v > self.freq_threshold} 
		self.pos_freq_dict = {k: v for k, v in sorted(self.pos_freq_dict.items(), key=lambda item: item[1]) if v > self.freq_threshold}
		self.pos_word2idx = {k: i for i, k in enumerate(list(self.pos_freq_dict.keys())) }
		self.neg_word2idx = {k: i for i, k in enumerate(list(self.neg_freq_dict.keys())) }
		print(f"neg freq dict size:{len(self.neg_freq_dict)}")
		print(f"pos freq dict size:{len(self.pos_freq_dict)}")

	def bag_of_pos_word(self,sentence):
		global pos_finish
		sentence = str(sentence)
		temp_freq_list = [0 for i in range(len(self.pos_freq_dict))]
		words=sentence.split(" ")
		print(pos_finish,end='\r')
		for word in words:
			if word in self.pos_word2idx:
			   temp_freq_list[self.pos_word2idx[word]] +=1
		pos_finish+=1
		return temp_freq_list
	def bag_of_neg_word(self,sentence):
		global neg_finish
		sentence = str(sentence)
		temp_freq_list = [0 for i in range(len(self.neg_freq_dict))]
		words=sentence.split(" ")
		print(neg_finish,end='\r')
		for word in words:
			if word in self.neg_word2idx:
			   temp_freq_list[self.neg_word2idx[word]] +=1
		neg_finish+=1
		return temp_freq_list














	
