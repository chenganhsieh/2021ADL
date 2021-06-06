import os
import torch
import pdb
import argparse
import json
import numpy as np
import re
from datasets import load_dataset
from multiprocessing import Pool
from torch.utils.data import Dataset
from tqdm import tqdm


class TextGenerationDataset(Dataset):
	def __init__(self, args, tokenizer,mode='train'):
		self.args = args
		self.mode = mode
		self.tokenizer = tokenizer
		self.max_seq_length = self.args.max_seq_length
		self.cached_features_file = os.path.join(
			args.cache_dir,
			"cached_{}_{}_{}".format(
				mode,
				list(filter(None, args.model_name_or_path.split("/"))).pop(),
				str(args.max_seq_length),
			),
		)
		self.cached_sample_file = os.path.join(
			args.cache_dir,
			"cached_sampel_{}_{}_{}".format(
				mode,
				list(filter(None, args.model_name_or_path.split("/"))).pop(),
				str(args.max_seq_length),
			),
		)
		

	def get_dataset(self):
		if os.path.exists(self.cached_features_file):
		    print(f"Loading features from cached file {self.cached_features_file}")
		    self.dataset = torch.load(self.cached_features_file)
		else:
			self.load_file()
			self.process_dataset()
			
		# torch.save(self.dataset, self.cached_features_file)
		amount_of_data = len(self.dataset[self.mode])
		print(f"{self.mode} dataset size: {amount_of_data}")
		
		if self.mode == "validation":
			self.sample_dataset =  {"sample":self.dataset["validation"].train_test_split(test_size=0.1)["test"]}
			sample_mount_of_data = len(self.sample_dataset["sample"])
			print(f"Sample dataset size: {sample_mount_of_data}")
			return self.dataset,self.sample_dataset
		# if self.mode == "eval":
		# 	self.dataset =  {"eval":self.dataset["eval"].train_test_split(test_size=0.01)["test"]}


		return self.dataset

	def load_file(self):
		files = {}
		sample_files = {}
		if self.mode == "train":
			files[self.mode] = self.args.data_dir
		elif self.mode == "validation":
			files[self.mode] = self.args.val_data_dir
		else:
			files[self.mode] = self.args.eval_data_dir

		self.dataset = load_dataset('json', data_files=files)


	def process_dataset(self):
		# dataset 包含的key: answer/ input_ids / token_type_ids attention_mask / article_ids
		self.dataset = self.dataset.map(
			self.preprocess_function,
			batched=True,
			num_proc=self.args.num_workers, #self.args.num_workers
		)
		torch.save(self.dataset, self.cached_features_file)
	
	def preprocess_function(self,examples):
		for idx,context in enumerate(examples["maintext"]):
			context = re.sub(r'延伸閱讀.+', r'', context) # 疊字全部刪除 -> (.) 任意字 \1第一個字 + 有重複 變成 \1 第一個字
			context = re.sub(r'《原文刊登於.+', r'', context)
			context = re.sub(r'來源：.+', r'', context)
			context = re.sub(r'來源：文．', r'', context)
			context = re.sub(r'文．', r'', context)
			context = re.sub(r'看這裡>>.+', r'', context)
			context = re.sub(r'看這裡>>.+', r'', context)
			context = re.sub(r'本文摘自.+', r'', context)
			context = re.sub(r'【(.*?)】', r'', context)
			context = re.sub(r'\((.*?)\)', r'', context)
			context = re.sub(r'\n', r'', context)
			context = re.sub(r'http\S+', r'', context)
			context = re.sub(r'圖／.+提供', r'', context)
			context = re.sub(r'圖／翻攝自臉書社團',r'',context)
			context = re.sub(r'（(.*?)）',r'',context)
			context = re.sub(r'《本文作者.+',r'',context)
			context = re.sub(r'地址：.+',r'',context)
			examples["maintext"][idx] = context
			

		


		context_result = self.tokenizer(
			text=examples["maintext"],
			max_length=self.args.max_seq_length,
			padding="max_length",
			truncation=True
		)
		if not self.args.predict:
			# for idx,context in enumerate(examples["title"]):
			# 	examples["title"][idx] = "<pad> " +context
			title_result = self.tokenizer(
				text=examples["title"],
				max_length=self.args.max_summar_length,
				padding="max_length",
				truncation=True
			)
			for idx,title_ids in enumerate(title_result['input_ids']):
				if title_ids[0] == 259:
					title_ids[0] = 0
				else:
					title_ids = title_ids[:-1]
					title_ids = [0] + title_ids
				title_result['input_ids'][idx] = title_ids

			y = np.array(title_result['input_ids'])
			# y = y[:,1:-1] # avoid string 
			attention_mask = np.array(title_result['attention_mask'])[:,:-1]
			target_id = y[:, :-1].tolist() # add start token
			target_label = y[:, 1:]
			target_label[y[:, 1:] == self.tokenizer.pad_token_id] = -100
			context_result["decoder_input_ids"] = target_id
			context_result["decoder_attention_mask"] = attention_mask
			context_result["labels"] = target_label.tolist()
		else:
			context_result["id"] = [int(str_id) for str_id in examples["id"]]

		return context_result
	

