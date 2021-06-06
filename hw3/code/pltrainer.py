import torch
import pytorch_lightning as pl
import pdb
import json
import os
from statistics import mean
from tw_rouge import get_rouge
from torch import nn
from torch.nn import CrossEntropyLoss,NLLLoss,Softmax
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.optim import Adam
from torch.utils.data import DataLoader 

class PLTrainer(pl.LightningModule):
	def __init__(self, args, model,datasets,tokenizer):
		super().__init__()
		self.hparams = args
		self.model = model
		self.loss = NLLLoss(ignore_index=0)
		self.softmax = Softmax(dim=2) # batch / sequences length / 250112(vocab size)
		self.train_dataset = datasets['train']
		self.val_dataset = datasets['val']
		self.sample_dataset = datasets['sample']
		self.test_dataset = datasets['eval']
		self.get_rouge = get_rouge
		self.tokenizer = tokenizer
		self.save_hyperparameters(args)

	def forward(self, batch, batch_idx):
		# in lightning, forward defines the prediction/inference actions
		embedding = self.model(x)
		return embedding
	
	# def configure_optimizers(self):
	# 	optimizer = AdamW(self.parameters(),
	# 					 lr=self.hparams.lr)
	# 	# warmup_steps = self.steps_per_epoch // 3
	# 	# total_steps = self.steps_per_epoch * self.n_epochs - warmup_steps
	# 	# scheduler = get_linear_schedule_with_warmup(optimizer,warmup_steps,total_steps)
	# 	scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
	# 	return {
	# 		'optimizer': optimizer,
	# 		'lr_scheduler': scheduler
	# 	}
	def configure_optimizers(self):
		optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
		scheduler = get_linear_schedule_with_warmup(
			optimizer, num_warmup_steps=0,
			num_training_steps=self.hparams.max_epochs * len(self.train_dataset))
		return {'optimizer': optimizer, 'lr_scheduler': scheduler}
	# # 自己的loss
	def criterion(self, outputs, targets):
		loss = self.loss(outputs, targets)
		return loss
	
	# 一個step
	def training_step(self, batch, batch_idx):
		output = self.model(**batch)
		loss = output[0]
		logits = output[1]
		rl_loss = 0
		
		# RL
		if self.hparams.rl_ratio!= 0 and self.trainer.current_epoch == self.hparams.rl_start_epoch:
			greedy_output = self.model.generate(batch["input_ids"], 
				max_length=self.hparams.max_summar_length
			)
			random_output = self.model.generate(batch["input_ids"], 
				do_sample=True, 
				max_length=self.hparams.max_summar_length, 
				top_k=0
			)
			min_length = min(random_output.shape[1],logits.shape[1])
			# free memory
			del output
			logits = logits[:,:min_length,:]
			random_output = random_output[:,:min_length]
			random_prob = self.softmax(logits).permute(0,2,1)
			random_output_loss = self.loss(random_prob,random_output)
			decode_greedy_words = self.tokenizer.batch_decode(greedy_output,skip_special_tokens=True)
			decode_random_words = self.tokenizer.batch_decode(random_output,skip_special_tokens=True)
			decode_greedy_words = [i if len(i)>0 else "無" for i in decode_greedy_words ]
			decode_random_words = [i if len(i)>0 else "無" for i in decode_random_words ]
			if len(decode_greedy_words) == 0:
				decode_greedy_words = ["無" for i in range(len(decode_ans_words))]
			if len(decode_random_words) == 0:
				decode_random_words = ["無" for i in range(len(decode_ans_words))]
			decode_ans_words = self.tokenizer.batch_decode(batch["decoder_input_ids"],skip_special_tokens=True)
			greedy_scores = get_rouge(decode_greedy_words,decode_ans_words)["rouge-l"]['f']
			random_scores = get_rouge(decode_random_words,decode_ans_words)["rouge-l"]['f']
			neg_reward = greedy_scores - random_scores
			rl_loss = neg_reward * random_output_loss
			loss = (1 - self.hparams.rl_ratio) * loss + self.hparams.rl_ratio * rl_loss
		
		logs = {
			't_loss': loss,
			'rl_loss': rl_loss
		}

		for k, v in logs.items():
			self.log(k, v, on_step=True, on_epoch=False, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx,dataloader_idx):
		# most data cal loss # less data cal score 
		if dataloader_idx == 0:
			output = self.model(**batch) # 1 
			loss = output[0]
			output = {
				'v_loss': loss, 
			}
		else:
			greedy_output = self.model.generate(batch["input_ids"], max_length=self.hparams.max_summar_length)
			
	
			# output = self.model(**batch) # 1 
			# loss = output[0]
			# logits = output[1]
			# words_predicted =torch.argmax(logits.cpu().data,dim=2)
			decode_words = self.tokenizer.batch_decode(greedy_output,skip_special_tokens=True)
			decode_words = [i if len(i)>0 else "無" for i in decode_words]
			decode_ans_words = self.tokenizer.batch_decode(batch["decoder_input_ids"],skip_special_tokens=True)
			scores = get_rouge(decode_words,decode_ans_words)
			# article = self.tokenizer.decode(batch["input_ids"][0],skip_special_tokens=True)
			# score = get_rouge(decode_words[0],decode_ans_words[0])
			# text = []
			# text.append('Sample {} - {}'.format(self.global_rank, batch_idx))
			# text.append('==: {}'.format(article))
			# text.append('<<: {}'.format(decode_ans_words[0]))
			# text.append('>>: {} | rouge1:{} | rouge2:{} | rougel:{}'.format(
			# 	decode_words[0],
			# 	score["rouge-1"]['f'],
			# 	score["rouge-2"]['f'],
			# 	score["rouge-l"]['f'],
			# 	))
			# text.append("")
			output = {
				'rouge1': scores["rouge-1"]['f'],
				'rouge2': scores["rouge-2"]['f'],
				'rougel': scores["rouge-l"]['f'],
				# 'text': text
			}		
		return output
	def validation_epoch_end(self, outputs):
		v_loss = torch.stack([output['v_loss'] for output in outputs[0]]).mean()
		rouge1 = mean([output['rouge1'] for output in outputs[1]])
		rouge2 = mean([output['rouge2'] for output in outputs[1]])
		rougel = mean([output['rougel'] for output in outputs[1]])
		# sample_text = [t for output in outputs[1] for t in output['text']]

		logs = {
			'v_loss': v_loss, 
			'rouge_1': rouge1,
			'rouge_2': rouge2,
			'rouge_l': rougel,
		}


		for k, v in logs.items():
			self.log(k, v, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		# with open(os.path.join(self.hparams.log_dir, 'sample.txt'), 'w') as f:
		# 	f.write(f'Valid Step: {self.global_step}\n')
		# 	f.write('\n'.join(sample_text))
		
	def test_step(self, batch, batch_idx): #定義 Test 階段
		if self.hparams.output_mode == "greedy":
			output = self.model.generate(batch["input_ids"], 
			max_length=self.hparams.max_summar_length)
		elif self.hparams.output_mode == "beam":
			output = self.model.generate(batch["input_ids"],
			max_length=self.hparams.max_summar_length,
			num_beams=self.hparams.nums_beam, 
			no_repeat_ngram_size=2, 
			early_stopping=True
			)
		elif self.hparams.output_mode == "topk":
			output = self.model.generate(batch["input_ids"],
			max_length=self.hparams.max_summar_length,
			do_sample=True,
			top_k=self.hparams.topk,
			temperature = self.hparams.temperature
			)
		elif self.hparams.output_mode == "topp":
			output = self.model.generate(batch["input_ids"],
			max_length=self.hparams.max_summar_length,
			do_sample=True,
			top_p=self.hparams.topp,
			top_k=0,
			temperature = self.hparams.temperature
			)
		elif self.hparams.output_mode =="synthesis":
			output = self.model.generate(batch["input_ids"],
			max_length=self.hparams.max_summar_length,
			do_sample=True,
			top_p=self.hparams.topp,
			top_k=self.hparams.topk,
			temperature = self.hparams.temperature
			)

		# output = output[:,2:]
		decode_words = self.tokenizer.batch_decode(output,skip_special_tokens=True)
		# pdb.set_trace()
		# for idx,word in enumerate(decode_words):
		# 	decode_words[idx] = word.lstrip(",")
		# 	decode_words[idx] = word.lstrip("!")
		# 	decode_words[idx] = word.lstrip("/")
		# 	decode_words[idx] = word.lstrip(":")
		# 	decode_words[idx] = word.lstrip("、")
		# 	decode_words[idx] = word.lstrip("/")
			
		# decode_words 
		# print(decode_words)
		# pdb.set_trace()

		return {'text': decode_words,'ids':batch["id"].tolist()}
	def test_epoch_end(self, outputs):
		with open(self.hparams.output_file, 'w') as f:
			for output in outputs:
				titles = output["text"]
				for idx,data in enumerate(titles):
					temp_data = {"title":data,"id":output["ids"][idx]}
					f.write(json.dumps(temp_data, ensure_ascii=False) + "\n")
	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True,num_workers=self.hparams.num_workers, pin_memory=True)
	
	def val_dataloader(self):
		validation_dataloader = DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
		sample_dataloader = DataLoader(self.sample_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)
		return  [validation_dataloader,sample_dataloader]

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, shuffle=False,num_workers=self.hparams.num_workers, pin_memory=True)