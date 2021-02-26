import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
	def __init__(self,data,target=None):
		super(MyDataset,self).__init__()
		self.data=data
		self.target=target
	def __getitem__(self,index):
		x = self.data[index]
		x = torch.tensor(x).float()
		if self.target != None:
			y=self.target[index]
			y=torch.tensor(y)
			return x,y
		else:
			return x
	def __len__(self):
		return len(self.data)

class ConcatDataset(Dataset):
	def __init__(self,posdata,negdata,target=None):
		super(ConcatDataset,self).__init__()
		self.posdata=posdata
		self.negdata=negdata
		self.target=target
	def __getitem__(self,index):
		pos_x = self.posdata[index]
		pos_x = torch.tensor(pos_x).float()

		neg_x = self.negdata[index]
		neg_x = torch.tensor(neg_x).float()

		if self.target != None:
			y=self.target[index]
			y=torch.tensor(y)
			return pos_x,neg_x,y
		else:
			return pos_x,neg_x
	def __len__(self):
		return len(self.posdata)