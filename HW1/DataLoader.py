import os,json
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tr
from torch.utils import data

main_path = os.getcwd()

class Rain_Loader(data.Dataset):
	
	def __init__(self, data_path, train):
		#cat-> 0 dog -> 1
		self.kind = [0, 1]
        self.files = pd.read_csv(data_path)
        
        if train:
            self.labels = self.files['Attribute23']
            self.files.drop(columns=['Attribute23'], inplace=True)

	def __len__(self):
		return len(self.files)

	def __getitem__(self,index):
		
		file = self.files.iloc[index]
		#img_path,label

		img = Image.open(file['img_path'])  #load img
		img = img.resize(self.crop_size,Image.BICUBIC)#resize
		img = np.asarray(img,np.float32)# to numpy(H x W x C)
		img = img.transpose((2, 0, 1))#swap channel C x H x W

		return img,file['label']


# if __name__ == "__main__":
# 	# Data Loader Unit Test
# 	dst = CatDog_Loader(data_path = 'data',train = True)
# 	train_dataloader = data.DataLoader(dataset = dst, batch_size = 10, shuffle = True)
# 	for i,data in enumerate(train_dataloader,start = 1):
# 		img,label = data
# 		print(img.shape)
# 		print(label)
# 		break
# 	print("Test OK!!")




