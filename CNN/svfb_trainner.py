import torch
import cnn_arch as cnn
from cnn_arch import ConvNet 
gpu = 0 #which gpu use(if multiple)

if torch.cuda.is_available(): #checks for gpu with cuda on the system
	cnn.device = torch.device('cuda:'+str(gpu))
	print('using gpu!')
else:
	cnn.device = torch.device('cpu')
	print('using cpu!')	

cnn.cnn = ConvNet().to(cnn.device) # create cnn object

cnn.dataset_location = '/home/jpvolt/Documents/projetos/svfb_ml/datasets/frame data'

cnn.modelname = 'trainnedModel'

cnn.num_epochs = 100

cnn.batch_size = 32

cnn.l_r = 0.0015

cnn.shuffle = True

cnn.train()