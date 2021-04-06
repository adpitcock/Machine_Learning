import torch
import torchvision
import numpy as np

########################################
#Tensor Initilaization from Numpy Array#
########################################
#data = [[1,2],[3,4]]
#x_data = torch.tensor(data)
#
#np_array = np.array(data)
#x_np = torch.from_numpy(np_array)
#
#x_ones = torch.ones_like(x_data)
#print(f"Ones Tensor: \n {x_ones} \n")
#x_rand = torch.rand_like(x_data, dtype=torch.float)
#print(f"Random Tensor: \n {x_rand} \n")

########################################
# Tensor Initilaization from Random or #
#           Constant Values            #
########################################
#shape = (5,10,)
#rand_tensor = torch.rand(shape)
#ones_tensor = torch.ones(shape)
#zeros_tensor = torch.zeros(shape)
#print(f"Random Tensor: \n {rand_tensor} \n")
#print(f"Ones Tensor: \n {ones_tensor} \n")
#print(f"Zeros Tensor: \n {zeros_tensor} \n")

########################################
#          Tensor Attributes           #
########################################
#tensor = torch.rand(3,4)
#print(f"Shape of Tensor: {tensor.shape}")
#print(f"Datatype of Tensor: {tensor.dtype}")
#print(f"Device tensor is stored on: {tensor.device}")

########################################
#          Tensor Operations           #
########################################
#if torch.cuda.is_available():
#    tensor = tensor.to('cuda')
#tensor alteration using NumPy splicing    
#tensor = torch.ones(4,4)
#tensor[:2,:2] = 0
#tensor[2:,2:] = 0
#t1 =torch.cat([tensor,tensor,tensor], dim=1)
          #[tensor,tensor,tensor],
          #[tensor,tensor,tensor])
#print(t1)

########################################
#     Tensor Matrix Multiplication     #
########################################
#shape = (4,4,)
#rand_tensor1 = torch.rand(shape)
#rand_tensor2 = torch.rand(shape)
#print(f"Tensor1: \n {rand_tensor1} \n")
#print(f"Tensor2: \n {rand_tensor2} \n")
#print(f"Tensor Multiplication: \n {rand_tensor1.mul(rand_tensor2)} \n")

########################################
#            Torch.AutoGrad            #
########################################

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

prediction = model(data)

loss = (prediction - labels).sum()
loss.backward()

optim = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=.9)

optim.step()