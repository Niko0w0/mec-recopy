import torch

dir = "models/simple_adversary/2308_241011_1521000/"
# Load the model from .pt file
modela_c1 = torch.load(dir+'a_c_0.pt')
modela_c2 = torch.load(dir+'a_c_1.pt')
modela_c3 = torch.load(dir+'a_c_2.pt')

modela_t1 = torch.load(dir+'a_t_0.pt')
modela_t2 = torch.load(dir+'a_t_1.pt')
modela_t3 = torch.load(dir+'a_t_2.pt')

modelc_c1 = torch.load(dir+'c_c_0.pt')
modelc_c2 = torch.load(dir+'c_c_1.pt')
modelc_c3 = torch.load(dir+'c_c_2.pt')

modelc_t1 = torch.load(dir+'c_t_0.pt')
modelc_t2 = torch.load(dir+'c_t_1.pt')
modelc_t3 = torch.load(dir+'c_t_2.pt')


# Now you can inspect the loaded model
print('*-' * 10 + '*')
print(modela_c1)
print(modela_c2)
print(modela_c3)

print('*-' * 10 + '*')
print(modela_t1)
print(modela_t2)
print(modela_t3)

print('*-' * 10 + '*')
print(modelc_c1)
print(modelc_c2)
print(modelc_c3)

print('*-' * 10 + '*')
print(modelc_t1)
print(modelc_t2)
print(modelc_t3)