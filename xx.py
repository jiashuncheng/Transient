import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
N = 0

path = "/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_v14_tansient_True_connection_0.2_region_True_mode_train_delay_random_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209091133_model_params_step1000.pth"
model_param = torch.load(path)
w_out = model_param["w_out"].detach().cpu().numpy()
b_out = model_param["b_out"].detach().cpu().numpy()
w_hidden = model_param["w_rnn"].detach().cpu().numpy()

#ANCHOR - 3s Data
s0m1 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay3_base_random_match_sample0_tansient_True_connection_0.2_region_True_mode_test_delay_3_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209103845_model_1_sample_0_match_1_theta_180_noiseh_False.pt' #mathc=1
s0m0 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay3_base_random_nonmatch_sample0_tansient_True_connection_0.2_region_True_mode_test_delay_3_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209094651_model_1_sample_0_match_0_theta_180_noiseh_False.pt' #mathc=0
s1m1 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay3_base_random_match_sample1_tansient_True_connection_0.2_region_True_mode_test_delay_3_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209104053_model_1_sample_1_match_1_theta_180_noiseh_False.pt' #mathc=1
s1m0 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay3_base_random_nonmatch_sample1_tansient_True_connection_0.2_region_True_mode_test_delay_3_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209104137_model_1_sample_1_match_0_theta_180_noiseh_False.pt' #mathc=0

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,N,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata0:',label, 2)
hidden_act = results.T
# print('output1:', hidden_act[0])
# print('output2:', hidden_act[1])
# print('output3:', hidden_act[2])
data0 = hidden_act
newdata0 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,N,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata1:',label, 1)
hidden_act = results.T
data0 = hidden_act
newdata1 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,N,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata2:',label, 2)
hidden_act = results.T
data0 = hidden_act
newdata2 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,N,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata3:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata3 = data0.copy()


netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,1,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata4:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata4 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,1,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata5:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata5 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,1,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata6:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata6 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,1,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata7:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata7 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,2,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata8:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata8 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,2,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata9:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata9 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,2,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata10:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata10 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,2,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata11:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata11 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,3,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata12:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata12 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,3,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata13:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata13 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,3,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata14:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata14 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,3,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata15:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata15 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,4,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata16:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata16 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,4,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata17:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata17 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,4,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata18:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata18 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,4,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata19:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata19 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,5,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata20:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata20 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,5,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata21:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata21 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,5,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata22:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata22 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,5,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata23:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata23 = data0.copy()

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import Isomap
from dPCA import dPCA

data = np.hstack((newdata0, newdata1, newdata2, newdata3, newdata4, newdata5, newdata6, newdata7, newdata8, newdata9, newdata10, newdata11, newdata12, newdata13, newdata14, newdata15, newdata16, newdata17, newdata18, newdata19, newdata20, newdata21, newdata22, newdata23))
h_data = data.copy()
print(data.shape)


N,T,S,D = 600,90,2,2
noise, n_samples = 0.2, 6
data = np.zeros((n_samples,N,S,D,T))
print(data.shape)
data[0,:,0,0,:] = newdata0
data[0,:,1,0,:] = newdata2
data[0,:,0,1,:] = newdata1
data[0,:,1,1,:] = newdata3
data[1,:,0,0,:] = newdata4
data[1,:,1,0,:] = newdata6
data[1,:,0,1,:] = newdata5
data[1,:,1,1,:] = newdata7
data[2,:,0,0,:] = newdata8
data[2,:,1,0,:] = newdata10
data[2,:,0,1,:] = newdata9
data[2,:,1,1,:] = newdata11
data[3,:,0,0,:] = newdata12
data[3,:,1,0,:] = newdata14
data[3,:,0,1,:] = newdata13
data[3,:,1,1,:] = newdata15
data[4,:,0,0,:] = newdata16
data[4,:,1,0,:] = newdata18
data[4,:,0,1,:] = newdata17
data[4,:,1,1,:] = newdata19
data[5,:,0,0,:] = newdata20
data[5,:,1,0,:] = newdata22
data[5,:,0,1,:] = newdata21
data[5,:,1,1,:] = newdata23

data3 = data.copy()


import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
N = 0

path = "/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_v14_tansient_True_connection_0.2_region_True_mode_train_delay_random_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240209091133_model_params_step1000.pth"
model_param = torch.load(path)
w_out = model_param["w_out"].detach().cpu().numpy()
b_out = model_param["b_out"].detach().cpu().numpy()
w_hidden = model_param["w_rnn"].detach().cpu().numpy()

#ANCHOR - 6s Data
s0m1 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay6_base_random_match_sample0_tansient_True_connection_0.2_region_True_mode_test_delay_6_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240211094403_model_1_sample_0_match_1_theta_180_noiseh_False.pt' #mathc=1
s0m0 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay6_base_random_nonmatch_sample0_tansient_True_connection_0.2_region_True_mode_test_delay_6_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240211094036_model_1_sample_0_match_0_theta_180_noiseh_False.pt' #mathc=0
s1m1 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay6_base_random_match_sample1_tansient_True_connection_0.2_region_True_mode_test_delay_6_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240211094554_model_1_sample_1_match_1_theta_180_noiseh_False.pt' #mathc=1
s1m0 = '/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_output_test_delay6_base_random_nonmatch_sample1_tansient_True_connection_0.2_region_True_mode_test_delay_6_hidd_600_seed_4_lr0.001_vm4.0_vgamma1.0_alpha0.6/20240211094716_model_1_sample_1_match_0_theta_180_noiseh_False.pt' #mathc=0

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,8,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata0:',label, 2)
hidden_act = results.T
# print('output1:', hidden_act[0])
# print('output2:', hidden_act[1])
# print('output3:', hidden_act[2])
data0 = hidden_act
newdata0 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,8,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata1:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata1 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,8,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata2:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata2 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,8,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata3:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata3 = data0.copy()


netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,1,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata4:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata4 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,1,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata5:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata5 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,1,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata6:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata6 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,1,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata7:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata7 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,9,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata8:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata8 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,9,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata9:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata9 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,9,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata10:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata10 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,9,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata11:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata11 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,3,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata12:',label, 2)
hidden_act = results.T
data0 = hidden_act

newdata12 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,3,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata13:',label, 1)
hidden_act = results.T
data0 = hidden_act

newdata13 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,3,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata14:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata14 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,3,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata15:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata15 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,4,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata16:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata16 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,4,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata17:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata17 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,4,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata18:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata18 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,4,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata19:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata19 = data0.copy()

netdir  = s0m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,5,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata20:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata20 = data0.copy()

netdir  = s0m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,5,:]
results[90:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[80:90]
print('newdata21:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata21 = data0.copy()

netdir  = s1m1 #mathc=1
results = torch.load(netdir)['hidden_act'][:,5,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata22:',label,2)
hidden_act = results.T
data0 = hidden_act

newdata22 = data0.copy()

netdir  = s1m0 #mathc=0
results = torch.load(netdir)['hidden_act'][:,5,:]
# results[60:] = 0
temp = np.zeros((90,3))
for i in range(temp.shape[0]):
    temp[i] = np.maximum(0,results[i][np.newaxis, :] @ w_out) + b_out
label = np.argmax(temp, axis=1)#[50:60]
print('newdata23:',label,1)
hidden_act = results.T
data0 = hidden_act

newdata23 = data0.copy()

from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.decomposition import IncrementalPCA
from sklearn.manifold import Isomap
from dPCA import dPCA


N,T,S,D = 600,90,2,2
noise, n_samples = 0.2, 6
data = np.zeros((n_samples,N,S,D,T))
print(data.shape)
data[0,:,0,0,:] = newdata0
data[0,:,1,0,:] = newdata2
data[0,:,0,1,:] = newdata1
data[0,:,1,1,:] = newdata3
data[1,:,0,0,:] = newdata4
data[1,:,1,0,:] = newdata6
data[1,:,0,1,:] = newdata5
data[1,:,1,1,:] = newdata7
data[2,:,0,0,:] = newdata8
data[2,:,1,0,:] = newdata10
data[2,:,0,1,:] = newdata9
data[2,:,1,1,:] = newdata11
data[3,:,0,0,:] = newdata12
data[3,:,1,0,:] = newdata14
data[3,:,0,1,:] = newdata13
data[3,:,1,1,:] = newdata15
data[4,:,0,0,:] = newdata16
data[4,:,1,0,:] = newdata18
data[4,:,0,1,:] = newdata17
data[4,:,1,1,:] = newdata19
data[5,:,0,0,:] = newdata20
data[5,:,1,0,:] = newdata22
data[5,:,0,1,:] = newdata21
data[5,:,1,1,:] = newdata23

data6 = data.copy()