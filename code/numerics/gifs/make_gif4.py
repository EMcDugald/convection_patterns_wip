import os
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics/')
import convection_patterns as cp

# the gif from below run is in my icloud drive
# cp.solveSH(120*np.pi,30*np.pi,512,128,.005,10,"SH1124_3",Rscale=.5,beta=.45,amplitude=.1,init_flag=3,energy=True)
# data = sio.loadmat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SH1124_3.mat")

cp.solveSH(120*np.pi,30*np.pi,512,128,.05,15,"SH1128",Rscale=.5,beta=.45,amplitude=.1,init_flag=3,energy=True)
data = sio.loadmat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SH1128.mat")

sh_png_dir = "/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics/gifs/sh_gif_4"
for idx in range(np.shape(data['uu'])[2]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.imshow(data['uu'][256:,:,idx].T,cmap='gray')
    plt.savefig(sh_png_dir+"/sh_"+f"{idx:0>4}"+".png", bbox_inches='tight')
    plt.close()

images = []
for file_name in sorted(os.listdir(sh_png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(sh_png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(sh_png_dir+'/gif1.gif', images,fps=15)

for f in os.listdir(sh_png_dir):
    if f.startswith('sh'):
        os.remove(os.path.join(sh_png_dir,f))

sh_energy_png_dir = "/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics/gifs/sh_energy_gif_4"
for idx in range(np.shape(data['ee'])[2]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,6))
    ax.imshow(data['ee'][256:,:,idx].T,cmap='gray')
    plt.savefig(sh_energy_png_dir+"/sh_energy_"+f"{idx:0>4}"+".png", bbox_inches='tight')
    plt.close()

images = []
for file_name in sorted(os.listdir(sh_energy_png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(sh_energy_png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(sh_energy_png_dir+'/gif1.gif', images,fps=15)

for f in os.listdir(sh_energy_png_dir):
    if f.startswith('sh'):
        os.remove(os.path.join(sh_energy_png_dir,f))