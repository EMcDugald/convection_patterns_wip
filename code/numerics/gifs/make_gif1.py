import os
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics')
import convection_patterns as cp

# cp.solveSH(20*np.pi,20*np.pi,256,256,.1,100,"SHAutoEncode_2",Rscale=.5,
#            beta=.45,amplitude=.1,init_flag=1,energy=False)

cp.solveSH(20*np.pi,20*np.pi,256,256,.2,100,"SHgif1",Rscale=.5,beta=.45,amplitude=.1,init_flag=1,energy=True)

png_dir = "/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics/gifs/sh_gif_1"
#data = sio.loadmat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHAutoEncode_2.mat")
data = sio.loadmat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHgif1.mat")
for idx in range(np.shape(data['uu'])[2]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    ax.imshow(data['uu'][:,:,idx].T,cmap='gray')
    plt.savefig(png_dir+"/sh_"+f"{idx:0>4}"+".png", bbox_inches='tight')
    plt.close()

images = []
for file_name in sorted(os.listdir(png_dir)):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave(png_dir+'/gif1.gif', images,fps=25)

for f in os.listdir(png_dir):
    if f.startswith('sh'):
        os.remove(os.path.join(png_dir,f))


sh_energy_png_dir = "/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics/gifs/sh_energy_gif_1"
for idx in range(np.shape(data['ee'])[2]):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    ax.imshow(data['ee'][:,:,idx].T,cmap='gray')
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