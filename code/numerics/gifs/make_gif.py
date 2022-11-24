import os
import imageio
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

png_dir = "/Users/edwardmcdugald/Research/convection_patterns_wip/code/numerics/gifs/gif_dir_1"
data = sio.loadmat("/Users/edwardmcdugald/Research/convection_patterns_wip/code/data/SHAutoEncode_2.mat")
#for idx in range(np.shape(data['uu'])[2]):
for idx in range(600):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3,3))
    ax.imshow(data['uu'][:,:,idx].T)
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