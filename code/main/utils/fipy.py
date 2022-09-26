#!/opt/anaconda3/bin/python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import os
import sys
import utils

from tkinter import filedialog
from matplotlib.colors import ListedColormap

arguments = len(sys.argv) - 1


print('Starting Fipy')

#print('Argument 1 ',sys.argv[1])

#time.sleep(10)


root = tk.Tk()
root.geometry("0x0+0+0")

if(arguments not in [0, 1, 2]):
    print("nunber of arguments is ",arguments," WTF ???")
    exit()

if(arguments == 0):
   # Dialog to select .d file
   root.withdraw()
   fpath = filedialog.askopenfilename(filetypes=[("Data files", "*.d")])

   if not fpath:
      exit()
   namepart = fpath.split('.')[0].split('/')[-1]
   data = utils.readd(fpath)



elif(arguments == 1):
   fpath=sys.argv[1]
   namepart = fpath.split('.')[0].split('/')[-1]
   print('reading',fpath)
   data = utils.readd(fpath)
   print('Done reading')

else:
   fpath1=sys.argv[1]
   fpath2=sys.argv[2]
   fpath = fpath1

   data1 = utils.readd(fpath1)
   data2 = utils.readd(fpath2)

   if(np.shape(data1) != np.shape(data2)):
       print('Data dimensions incompatible')
       exit()

   namepart1 = fpath1.split('.')[0].split('/')[-1]
   namepart2 = fpath2.split('.')[0].split('/')[-1]

   data = data1 - data2
   namepart = namepart1 + " - " + namepart2


print('Calling myplot')

utils.myplotimage(data,1,namepart)


mpl.rcParams["savefig.directory"] = os.path.dirname(os.path.realpath(fpath))
mpl.rcParams["savefig.format"] = "pdf"

#plt.close('all')

root.destroy()

plt.show()

exit()

# while True:
#     if msvcrt.kbhit():
#
#         while True:
#
#            ch = bytes.decode(msvcrt.getch())
#
#            if ((ch =='x') or (ch =='X') or (ch == chr(27))):
#                exit()


