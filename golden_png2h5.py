import os
#import png2h5
import cv2
import h5py
import numpy as np



def data(path,dir_name):
    data = []
  
    for root,dirs,files in os.walk(os.path.join(path,dir_name)):
        for f in files:
            #print(f)
            if os.path.splitext(f)[-1] == '.png':
                path = os.path.join(root,f)           
                img1 =cv2.imread(path)   ##(w,h,c)
                img = np.transpose(img1, (2,0,1)) #(c,w,h)
                data.append(img)
                
    return data




def png2h5(path,file_dir,depth_dir):
    rgb = data(path,file_dir)
    print(len(rgb))
    depth=data(path,depth_dir)
    print(len(depth))


    if len(rgb) != len(depth):
        try:
            raise ValueError("len(train) != len(train_depth")
        except ValueError :
            print("the number between data and label is not equal")
            
    h5_name = 'image_02_h5'
    h5_dir = os.path.join(path,h5_name)
    print(h5_dir)
    #cmd = ("rm -rf {}".format(os.path.join(path,'image_02_h5')))
    cmd =("mkdir {}".format(h5_dir))
    
    os.system(cmd)

    for i,j,k in zip(range(len(rgb)), rgb, depth):
        f = h5py.File('{}/{}.h5'.format(h5_dir,i),'w')
        f['rgb'] = j
        f['depth'] = k
        f.close()


def golden(dir):
    cmd = ("cd {}".format(dir))
    os.system(cmd)
    for i in os.listdir(dir):
        path = os.path.join(dir,i)
        print(path)
        cmd = ("cd {}".format(path))
        os.system(cmd)
        file_dir = './image_02/data/'
        depth_dir = './proj_depth/groundtruth/image_02/'
        png2h5(path,file_dir,depth_dir)

if __name__ =='__main__':
    train_dir = '/home/lisa/data/kitti/train/'
    val_dir = '/home/lisa/data/kitti/val/'

    

    train = golden(train_dir)
    val = golden(val_dir) 
             
