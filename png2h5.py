import os
import cv2
import h5py
import numpy as np



file_dir = './image_02/data/'
depth_dir = './proj_depth/groundtruth/image_02/'



### png 2 h5

def data(dir_name):
    data = []
  
    for root,dirs,files in os.walk(dir_name):
        for f in files:
            if os.path.splitext(f)[-1] == '.png':
                path = os.path.join(root,f)           
                img=cv2.imread(path)
                img = np.transpose(img, (2,0,1)) #(c,w,h)
                data.append(img)
    return data

train = data(file_dir)
print(len(train))
train_depth=data(depth_dir)
print(len(train_depth))

#val = data(file_dir)
#val_depth=depth(depth_dir)

if len(train) != len(train_depth):
    try:
        raise ValueError("len(train) != len(train_depth")
    except ValueError :
        print("the number between data and label is not equal")

    # shuffle
    #temp = np.array([data, label])
    #temp = temp.transpose()
    #np.random.shuffle(temp)

   



#cmd = ("mkdir {}").format('./image_02_h5')
#os.system(cmd)
# added 'image_02_h5' folder by hand

for i,j,k in zip(range(len(train)),train,train_depth):
    f = h5py.File('./image_02_h5/{}.h5'.format(i),'w')
    f['rgb'] = j
    #f['val']=j
    f['depth'] = k
    f.close()
            
            

#print(train_s.shape)
#print(depth_s.shape)

#plt.imshow(train_s[333])
#print(train_s[333])
