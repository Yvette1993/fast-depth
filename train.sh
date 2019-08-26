#sudo python2 main.py --arch MobileNetSkipAdd  --gpu 0  #(--epochs 15 -b 8, --print-freq 15) -> nyudepthv2.samples=0.modality=rgb.arch=MobileNetSkipAdd.criterion=l1.lr=0.01.bs=8.pretrained=True  best_model=epoch 6

#sudo python2 main.py --arch MobileNetSkipAdd -b 4 --epochs 40  --print-freq 100 --gpu 0   #-> nyudepthv2.samples=0.modality=rgb.arch=MobileNetSkipAdd.criterion=l1.lr=0.01.bs=4.pretrained=True  best_model=epoch 3

#sudo python2 main.py --arch MobileNetSkipAdd  --epochs 20  --print-freq 100 --gpu 0  #->nyudepthv2.samples=0.modality=rgb.arch=MobileNetSkipAdd.criterion=l1.lr=0.01.bs=8.pretrained=True    best_model=epoch 18


python2 main.py --arch MobileNetSkipAdd -b 10 --epochs 20  --print-freq 50 --gpu 0
