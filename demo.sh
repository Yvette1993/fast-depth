
#sudo python demo.py --evaluate /home/lisa/fast-depth/results/mobilenet-nnconv5dw-skipadd.pth.tar

#sudo python demo.py --evaluate /home/lisa/fast-depth/results/nyudepthv2.samples=0.modality=rgb.arch=MobileNetSkipAdd.criterion=l1.lr=0.01.bs=4.pretrained=True/model_best.pth.tar

#sudo python demo.py --evaluate /home/lisa/fast-depth/results/nyudepthv2.samples\=0.modality\=rgb.arch\=MobileNetSkipAdd.criterion\=l1.lr\=0.01.bs\=8.pretrained\=True/model_best.pth.tar

sudo python demo.py --evaluate /home/lisa/fast-depth/results/nyudepthv2.samples=0.modality=rgb.arch=MobileNetSkipAdd.criterion=l1.lr=0.01.bs=8.pretrained=True/model_best.pth.tar

