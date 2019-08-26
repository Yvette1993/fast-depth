import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

import models
from metrics import AverageMeter, Result
import utils

import visdom
viz = visdom.Visdom(env='demo')

args = utils.parse_command()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # Set the GPU.

fieldnames = ['rmse', 'mae', 'delta1', 'absrel',
            'lg10', 'mse', 'delta2', 'delta3', 'data_time', 'gpu_time']
best_fieldnames = ['best_epoch'] + fieldnames
best_result = Result()
best_result.set_to_worst()

def main():
    global args, best_result, output_directory, train_csv, test_csv

    # Data loading code
    print("=> creating data loaders...")
    valdir = os.path.join('..', 'data', args.data, 'val')

    if args.data == 'nyudepthv2':
        from dataloaders.nyu import NYUDataset
        val_dataset = NYUDataset(valdir, split='val', modality=args.modality)
    else:
        raise RuntimeError('Dataset not found.')

    # set batch size to be 1 for validation
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
    print("=> data loaders created.")

    
    assert os.path.isfile(args.evaluate), \
    "=> no model found at '{}'".format(args.evaluate)
    print("=> loading model '{}'".format(args.evaluate))
    checkpoint = torch.load(args.evaluate)
    if type(checkpoint) is dict:
        args.start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        model = checkpoint['model']
        print("=> loaded best model (epoch {})".format(checkpoint['epoch']))
    else:
        model = checkpoint
        args.start_epoch = 0
    output_directory = os.path.dirname(args.evaluate)
    demo(val_loader, model, args.start_epoch, write_to_file=False)
    return


def demo(val_loader, model, epoch, write_to_file=True):
    average_meter = AverageMeter()
    model.eval() # switch to evaluate mode
    end = time.time()

    viz.line([[0.,0.]],[0],win='demo2',opts=dict(title='resluts2',legend=['RMSE','MAE']))
    viz.line([[0.,0.,0.,0.]],[0],win='demo1',opts=dict(title='resluts1',legend=['t_GPU','Delta1','REL','lG10']))

    for i, (input, target) in enumerate(val_loader):
        input, target = input.cuda(), target.cuda()
        # torch.cuda.synchronize()
        data_time = time.time() - end

        # compute output
        end = time.time()
        with torch.no_grad():
            pred = model(input)
        # torch.cuda.synchronize()
        gpu_time = time.time() - end

        # measure accuracy and record loss
        result = Result()
        result.evaluate(pred.data, target.data)
        average_meter.update(result, gpu_time, data_time, input.size(0))
        end = time.time()

        
        # step 50 images for visualization
        skip = 10

        if args.modality == 'rgb':
            rgb = input

        img_merge = utils.merge_into_row(rgb, target, pred)
        #row = utils.merge_into_row(rgb, target, pred)
        #img_merge = utils.add_row(img_merge, row)
        #filename = output_directory + '/comparison_' + str(epoch) + '.png'
        #utils.save_image(img_merge, filename)
        #print(img_merge)
        if i%skip ==0:
            img = np.transpose(img_merge, (2,0,1))
            img = torch.from_numpy(img)
            viz.images(img, win='depth_estimation')
        
            
            
            t_GPU =float( '{gpu_time:.3f}'.format(gpu_time=gpu_time))
            RMSE = float('{result.rmse:.2f}'.format(result=result))
            MAE = float('{result.mae:.2f}'.format(result=result)) 
            Delta1=float('{result.delta1:.3f}'.format(result=result))
            REL=float('{result.absrel:.3f}'.format(result=result))
            Lg10=float('{result.lg10:.3f}'.format(result=result))
        
            #print(t_GPU)
            viz.line([[RMSE,MAE]],[i],win='demo2',update='append')
            viz.line([[t_GPU,Delta1,REL,Lg10]],[i],win='demo1',update='append')
            time.sleep(0.2)   
        
        

    avg = average_meter.average()

    viz.text('\n*\n'
        'RMSE={result.rmse:.2f}({average.rmse:.3f})\n\t'
        'MAE={result.mae:.2f}({average.mae:.3f})\n\t'
        'Delta1={result.delta1:.3f}({average.delta1:.3f})\n\t'
        'REL={result.absrel:.3f}({average.absrel:.3f})\n\t'
        'Lg10={result.lg10:.3f}({average.lg10:.3f})\n\t'
        't_GPU={gpu_time:.3f}{time:.3f}\n'.format(
        gpu_time=gpu_time,average=avg, time=avg.gpu_time,result=result))

    if write_to_file:
        with open(test_csv, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'mse': avg.mse, 'rmse': avg.rmse, 'absrel': avg.absrel, 'lg10': avg.lg10,
                'mae': avg.mae, 'delta1': avg.delta1, 'delta2': avg.delta2, 'delta3': avg.delta3,
                'data_time': avg.data_time, 'gpu_time': avg.gpu_time})
    return avg, img_merge

if __name__ == '__main__':
    main()
