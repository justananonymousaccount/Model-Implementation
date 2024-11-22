import torch
import numpy as np
import argparse
from scripts.diffusion import create_gaussian_diffusion
from models.nextnet import NextNet
from scripts.trainer import MultiscaleTrainer
from PIL import Image

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--scope", help='choose training scope.', default='scene')
    parser.add_argument("--mode", help='sr')
       
    parser.add_argument("--dataset_folder", help='choose dataset folder.', default='./datasets/scene/')
    parser.add_argument("--image_name", help='choose image name.', default='scene.png')
    parser.add_argument("--results_folder", help='choose results folder.', default='./results/')
    
    parser.add_argument("--timesteps", help='total diffusion timesteps.', default=15, type=int)
    parser.add_argument("--train_batch_size", help='batch size during training.', default=12, type=int)
    parser.add_argument("--grad_accumulate", help='gradient accumulation (bigger batches).', default=1, type=int)
    parser.add_argument("--train_steps_list", help='scale wise training steps', default=[100,100], type=int)
    parser.add_argument("--scale_list", help='scale list for training', default=[0.90,1], type=float)
    parser.add_argument("--save_and_sample_every", help='n. steps for checkpointing model.', default=1000, type=int)
    parser.add_argument("--avg_window", help='window size for averaging loss (visualization only).', default=10, type=int)
    parser.add_argument("--train_lr", help='starting lr.', default=1e-2, type=float)
    parser.add_argument("--sched_k_milestones", nargs="+", help='lr scheduler steps x 1000.',
                        default=[], type=int)
    
    parser.add_argument('--network_depth', type=int, help='Depth of the backbone network (amount of blocks)',default=4)
    parser.add_argument('--network_filters', type=int,
                       help='Amount of filters per convolutional level in the backbone networks',default=8)
    parser.add_argument("--load_milestone", help='load specific milestone.', default=0, type=int)

    parser.add_argument("--device_num", help='use specific cuda device.', default=0, type=int)

    parser.add_argument("--widthl", help='list of sfs for width for training', nargs='*', type=float, default=[2.0])
    parser.add_argument("--heightl", help='list of sfs for height for training', nargs='*', type=float, default=[2.0])

    args = parser.parse_args()

    print('num devices: '+ str(torch.cuda.device_count()))
    device = f"cuda:{args.device_num}"
    sched_milestones = [val for val in args.sched_k_milestones]
    results_folder = args.results_folder + '/' + args.scope

    img = Image.open(args.dataset_folder + args.image_name).convert("RGB")
    
    model = NextNet(in_channels=6, filters_per_layer=args.network_filters, depth=args.network_depth)
    model.to(device)

    ms_diffusion = create_gaussian_diffusion(
        denoise_fn=model,
        steps=args.timesteps,
    ).to(device)



    ScaleTrainer = MultiscaleTrainer(
            img,
            ms_diffusion,
            train_batch_size=args.train_batch_size,
            train_lr=args.train_lr,
            train_steps_list=args.train_steps_list,  # total training steps
            gradient_accumulate_every=args.grad_accumulate,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            fp16=False,  # turn on mixed precision training with apex
            save_and_sample_every=args.save_and_sample_every,
            avg_window=args.avg_window,
            sched_milestones=sched_milestones,
            results_folder=results_folder,
            device=device,
            scale_list=args.scale_list

        )

    if args.load_milestone > 0:
        ScaleTrainer.load(milestone=args.load_milestone)
    if args.mode == 'train':
        ScaleTrainer.train()
        # Sample after training is complete
        ScaleTrainer.SR(input_folder=args.dataset_folder,input_file=args.image_name,device=device,wl=args.widthl,hl=args.heightl)
    elif args.mode=='SR':
        ScaleTrainer.SR(input_folder=args.dataset_folder,input_file=args.image_name,device=device,wl=args.widthl,hl=args.heightl)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()
    quit()
