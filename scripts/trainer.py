import copy
import os
import datetime
from functools import partial

from scripts.functions import *

from torch.utils import data
from torchvision import transforms, utils
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from matplotlib import pyplot as plt
from skimage.exposure import match_histograms #type:ignore
from tqdm import tqdm
import torch.nn.functional as F
import random
from scripts.kernel_degradation import circular_lowpass_kernel,random_mixed_kernels,random_add_gaussian_noise_pt,random_add_poisson_noise_pt
import math
import torch.nn as nn
from scripts.diff_jpeg import DiffJPEG
import cv2
import lpips
from PIL import Image

try: 
    from apex import amp #type:ignore

    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
    
# Apply the degradation kernel to the image without changing its size : Prepare data
def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # Apply the same kernel to all the batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)
    
class USMSharp(nn.Module):
    def __init__(self, radius=10, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=1, threshold=5):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


class DegradationSet(data.Dataset):
    """
    A dataset comprised of crops of a single image or several images.
    """
    def __init__(self, image, opt, dataset_size=500):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
                                  Can be of shape (C,H,W) or (B,C,H,W) in case of several images.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            use_flip (bool):    Wheather to use horizontal flips of the image.
            dataset_size (int): The amount of images in a single epoch of training. For training datasets,
                                this should be a high number to avoid overhead from pytorch_lightning.
        """
        self.dataset_size = dataset_size
        self.img = image

        # self.crop_size = crop_size
        # transform_list = [transforms.RandomHorizontalFlip()] if use_flip else []
        # transform_list += [
        #     transforms.RandomCrop(self.crop_size, pad_if_needed=False, padding_mode='constant'),
        # ]

        # self.transform = transforms.Compose(transform_list)

        # Blur Kernel Settings for the First Degradation
        self.blur_kernel_size=opt['blur_kernel_size']
        self.kernel_list=opt['kernel_list']
        self.kernel_prob=opt['kernel_prob'] # Probability of application for each kernel
        self.blur_sigma=opt['blur_sigma'] # How much blur
        self.sinc_prob=opt['sinc_prob'] # The probability of application for sinc filters

        # Kernel Size Range
        # Range1: 3 to 21 (Only odd values are allowed for kernels)
        self.kernel_range=[x for x in range(3,opt['blur_kernel_size'],2)]
        self.use_sharpener=USMSharp()
        self.jpeger=DiffJPEG(differentiable=False)
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        
      	# Generate Kernels (Used in the first degradation)
        # Choose any random kernel size
        kernel_size=random.choice(self.kernel_range)
        # Either apply sinc kernel or random mixture of kernels
        if np.random.uniform() < self.sinc_prob:
            if kernel_size<13:
                omega_c=np.random.uniform(np.pi/3,np.pi)
            else:
                omega_c=np.random.uniform(np.pi/5,np.pi)
            kernel=circular_lowpass_kernel(omega_c,kernel_size,pad_to=False)
        else:
            kernel=random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma, # X range of sigma
                self.blur_sigma, # Y range of sigma
                [-math.pi,math.pi],
                noise_range=None
            )

        # Pad the kernel to Blur Kernel Size

        # Always an even number
        pad_size=(self.blur_kernel_size-kernel_size)//2
        # Top, Bottom, Left, Right
        kernel=np.pad(kernel,((pad_size,pad_size),(pad_size,pad_size)))

        kernel=torch.FloatTensor(kernel,device='cpu')

        img=transforms.ToTensor()(self.img)
        img_lq=img
    
        img_lq=torch.unsqueeze(img_lq,0)
        img=torch.unsqueeze(img,0)
        img=self.use_sharpener(img)
        img=torch.squeeze(img,0)
        img=img*2-1
        
        return img, self.degradation(img_lq,kernel=kernel)


    
    @torch.no_grad()
    def degradation(self,img,kernel):

        ori_h,ori_w=img.size()[2:4]
        #out=self.use_sharpener(img)
        out=filter2D(img,kernel)    
        # random resize
        # updown_type = random.choices(
        #         ['up', 'down', 'keep'],
        #         [0.2,0.7,0.1],
        #         )[0]
        # if updown_type == 'up':
        #     scale = random.uniform(1, 2)
        # elif updown_type == 'down':
        #     scale = random.uniform(0.5, 1)
        # else:
        #     scale = 1
        # mode = random.choice(['area', 'bilinear', 'bicubic'])
        # out = F.interpolate(out, scale_factor=scale, mode=mode)

        scale = random.uniform(0.8, 1)
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Add Noise : Either Gaussian Noise or Poisson Noise
        gray_noise_prob=0.4
        if random.random()<0.5:
            out=random_add_gaussian_noise_pt(
                out,
                sigma_range=[1,3],
                clip=True,
                rounds=False,
                gray_prob=gray_noise_prob,
            )
        else:
            out=random_add_poisson_noise_pt(
                out,
                scale_range=[0.05, 0.5],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False
            )  

        # list=[70, 95]
         # JPEG Compression
         # .uniform_ fills the tensor with uniform values in the range
         # returns a tensor with size equals to number of images in the batch
        # jpeg_p=out.new_zeros(out.size(0)).uniform_(*list)
        
        # # # Clamp to [0,1], otherwise JPEGer will result in unpleasant artifacts
        # out=torch.clamp(out,0,1) 
        # #Apply the compression
        # out=self.jpeger(out,quality=jpeg_p)

        # mode=random.choice(['area','bilinear','bicubic'])
        # out=F.interpolate(
        #         out,
        #         size=(ori_h,ori_w),
        #         mode=mode,
        # ) 

        mode=random.choice(['area','bilinear','bicubic'])
        out=F.interpolate(
                out,
                size=(ori_h,ori_w),
                mode=mode,
        ) 

        img_lq=torch.clamp((out*255.0).round(),0,255) / 255.
        img_lq=img_lq*2-1
        return torch.squeeze(img_lq,0)

class MultiscaleTrainer(object):

    def __init__(
            self,
            image,
            ms_diffusion_model,
            scale_list,
            *,
            ema_decay=0.995,
            train_batch_size=32,
            train_lr=2e-4,
            train_num_steps=8000,
            gradient_accumulate_every=1,
            fp16=False,
            step_start_ema=200,
            update_ema_every=10,
            save_and_sample_every=2000,
            avg_window=100,
            sched_milestones=None,
            results_folder='./results',
            device=None
    ):
        super().__init__()
        self.device = device
        if sched_milestones is None:
            self.sched_milestones = [10000, 30000, 60000, 80000, 90000]
        else:
            self.sched_milestones = sched_milestones

        self.model = ms_diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.avg_window = avg_window

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.img=image


        opt={}
        opt['blur_kernel_size']=7
        opt['kernel_list']= ['iso', 'aniso']
        opt['kernel_prob']= [0.60, 0.40]
        opt['blur_sigma']= [0.2,0.5]
        opt['sinc_prob']=0.1

        # crop_size = int(min(image.size[-2:]) * 0.95)

        self.load_list=[]
        img=image
        self.scale_list=self.scale_list
        for i in range(5):
            cur_scale=self.scale_list[i]
            cur_size=(int(round(image.size[0]*cur_scale)),int(round(image.size[1]*cur_scale)))
            img=image.resize(cur_size, Image.LANCZOS)
            self.train_dataset = DegradationSet(image=img,opt=opt)
            cycle_dl=cycle(data.DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True))
            self.train_loader = next(cycle_dl)
            self.train_loader = (self.train_loader[0].to(self.device), self.train_loader[1].to(self.device))
            self.load_list.append(self.train_loader)

        self.train_steps_list = self.train_num_steps

        self.opt = Adam(ms_diffusion_model.parameters(), lr=train_lr)

        self.scheduler = MultiStepLR(self.opt, milestones=self.sched_milestones, gamma=0.5)

        self.step = 0
        self.running_loss = []
        self.running_scale = []
        self.avg_t = []


        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')
            
        self.lpips_loss = lpips.LPIPS(net="vgg").to(self.device)
        for params in self.lpips_loss.parameters():
            params.requires_grad_(False)
        self.lpips_loss.eval()


        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'sched': self.scheduler.state_dict(),
            'running_loss': self.running_loss,
            'running_scale': self.running_scale
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
        plt.rcParams['figure.figsize'] = [16, 8]

        plt.plot(self.running_loss)
        plt.grid(True)
        plt.ylim((0, 0.2))
        plt.savefig(str(self.results_folder / 'running_loss'))
        plt.clf()

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=self.device)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
        self.scheduler.load_state_dict(data['sched'])
        self.running_loss = data['running_loss']

    def train(self):
        backwards = partial(loss_backwards, self.fp16)
        loss_avg = 0
        loss_avg_mse = 0
        loss_avg_lpips = 0
        sum=0
        for j in range(len(self.scale_list)):
            print(f"Started Training for Scale Factor {self.scale_list[j]}!")
            data_s=self.load_list[j]
            flag=True
            sum+=self.train_steps_list[j]
            while (flag or self.step % sum !=0):
                flag=False
                for i in range(self.gradient_accumulate_every):
                    loss, _, x0_pred=self.model(data_s)
                    loss["lpips"] = self.lpips_loss(
                            x0_pred.clamp(-1.0, 1.0),
                            data_s[0],
                            ).to(x0_pred.dtype).view(-1)  
                    flag_nan = torch.any(torch.isnan(loss["lpips"]))
                    if flag_nan:
                        loss["lpips"] = torch.nan_to_num(loss["lpips"], nan=0.0)      
                    if flag_nan:
                        loss["loss"] = loss["mse"]
                    else:
                        loss["loss"] = loss["mse"] + loss["lpips"]

                    loss_avg += loss["loss"].mean().item()
                    loss_avg_mse += loss["mse"].mean().item()
                    loss_avg_lpips += loss["lpips"].mean().item()
                    backwards(loss["loss"].mean() / self.gradient_accumulate_every, self.opt)



                if self.step % self.avg_window == 0:
                    print(f'step:{self.step} loss_mse:{loss_avg_mse/self.avg_window} loss_lpips:{loss_avg_lpips/self.avg_window} loss_avg: {loss_avg/self.avg_window}')
                    self.running_loss.append(loss_avg/self.avg_window)
                    loss_avg = 0
                    loss_avg_mse = 0
                    loss_avg_lpips=0
                
                self.opt.step()
                self.opt.zero_grad()

                if self.step % self.update_ema_every == 0:
                    self.step_ema()
                self.scheduler.step()
                self.step += 1
                if self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)
            print(f"Finished Training for Scale Factor {self.scale_list[j]}!")

        print("Training Completed!")


def SR(self, input_folder, input_file, device, wl, hl):
    """
    Gradually super-resolve an image based on given width and height scale lists.

    Args:
        input_folder (str): Path to the folder containing the input image.
        input_file (str): Name of the input image file.
        device (torch.device): The device to perform the computation on.
        wl (list): List of width scale factors for super-resolution.
        hl (list): List of height scale factors for super-resolution.
    """

    # Load image
    img = Image.open(os.path.join(input_folder, input_file)).convert('RGB')
    original_size = img.size  # Save original size (width, height)

    # Create output folder
    final_results_folder = Path(str(self.results_folder / 'SR'))
    final_results_folder.mkdir(parents=True, exist_ok=True)

    # Convert the image to a tensor and normalize to [-1, 1]
    input_img_tensor = (transforms.ToTensor()(img) * 2 - 1).unsqueeze(0).to(device)  # Add batch dimension

    # Loop through each scale
    for w_scale, h_scale in zip(wl, hl):
        new_height = int(original_size[1] * h_scale)
        new_width = int(original_size[0] * w_scale)
        new_size = (new_height, new_width)  # Note: height comes before width

        # Upscale the image to the new size using bicubic interpolation
        input_img_tensor = F.interpolate(input_img_tensor, size=new_size, mode='bicubic')

        # Apply the diffusion model's processing step
        input_img_tensor = self.model.p_sample_loop(input_img_tensor)

        # Save intermediate result
        intermediate_file = os.path.join(final_results_folder, input_file + f'_intermediate_{w_scale}x{h_scale}.png')
        utils.save_image((input_img_tensor + 1) * 0.5, intermediate_file)

    # Final super-resolved image
    final_img = self.model.p_sample_loop(input_img_tensor)  # Final diffusion step

    # Convert from [-1, 1] to [0, 1] and clamp values
    final_img = (final_img + 1) * 0.5
    final_img = torch.clamp(final_img * 255, 0, 255) / 255

    # Save the final super-resolved image
    final_file = os.path.join(final_results_folder, input_file + f'_hr.png')
    utils.save_image(final_img, final_file)




    
        

        





