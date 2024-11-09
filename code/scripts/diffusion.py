import torch
import torch.nn as nn
import torch.nn.functional as F
import enum
import numpy as np
import math

# Eta Scheduler for Diffusion Model
def get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps,
        min_noise_level,
        etas_end=0.99,
        kappa=1.0,
        power=0.3
    ):
    
    if schedule_name=='exponential':
        # Starting value
        etas_start=min(min_noise_level/kappa,min_noise_level)
        # increaser = (a)**(1/m) where m = num_diffusion_timesteps-1 and m = etas_end/etas_start
        increaser=math.exp(1/(num_diffusion_timesteps-1)*math.log(etas_end/etas_start))
        
        # Getting the array: [increaser,increaser,increaser...] upto number of timesteps
        base=np.ones([num_diffusion_timesteps,])*increaser
        
        # Create an array of timesteps, starting from 0 upto 1 (both included)
        # With elements equal to number of timesteps
        # Raise each value with the power
        power_timestep=np.linspace(0,1,num_diffusion_timesteps,endpoint=True)**power
        # Now multiply each element with the value (num_diffusion_timesteps-1)
        # If we have not raised the power, we would get (0,1,2,...,timesteps-1)
        power_timestep*=(num_diffusion_timesteps-1)

        # We get: (a**(t_/m)) where t_ is the power timestep
        sqrt_etas=np.power(base,power_timestep)*etas_start
    else:
        raise ValueError(f"Unknow schedule_name {schedule_name}")

    return sqrt_etas

# Base Model for Diffusion : Spaced Gaussian Diffusion
def create_gaussian_diffusion(
        denoise_fn,
        schedule_name='exponential',
        min_noise_level=0.01,
        steps=100,
        kappa=1,
        etas_end=0.99,
        weighted_mse=False,
        predict_type='xstart',
    ):
    
    # sqrt(eta_t) = sqrt(eta_1) * (b0 ^ beta_t)
    # b0 ^ beta_t depends on the current timestep and hence act as the scheduler
    # we have calculated its value for every timestep

    sqrt_etas=get_named_eta_schedule(
        schedule_name,
        num_diffusion_timesteps=steps,
        min_noise_level=min_noise_level,
        etas_end=etas_end,
        kappa=kappa,
    )
    
    if predict_type == 'xstart':
        model_mean_type = ModelMeanType.START_X
    else:
        raise ValueError(f'Unknown Predicted type: {predict_type}')
    
    return GaussianDiffusion(
        denoise_fn=denoise_fn,
        sqrt_etas=sqrt_etas,
        kappa=kappa,
        model_mean_type=model_mean_type,
        loss_type=LossType.WEIGHTED_MSE if weighted_mse else LossType.MSE,
        normalize_input=False,
        latent_flag=False
    )


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


# Extract the array elements indexed with timesteps and change the shape as per the broadcast_shape
def _extract_into_tensor(arr,timesteps,broadcast_shape):

    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.

    """   

    res=torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape)<len(broadcast_shape):
        res=res[...,None]
    return res.expand(broadcast_shape)

# Specifies the type of output that the model predicts
class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """
    START_X = enum.auto()       # the model predicts x_0

# Specifies the type of loss used with the model
class LossType(enum.Enum):
    MSE = enum.auto()           # simplied MSE
    WEIGHTED_MSE = enum.auto()  # weighted mse derived from KL


class GaussianDiffusion(nn.Module):
    
    """
    Utilities for training and sampling diffusion models.

    :param sqrt_etas: a 1-D numpy array of etas for each diffusion timestep,
                starting at T and going to 1.
    :param kappa: a scaler controling the variance of the diffusion kernel
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param loss_type: a LossType determining the loss function to use.
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    :param scale_factor: a scaler to scale the latent code
    :param sf: super resolution factor

    """

    def __init__(
            self,
            denoise_fn,
            sqrt_etas,
            kappa,
            model_mean_type,
            loss_type,
            normalize_input=False,
            latent_flag=False,

    ):
        super().__init__()
        self.kappa=kappa
        self.model_mean_type=model_mean_type
        self.loss_type=loss_type
        self.normalize_input=normalize_input
        self.latent_flag=latent_flag

        # Use float64 for accuracy
        self.sqrt_etas=sqrt_etas
        self.etas=sqrt_etas**2
        self.denoise_fn=denoise_fn

        # Check if the scheduler is 1D
        assert len(self.etas.shape) == 1
        assert (self.etas>0).all() and (self.etas<=1).all()

        self.num_timesteps=int(self.etas.shape[0])

        # Append 0.0 at the starting of the self.etas array without the last element
        self.etas_prev=np.append(0.0,self.etas[:-1])
        # Calculating alpha : alpha_t = eta_t -eta_(t-1)
        self.alpha=self.etas-self.etas_prev

        # Calculations for posterior q(x_{t-1} | x_t, x_0) for each timestep

        # Variance Calculations
        self.posterior_variance=kappa**2 * self.etas_prev / self.etas * self.alpha
        
        # Replace 0 variance with variance[1]
        # Clippped due to we are taking log in the next step
        self.posterior_variance_clipped=np.append(
            self.posterior_variance[1],self.posterior_variance[1:]
        )
        self.posterior_log_variance_clipped=np.log(self.posterior_variance_clipped)

        # Mean Calculations
        self.posterior_mean_coef1=self.etas_prev / self.etas
        self.posterior_mean_coef2=self.alpha / self.etas

        # Weight for the MSE Loss
        if model_mean_type in [ModelMeanType.START_X]:
            weight_loss_mse = 0.5 / self.posterior_variance_clipped * (self.alpha / self.etas)**2
            
        else:
            raise NotImplementedError(model_mean_type)
        
        self.weight_loss_mse=weight_loss_mse

    def _scale_input(self, inputs, t):
        if self.normalize_input:
            if self.latent_flag:
                # The variance of latent code is around 1.0
                std = torch.sqrt(_extract_into_tensor(self.etas, t, inputs.shape) * self.kappa**2 + 1)
                inputs_norm = inputs / std
            else:
                inputs_max = _extract_into_tensor(self.sqrt_etas, t, inputs.shape) * self.kappa * 3 + 1
                inputs_norm = inputs / inputs_max
        else:
            inputs_norm = inputs
        return inputs_norm


    # Add noise to the clean HQ image (x_start)
    def q_sample(self,x_start,y,t,noise=None):

        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.

        """        
        # q(xt|x0, y0) = N(xt; x0 + (η_t)*e0, (κ**2)*(η_t)*I)

        if noise is None:
            noise=torch.randn_like(x_start)
        assert noise.shape==x_start.shape

        return (
            _extract_into_tensor(self.etas,t,x_start.shape)*(y-x_start) + x_start
            + _extract_into_tensor(self.sqrt_etas*self.kappa,t,x_start.shape) * noise
        )

    
    # Finding mean and variance to add to the HQ image to get the LQ image
    # y is the x_reconstructed
    # x_start is the original downscaled input
    def q_mean_variance(self, x_start, y, t):

        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.etas, t, x_start.shape) * (y - x_start) + x_start
        variance = _extract_into_tensor(self.etas, t, x_start.shape) * self.kappa**2
        log_variance = variance.log()
        return mean, variance, log_variance


    # Reverse Diffusion Process
    # We do not have x_start (HQ image) but we can have the predicted x_start, predicted HQ image 
    # Return the mean and variance of x_{t-1} : Given the diffused image at timestep t and the predicted x_start (HQ image) 
    def q_posterior_mean_variance(self,x_start,x_t,t):

        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """

        assert x_start.shape==x_t.shape
        posterior_mean=(
            _extract_into_tensor(self.posterior_mean_coef1,t,x_t.shape)*x_t
            + _extract_into_tensor(self.posterior_mean_coef2,t,x_t.shape)*x_start
        )

        posterior_variance=_extract_into_tensor(self.posterior_variance,t,x_t.shape)
        posterior_log_variance_clipped=_extract_into_tensor(
            self.posterior_log_variance_clipped,t,x_t.shape
        )

        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )

        return posterior_mean,posterior_variance,posterior_log_variance_clipped
    
    # Return the mean and variance of x_{t-1} : Given diffused image at timestep t and the LQ image 
    # So, firstly predict the HQ image and then pass it to the q_posterior_mean_variance
    @torch.no_grad()
    def p_mean_variance(
            self,
            x_t,
            y,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None
    ):
        
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x_t: the [N x C x ...] tensor at time t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        if model_kwargs is None:
            model_kwargs={}

        B,C=x_t.shape[:2]
        assert t.shape==(B,)
        
        model_output=self.denoise_fn(self._scale_input(x_t,t),t,y)
        model_variance=_extract_into_tensor(self.posterior_variance,t,x_t.shape)
        model_log_variance=_extract_into_tensor(self.posterior_log_variance_clipped,t,x_t.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x=denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1,1)
            return x

        if self.model_mean_type==ModelMeanType.START_X:        # Predict x_0
            pred_xstart=process_xstart(model_output)

        else:
            raise ValueError(f'Unknown Mean Type: {self.model_mean_type}')
        
        model_mean,_,_=self.q_posterior_mean_variance(
            x_start=pred_xstart,x_t=x_t,t=t
        )

        assert(
            model_mean.shape==model_log_variance.shape==pred_xstart.shape==x_t.shape
        )

        return {
            "mean":model_mean,
            "variance":model_variance,
            "log_variance":model_log_variance,
            "pred_xstart":pred_xstart
        }
    @torch.no_grad() 
    # Return the x_{t-1} (denoised image at timestep t-1) given the LQ image and the correponding denoised image at timestep t
    def p_sample(self,x,y,t,clip_denoised=True,model_kwargs=None,noise_repeat=False,denoised_fn=None):
        
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_t.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """

        # Get the mean and variance of the distribution x_{t-1}
        out=self.p_mean_variance(
            x,
            y,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs
        )

        noise=torch.randn_like(x)
        if noise_repeat:
            noise=noise[0,].repeat(x.shape[0],1,1,1)
        nonzero_mask=(
            (t!=0).float().view(-1,*([1] * (len(x.shape) - 1)))
        ) # No noise when t==0

        # Get the sample x_{t-1} using the mean and variance of the distribution
        sample=out["mean"] + nonzero_mask*torch.exp(0.5*out["log_variance"])*noise
        return {"sample":sample,"pred_xstart":out["pred_xstart"],"mean":out["mean"]}
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        y,
        noise=None,
        noise_repeat=False,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param model: the model module.
   

        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
                
        final=None
        count=0
        for sample in self.p_sample_loop_progressive(
            y,
            noise=noise,
            noise_repeat=noise_repeat,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            count=count+1
            final=sample["sample"]
        
        return final



    @torch.no_grad()
    def p_sample_loop_progressive(
            self,
            y,
            noise=None,
            noise_repeat=False,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().

        """        
        
        if device is None:
            device=next(self.denoise_fn.parameters()).device
        z_y=y

        # Generating Noise
        if noise is None:
            noise=torch.randn_like(z_y)
        if noise_repeat:
            noise=noise[0,].repeat(z_y.shape[0],1,1,1)
        z_sample=self.prior_sample(z_y,noise)

        indices=list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm
            from tqdm.auto import tqdm
            indices=tqdm(indices)

        for i in indices:
            t=torch.tensor([i]*y.shape[0],device=device)
            with torch.no_grad():
                out=self.p_sample(
                    z_sample,
                    z_y,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_repeat=noise_repeat
                )
                yield out
                z_sample=out["sample"]


    def prior_sample(self,y,noise=None):

        """
        Generate samples from the prior distribution, i.e., q(x_T|x_0) ~= N(x_T|y, ~)

        :param y: the [N x C x ...] tensor of degraded inputs.
        :param noise: the [N x C x ...] tensor of degraded inputs.
        """

        if noise is None:
            noise=torch.randn_like(y)
        t=torch.tensor([self.num_timesteps-1,]*y.shape[0],device=y.device).long()
        # Adding noise to the LQ image
        return y + _extract_into_tensor(self.kappa*self.sqrt_etas,t,y.shape)*noise
        
    
    def training_losses(
            self,
            x_start,
            y,
            t,
            model_kwargs=None,
            noise=None,
    ):
        
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param y: the [N x C x ...] tensor of degraded inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :param up_sample_lq: Upsampling low-quality image before encoding
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        if model_kwargs is None:
            model_kwargs={}
        
        # Transforming to Latent Space
        z_y=y
        z_start=x_start

        if noise is None:
            noise=torch.randn_like(z_start)

        # Sample out the diffused images at timestep t
        z_t=self.q_sample(z_start,z_y,t,noise=noise)

        terms={}

        if self.loss_type==LossType.MSE or self.loss_type==LossType.WEIGHTED_MSE:
            model_output=self.denoise_fn(self._scale_input(z_t,t),t,z_y)
            target={
                ModelMeanType.START_X: z_start,
            }[self.model_mean_type]

            terms["mse"]=mean_flat((target-model_output)**2)

            if self.loss_type == LossType.WEIGHTED_MSE:
                weights = _extract_into_tensor(self.weight_loss_mse, t, t.shape)
            else:
                weights = 1

            terms["mse"]*=weights
        else:
            raise NotImplementedError(self.loss_type)
        
        if self.model_mean_type == ModelMeanType.START_X:      # predict x_0
            pred_zstart = model_output
        else:
            raise NotImplementedError(self.model_mean_type)

        return terms, z_t, pred_zstart
    
    
    def forward(self, x, *args, **kwargs):
        x_orig = x[0]
        x_recon = x[1]
        b, c, h, w = x_orig.shape
        device = x_orig.device

        # pad_h = (4 - h % 4) % 4  # How much to pad in height
        # pad_w = (4 - w % 4) % 4  # How much to pad in width

        # # Apply padding (pad left, pad right, pad top, pad bottom)
        # x_orig = F.pad(x_orig, (0, pad_w, 0, pad_h), mode='reflect')
        # x_recon = F.pad(x_recon, (0, pad_w, 0, pad_h), mode='reflect')

        
        #Random 32 timesteps to noisy the reconstructed image
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        #Ground Truth as well as reconstructed is given
        #The model predict from the x_recon
        #The loss is calculated between x_recon and x_orig
        return self.training_losses(x_orig, x_recon, t,*args, **kwargs)