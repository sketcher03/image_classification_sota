import torch
from torch import nn
import torch.nn.functional as F
from .diffkd_modules import DiffusionModel, NoiseAdapter, AutoEncoder, DDIMPipeline, DualAttention, DiffusionTimestepScheduler
from .scheduling_ddim import DDIMScheduler


class DiffKD(nn.Module):
    def __init__(
            self,
            student_channels,
            teacher_channels,
            kernel_size=3,
            inference_steps=5,
            num_train_timesteps=1000,
            use_ae=False,
            ae_channels=None,
            weight_attention=1.0,
            T_min=5,
            T_max=50,
            total_epochs=100
    ):
        super().__init__()
        
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        # AE for compress teacher feature
        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels
        
        # Initialize scheduler for timesteps
        self.diff_scheduler = DiffusionTimestepScheduler(T_min, T_max, total_epochs)
        
        self.noise_adapter = NoiseAdapter(teacher_channels, kernel_size, weight_attention)

        # transform student feature to the same dimension as teacher
        self.trans = nn.Conv2d(student_channels, teacher_channels, 1)
        # diffusion model - predict noise
        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, clip_sample=False, beta_schedule="linear")

        # DualAttention for feature alignment
        self.dual_attention = DualAttention(teacher_channels)

        # pipeline for denoising student feature
        self.pipeline = DDIMPipeline(self.model, self.scheduler, self.noise_adapter)
        self.proj = nn.Sequential(nn.Conv2d(teacher_channels, teacher_channels, 1), nn.BatchNorm2d(teacher_channels))

    def forward(self, student_feat, teacher_feat, epoch):

        # Dynamically adjust diffusion timesteps
        num_inference_steps = self.diff_scheduler.get_timesteps(epoch)
        # print(f"Epoch {epoch}: Using {num_inference_steps} timesteps for diffusion.")
        
        # project student feature to the same dimension as teacher feature
        student_feat = self.trans(student_feat)

        """
        Forward pass for DiffKD with feature and logits distillation and progressive timestep adjustment.

        Args:
            teacher_feat: Features from the teacher model.
            student_feat: Features from the student model.

        Returns:
            refined_feat: Denoised student features
            teacher_feat: Processed teacher features
            ddim_loss: Loss for training the diffusion model
            rec_loss: Reconstruction loss for the autoencoder (if used)
        """

        # use autoencoder on teacher feature
        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        # Apply DualAttention for feature enhancement
        teacher_feat = self.dual_attention(teacher_feat)
        student_feat = self.dual_attention(student_feat)

        # denoise student feature
        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            num_inference_steps=num_inference_steps,
            proj=self.proj
        )
        refined_feat = self.proj(refined_feat)
        
        # train diffusion model
        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        # Sample noise to add to the images
        noise = torch.randn(gt_feat.shape, device=gt_feat.device) #.to(gt_feat.device)
        bs = gt_feat.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (bs,), device=gt_feat.device).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
