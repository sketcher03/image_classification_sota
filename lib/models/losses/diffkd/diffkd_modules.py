import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionTimestepScheduler:
    """
    Scheduler to progressively adjust diffusion timesteps during training.
    """
    def __init__(self, T_min, T_max, total_epochs):
        self.T_min = T_min
        self.T_max = T_max
        self.total_epochs = total_epochs

    def get_timesteps(self, epoch):
        """
        Calculate the number of timesteps for the current epoch.
        """
        return int(self.T_min + (self.T_max - self.T_min) * (epoch / self.total_epochs))
    

class NoiseAdapter(nn.Module):
    def __init__(self, channels, kernel_size=3, weight_attention=1.0):
        super().__init__()
        
        # Store the weight for attention scaling
        self.weight_attention = weight_attention

        '''
        # Define the spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Existing feature extraction mechanism
        if kernel_size == 3:
            self.feat = nn.Sequential(
                Bottleneck(channels, channels, reduction=8),
                nn.AdaptiveAvgPool2d(1)
            )
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            )

        '''
        
        # Prediction layer
        self.pred = nn.Linear(channels, 2)
        self.kernel_size = kernel_size
        self.spatial_attention = None  # To be initialized dynamically
        self.feat = None  # Feature extraction module (dynamic)
        self.pred = None  # Prediction head (dynamic)

    def _initialize_feat(self, channels, device):
        """Dynamically initialize or update the feature extraction module."""
        # print(f"Reinitializing feat for {channels} channels with kernel_size={self.kernel_size}")
        if self.kernel_size == 3:
            self.feat = nn.Sequential(
                Bottleneck(channels, channels, reduction=8),
                nn.AdaptiveAvgPool2d(1)
            ).to(device)
        else:
            self.feat = nn.Sequential(
                nn.Conv2d(channels, channels * 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels * 2, channels, 1),
                nn.BatchNorm2d(channels),
            ).to(device)

    def forward(self, x):

        # print(f"self.feat before reinitialization: {self.feat}")

        # Channel Attention
        b, c, h, w = x.size()

        # Dynamically initialize spatial attention if not already set
        if self.spatial_attention is None or self.spatial_attention[0].in_channels != c:
            # print(f"Initializing/Updating Spatial Attention for {c} channels")
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(c, c // 8, kernel_size=1, bias=False),  # Reduce channels
                nn.BatchNorm2d(c // 8),
                nn.ReLU(inplace=True),
                nn.Conv2d(c // 8, 1, kernel_size=1, bias=False),  # Map to single channel
                nn.Sigmoid()
            ).to(x.device)

        # print(f"self.feat before reinitialization: {self.feat}")

        # Dynamically initialize feature extraction module (self.feat)
        if self.feat is None or (
            isinstance(self.feat[0], nn.Conv2d) and self.feat[0].in_channels != c
        ):
            self._initialize_feat(c, x.device)

        # print(f"self.feat after reinitialization: {self.feat}")

        # Dynamically initialize prediction head
        if self.pred is None or self.pred.in_features != c:
            # print(f"Initializing Prediction Head for {c} channels")
            self.pred = nn.Linear(c, 2).to(x.device)

        # Apply spatial attention
        attention_weights = self.spatial_attention(x)
        x = x * (attention_weights * self.weight_attention)  # Scale attention map with weight_attention

        # print(f"Input Shape: {x.shape}")
        # print(f"Spatial Attention Weights Shape: {attention_weights.shape}")
        
        # Pass through existing feature extraction and prediction layers
        x = self.feat(x).flatten(1)
        x = self.pred(x).softmax(1)[:, 0]
        return x



class DualAttention(nn.Module):
    """
    Dual Attention module applies spatial and channel attention mechanisms to a feature map.
    """

    def __init__(self, channels, reduction=8):
        super(DualAttention, self).__init__()
        

        # Channel Attention
        self.channel_attention = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel Attention
        b, c, h, w = x.size()
        channel_weights = self.channel_attention(x.view(b, c, -1).mean(dim=2)).view(b, c, 1, 1)
        x = x * channel_weights

        # Spatial Attention
        spatial_weights = self.spatial_attention(x.mean(dim=1, keepdim=True))
        x = x * spatial_weights

        return x


class DiffusionModel(nn.Module):
    def __init__(self, channels_in, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                Bottleneck(channels_in, channels_in),
                Bottleneck(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in)
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1)
            )

    def forward(self, noisy_image, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image
        feat = feat + self.time_embedding(t)[..., None, None]
        ret = self.pred(feat)
        return ret


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, channels, 1, padding=0),
        )

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out

    def forward_encoder(self, x):
        return self.encoder(x)
    

class DDIMPipeline:
    '''
    Modified from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddim/pipeline_ddim.py
    '''

    def __init__(self, model, scheduler, noise_adapter=None, solver='ddim'):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter = noise_adapter
        self._iter = 0
        self.solver = solver

    def __call__(
            self,
            batch_size,
            device,
            dtype,
            shape,
            feat,
            generator = None,
            eta: float = 0.0,
            num_inference_steps: int = 50,
            proj = None
    ):

        # Sample gaussian noise to begin loop
        image_shape = (batch_size, *shape)

        if self.noise_adapter is not None:
            noise = torch.randn(image_shape, device=device, dtype=dtype)
            timesteps = self.noise_adapter(feat)
            image = self.scheduler.add_noise_diff2(feat, noise, timesteps)
        else:
            image = feat

        # set step values
        self.scheduler.set_timesteps(num_inference_steps*2)

        for t in self.scheduler.timesteps[len(self.scheduler.timesteps)//2:]:
            noise_pred = self.model(image, t.to(device))

            # 2. predict previous mean of image x_t-1 and add variance depending on eta
            # eta corresponds to η in paper and should be between [0, 1]
            # do x_t -> x_t-1
            image = self.scheduler.step(
                noise_pred, t, image, eta=eta, use_clipped_model_output=True, generator=generator
            )['prev_sample'] 
                
        self._iter += 1        
        return image


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)
        return out + identity
