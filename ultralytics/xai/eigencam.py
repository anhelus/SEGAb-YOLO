import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Optional

class EigenCAM:
    """
    EigenCAM: Principle Component Analysis (PCA) based Class Activation Mapping.
    Computes the first principle component of the 2D activations.
    Reference: https://arxiv.org/abs/2008.00299
    """
    def __init__(self, model, target_layers: List[torch.nn.Module], task: str = 'detect'):
        self.model = model
        self.target_layers = target_layers
        self.activations = []
        self.handles = []
        self.task = task
        self._register_hooks()

    def _register_hooks(self):
        def hook(module, input, output):
            # If output is a tuple (common in YOLO heads), take the first element
            if isinstance(output, tuple):
                output = output[0]
            self.activations.append(output.detach())

        for layer in self.target_layers:
            self.handles.append(layer.register_forward_hook(hook))

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _process_activations(self, activations):
        """
        Computes the first principal component of the activations.
        """
        # Upsample activations to the largest spatial size in the list
        target_size = activations[0].shape[2:]
        upsampled_activations = []
        for act in activations:
            if act.ndim == 3: # missing batch dim
                act = act.unsqueeze(0)
            if act.shape[2:] != target_size:
                act = F.interpolate(act, size=target_size, mode='bilinear', align_corners=False)
            upsampled_activations.append(act)

        # Concatenate all activations along channel dimension
        # Shape: [B, C_total, H, W]
        concatenated_activations = torch.cat(upsampled_activations, dim=1)
        
        # Reshape for PCA: [B, H*W, C_total] -> project to 1D
        # For simplicity, we process batch items individually or average? 
        # Standard EigenCAM works per image.
        
        cams = []
        for i in range(concatenated_activations.shape[0]):
            act = concatenated_activations[i] # [C, H, W]
            c, h, w = act.shape
            
            # Reshape to [C, H*W] and transpose to [H*W, C]
            reshaped_act = act.reshape(c, -1).transpose(0, 1)
            
            # Center the data
            reshaped_act = reshaped_act - reshaped_act.mean(dim=0, keepdim=True)
            
            # SVD
            U, S, V = torch.svd(reshaped_act)
            
            # First principal component is the first column of U
            # U is [H*W, K]
            pc1 = U[:, 0]
            
            # Reshape back to [H, W]
            cam = pc1.reshape(h, w)
            
            # Normalize to 0-1
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-7)
            
            cams.append(cam.cpu().numpy())
            
        return cams

    def __call__(self, x):
        self.activations = []
        # Run inference
        # We assume the model is already in eval mode
        with torch.no_grad():
            if isinstance(x, (str, np.ndarray)):
                # If input is path or numpy, we let the model handle preprocessing usually
                # But to stick to pure PyTorch hooks, we prefer getting the preprocessed tensor.
                # If x is already a tensor:
                self.model(x)
            else:
                 self.model(x)
                 
        if not self.activations:
            raise ValueError("No activations captured. Check target_layers.")

        cams = self._process_activations(self.activations)
        
        # Clear activations for next run
        self.activations = []
        
        return cams
