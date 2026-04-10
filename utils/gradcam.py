import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imutils

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_backward_hook(self.save_gradients)
        
    def save_activations(self, module, input, output):
        self.activations = output
        
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def generate_heatmap(self, input_tensor, class_idx=None):
        # Ensure the input tensor has gradient tracking enabled
        input_tensor.requires_grad = True
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        loss = output[0, class_idx]
        loss.backward()
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the channels by corresponding gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels to get the heatmap
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        
        # ReLU on heatmap
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        
        # Normalize between 0 and 1
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        return heatmap

def overlay_heatmap(heatmap, image, alpha=0.4):
    """
    Overlay Grad-CAM heatmap and detect tumor boundaries.
    """
    heatmap_u8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    image_np = np.array(image)
    heatmap_color = cv2.resize(heatmap_color, (image_np.shape[1], image_np.shape[0]))
    
    # Adaptive thresholding for multi-peak detection
    _, thresh = cv2.threshold(heatmap_u8, 127, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    regions = []
    overlayed = cv2.addWeighted(heatmap_color, alpha, image_np, 1 - alpha, 0)
    
    for i, c in enumerate(cnts):
        area = cv2.contourArea(c)
        if area < 100: continue
        
        x, y, w, h = cv2.boundingRect(c)
        regions.append({
            "id": i,
            "bbox": (x, y, w, h),
            "area": area,
            "centroid": (int(x + w/2), int(y + h/2))
        })
        
        # Highlight green boundaries
        cv2.drawContours(overlayed, [c], -1, (0, 255, 0), 2)
        cv2.rectangle(overlayed, (x, y), (x + w, y + h), (255, 255, 0), 1)
        
    return overlayed, regions
