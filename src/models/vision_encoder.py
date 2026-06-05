import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig
from transformers.models.vit.modeling_vit import ViTModel as HFViTModel
# Note: TrOCR's encoder is actually an instance of ViTModel, despite the name DeiT being used sometimes in literature.
# The microsoft/trocr-base-stage1 uses ViT for its encoder.

class DeiTClassifier(nn.Module):
    def __init__(self, num_classes=454, model_name='microsoft/trocr-base-stage1'):
        super().__init__()
        
        # Load the encoder part of TrOCR
        # We can load it by downloading the encoder only using ViTModel
        # However, TrOCR combines VisionEncoderDecoderModel. 
        # To get just the vision encoder, we can use ViTModel and point to the trocr checkpoint.
        # But wait, trocr weights are saved as encoder.xx. 
        # A simpler way is to load the full TrOCR model and extract the encoder.
        from transformers import VisionEncoderDecoderModel
        print(f"Loading encoder from {model_name}...")
        trocr = VisionEncoderDecoderModel.from_pretrained(model_name)
        
        # The encoder is a ViTModel
        self.encoder = trocr.encoder
        
        # Determine the hidden size
        hidden_size = self.encoder.config.hidden_size
        
        # Add a classification head
        self.classifier = nn.Linear(hidden_size, num_classes)
        
    def forward(self, pixel_values):
        # Pass through the encoder
        outputs = self.encoder(pixel_values=pixel_values)
        
        # We take the pooled output for classification (usually index 0, the CLS token)
        # ViTModel returns (last_hidden_state, pooler_output, ...)
        # pooler_output is already derived from the CLS token
        if outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Fallback if pooler is None
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        logits = self.classifier(pooled_output)
        return logits
    
    def get_encoder_weights(self):
        """
        Returns the encoder module itself. Useful for Stage 2 transfer.
        """
        return self.encoder
