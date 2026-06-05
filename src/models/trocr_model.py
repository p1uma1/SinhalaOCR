import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, AutoTokenizer

class SinhalaTrOCR(nn.Module):
    def __init__(self, encoder_pretrained_path=None, decoder_model_name='nlp-rilab/sinbert-base'):
        super().__init__()
        
        # Initialize the VisionEncoderDecoderModel
        # This automatically sets up the cross-attention layers in the SinBERT model 
        # so it can act as a decoder.
        print("Initializing VisionEncoderDecoderModel...")
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            "microsoft/trocr-base-stage1", 
            decoder_model_name
        )
        
        # If we have weights from Stage 1, load them into the encoder
        if encoder_pretrained_path is not None:
            print(f"Loading Stage 1 encoder weights from {encoder_pretrained_path}...")
            self.model.encoder.load_state_dict(torch.load(encoder_pretrained_path))
            
        # Configure decoder tokens using the SinBERT tokenizer
        tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        
        self.model.config.decoder_start_token_id = tokenizer.cls_token_id
        self.model.config.pad_token_id = tokenizer.pad_token_id
        
        # Ensure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        
        # Beam search configuration for evaluation
        self.model.config.eos_token_id = tokenizer.sep_token_id
        self.model.config.max_length = 128
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4

    def forward(self, pixel_values, labels=None):
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def generate(self, pixel_values):
        return self.model.generate(pixel_values)
