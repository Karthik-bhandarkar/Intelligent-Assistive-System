from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import re

# Global cache for model to prevent reloading
_processor = None
_model = None
_device = None

def load_blip():
    global _processor, _model, _device
    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading BLIP model on {_device}...")
        
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
        if _device.type == 'cuda':
             _model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large", 
                torch_dtype=torch.float16
            ).to(_device)
        else:
            _model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-large"
            ).to(_device)
            
        print("BLIP model loaded.")
    return _processor, _model, _device

def generate_caption(image):
    """Generate a caption using BLIP model with conditional prompting and beam search."""
    try:
        processor, model, device = load_blip()
        
        # Conditional Prompting
        text = "Describe the image in detail:"
        
        # Preprocess input
        inputs = processor(image, text, return_tensors="pt").to(device)
        
        # Beam Search for higher quality
        out = model.generate(
            **inputs,
            num_beams=5,
            max_length=40,
            min_length=10,
            repetition_penalty=1.1
        )
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        # Basic post-processing
        caption = caption.replace("Describe the image in detail: ", "").strip()
        
        # Logic to clean artifacts
        prompt = "describe the image in detail"
        lower_caption = caption.lower()
        if prompt in lower_caption:
            caption = caption[lower_caption.index(prompt) + len(prompt):].strip()
        
        # Remove leading punctuation/artifacts like ": ", "ly," using regex
        caption = re.sub(r'^[:,\s]*(ly|ly,|ly\.)[:,\s]*', '', caption, flags=re.IGNORECASE).strip()
        
        # Capitalize first letter
        caption = caption.capitalize()
        
        return caption
    except Exception as e:
        print(f"Captioning error: {e}")
        return "Error generating caption."
