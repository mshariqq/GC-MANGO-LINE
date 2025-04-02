!pip install transformers diffusers datasets accelerate torch torchvision requests

import torch
from diffusers import StableDiffusionXLPipeline
import gc
import psutil
import os
from typing import List, Optional
from google.colab import files  # For downloading files in Colab

def clean_system_and_gpu_memory():
    """Forcefully clear all system RAM and GPU RAM."""
    print("Cleaning system RAM and GPU RAM...")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # Clear system RAM
    gc.collect()
    for _ in range(3):  # Multiple passes to ensure thorough cleanup
        gc.collect()

    print("System and GPU memory cleaned.")

class LowMemorySDXL:
    def __init__(self, 
                 model_id: str = "SG161222/RealVisXL_V4.0",  # RealVis XL model
                 chunk_size: int = 1,
                 height: int = 1024,  # RealVis XL works best at 1024x1024
                 width: int = 1024):
        """
        Initialize SDXL with memory optimization
        
        Args:
            model_id: Hugging Face model ID
            chunk_size: Number of images to generate per chunk
            height: Output image height
            width: Output image width
        """
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.height = height
        self.width = width
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with optimizations
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # FP16 for lower memory
            variant="fp16",
            use_safetensors=True  # Faster loading, less memory
        )
        
        # Enable memory-efficient features
        self.pipe.enable_attention_slicing()  # Split attention layers into chunks
        self.pipe.enable_vae_slicing()  # Split VAE into chunks
        self.pipe.enable_vae_tiling()  # Tile VAE for large images
        
        # Move to GPU
        self.pipe.to(self.device)
        
        # Clear initial memory
        self.clear_memory()

    def clear_memory(self):
        """Clear GPU and system memory"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Multiple passes for system RAM
        for _ in range(2):
            gc.collect()

    def get_memory_stats(self) -> dict:
        """Get current memory usage stats"""
        process = psutil.Process(os.getpid())
        ram_usage = process.memory_info().rss / 1024**2  # in MB
        
        gpu_usage = 0
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / 1024**2  # in MB
            
        return {
            "ram_mb": ram_usage,
            "gpu_mb": gpu_usage
        }

    def generate_images(self, 
                       prompts: List[str],
                       num_inference_steps: int = 30,
                       guidance_scale: float = 7.5,
                       negative_prompts: Optional[List[str]] = None) -> List:
        """
        Generate images in chunks with memory management
        
        Args:
            prompts: List of text prompts
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            negative_prompts: Optional list of negative prompts
            
        Returns:
            List of PIL images
        """
        results = []
        
        # Ensure negative prompts match length of prompts
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        
        # Process in chunks
        for i in range(0, len(prompts), self.chunk_size):
            chunk_prompts = prompts[i:i + self.chunk_size]
            chunk_neg_prompts = negative_prompts[i:i + self.chunk_size]
            
            try:
                # Generate images with no gradient computation
                with torch.no_grad():
                    # Use torch.autocast for mixed precision
                    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                        images = self.pipe(
                            prompt=chunk_prompts,
                            negative_prompt=chunk_neg_prompts,
                            height=self.height,
                            width=self.width,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale
                        ).images
                
                results.extend(images)
                
                # Clear memory after each chunk
                self.clear_memory()
                
                # Print memory usage
                memory_stats = self.get_memory_stats()
                print(f"Chunk {i//self.chunk_size + 1}/{len(prompts)//self.chunk_size + 1} - "
                      f"RAM: {memory_stats['ram_mb']:.2f}MB, "
                      f"GPU: {memory_stats['gpu_mb']:.2f}MB")
                
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                self.clear_memory()
                continue
                
        return results

    def cleanup(self):
        """Complete cleanup of model and memory"""
        del self.pipe
        self.clear_memory()

# Example usage
def main():
    # Clean all system and GPU memory before starting
    clean_system_and_gpu_memory()

    # Initialize SDXL model
    model = LowMemorySDXL(
        model_id="SG161222/RealVisXL_V4.0",  # RealVis XL model
        chunk_size=1,  # Keep at 1 for SDXL due to size
        height=768,   # RealVis XL works best at 1024x1024
        width=1024
    )
    
    # Sample prompts
    prompts = [
        "A businessman in a sharp suit confidently presenting his latest laptop on stage, with a large, vibrant screen behind him displaying the product's features and specs. The audience is engaged, and the atmosphere is professional and dynamic"
    ]
    
    # Optional negative prompts for better quality
    negative_prompts = [
        "blurry, low quality, cartoonish, distorted, extra limbs, sunny, overexposed, underexposed, grainy, noisy, unrealistic, poorly drawn, bad anatomy."
    ]
    
    # Generate images
    print("Generating SDXL images...")
    images = model.generate_images(
        prompts,
        num_inference_steps=30,  # Fewer steps for memory
        guidance_scale=7.5,
        negative_prompts=negative_prompts
    )
    
    # Save images
    for idx, img in enumerate(images):
        image_path = f"sdxl_image_{idx}.png"
        img.save(image_path)
        print(f"Saved SDXL image {idx} at {image_path}")
        
        # Download the image in Google Colab
        files.download(image_path)
    
    # Cleanup
    model.cleanup()

if __name__ == "__main__":
    main()