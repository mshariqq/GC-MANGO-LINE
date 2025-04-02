!pip install transformers diffusers datasets accelerate torch torchvision requests gradio

import torch
from diffusers import StableDiffusionXLPipeline
import gc
import psutil
import os
from typing import List, Optional
import gradio as gr
import tempfile
from datetime import datetime, timedelta
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

# Initialize the model once and reuse it
clean_system_and_gpu_memory()
model = LowMemorySDXL(
    model_id="SG161222/RealVisXL_V4.0",
    chunk_size=1,
    height=1024,
    width=1024
)

# Temporary link storage (in-memory for demo purposes)
# In a production environment, you'd want to use a proper database
temporary_links = {}

def generate_and_share_image(prompt, negative_prompt, steps, guidance_scale):
    try:
        # Generate the image
        images = model.generate_images(
            prompts=[prompt],
            negative_prompts=[negative_prompt],
            num_inference_steps=steps,
            guidance_scale=guidance_scale
        )
        
        if not images:
            return None, "Error: No images were generated."
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            images[0].save(tmp_file.name)
            tmp_file_path = tmp_file.name
        
        # Create a temporary link (expires in 1 hour)
        import uuid
        link_id = str(uuid.uuid4())
        expiry_time = datetime.now() + timedelta(hours=1)
        
        temporary_links[link_id] = {
            'path': tmp_file_path,
            'expiry': expiry_time,
            'prompt': prompt
        }
        
        # In a real deployment, you'd generate a proper URL
        link_url = f"http://localhost:7860/temp/{link_id}"  # This is just for display
        
        return images[0], f"Temporary link created (expires in 1 hour): {link_url}"
    
    except Exception as e:
        return None, f"Error generating image: {str(e)}"

def get_temporary_image(link_id):
    """Retrieve a temporary image by its ID (for demo purposes)"""
    if link_id not in temporary_links:
        return None, "Link not found or expired"
    
    link_data = temporary_links[link_id]
    
    if datetime.now() > link_data['expiry']:
        # Clean up expired link
        try:
            os.remove(link_data['path'])
        except:
            pass
        del temporary_links[link_id]
        return None, "Link expired"
    
    return link_data['path'], f"Generated from prompt: {link_data['prompt']}"

# Gradio interface
with gr.Blocks(title="SDXL Image Generator") as demo:
    gr.Markdown("# 🎨 SDXL Image Generator")
    gr.Markdown("Generate high-quality images using Stable Diffusion XL")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Describe the image you want to generate...")
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value="blurry, low quality, cartoonish, distorted, extra limbs, sunny, overexposed, underexposed, grainy, noisy, unrealistic, poorly drawn, bad anatomy.",
                placeholder="What to exclude from the image..."
            )
            steps = gr.Slider(label="Inference Steps", minimum=10, maximum=50, value=30, step=1)
            guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=15.0, value=7.5, step=0.1)
            generate_btn = gr.Button("Generate Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Generated Image")
            output_link = gr.Textbox(label="Temporary Link", interactive=False)
    
    # Example prompts
    gr.Examples(
        examples=[
            ["A beautiful sunset over a mountain landscape, photorealistic, 8K"],
            ["A futuristic cityscape at night, neon lights, cyberpunk style"],
            ["A cute kitten playing with a ball of yarn, soft lighting"]
        ],
        inputs=prompt
    )
    
    generate_btn.click(
        fn=generate_and_share_image,
        inputs=[prompt, negative_prompt, steps, guidance_scale],
        outputs=[output_image, output_link]
    )

# For Colab deployment
def run_in_colab():
    print("Launching Gradio interface...")
    demo.launch(share=True)

if __name__ == "__main__":
    # For script execution
    import sys
    if 'google.colab' in sys.modules:
        run_in_colab()
    else:
        demo.launch()