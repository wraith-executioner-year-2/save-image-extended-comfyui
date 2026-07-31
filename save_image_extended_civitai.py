from .save_image_extended import SaveImageExtended
import nodes

class SaveImageExtendedCivitai(SaveImageExtended):

  @classmethod
  def INPUT_TYPES(cls):
    inputs = super().INPUT_TYPES()
    inputs["required"].update({
      "civitai_positive": ("STRING", {"multiline": True, "default": ""}),
      "civitai_negative": ("STRING", {"multiline": True, "default": ""}),
      "civitai_seed": ("INT", {"default": 0}),
      "civitai_steps": ("INT", {"default": 20}),
      "civitai_cfg": ("FLOAT", {"default": 7.0}),
      # "civitai_sampler": (comfy.samplers.KSampler.SAMPLERS,),
      # "civitai_sampler": ("STRING", {"default": "euler"}),
      # "civitai_sampler": (nodes.KSampler.INPUT_TYPES()["required"]["sampler_name"][0],),
      "civitai_sampler": nodes.KSampler.INPUT_TYPES()["required"]["sampler_name"],
    })
    return inputs

  def save_images(self, images, filename_prefix, filename_keys, foldername_prefix, foldername_keys, delimiter, save_job_data, job_data_per_image, job_custom_text, save_metadata, counter_digits, counter_position, one_counter_per_folder, image_preview, output_ext, negative_text_opt=None, positive_text_opt=None, extra_pnginfo=None, prompt=None, quality=90, named_keys=False, civitai_positive="", civitai_negative="", civitai_seed=0, civitai_steps=20, civitai_cfg=7.0, civitai_sampler="euler"):
    self.original_prompt = prompt
    self.original_extra_pnginfo = extra_pnginfo

    self.civitai_positive = civitai_positive
    self.civitai_negative = civitai_negative
    self.civitai_seed = civitai_seed
    self.civitai_steps = civitai_steps
    self.civitai_cfg = civitai_cfg
    self.civitai_sampler = civitai_sampler

    return super().save_images(
      images,
      filename_prefix,
      filename_keys,
      foldername_prefix,
      foldername_keys,
      delimiter,
      save_job_data,
      job_data_per_image,
      job_custom_text,
      save_metadata,
      counter_digits,
      counter_position,
      one_counter_per_folder,
      image_preview,
      output_ext,
      negative_text_opt,
      positive_text_opt,
      extra_pnginfo,
      prompt,
      quality,
      named_keys
    )

  def writeImage(self, image_path, img, prompt, save_metadata=True, extra_pnginfo=None, quality=90):
    if image_path.lower().endswith(".png"):
        if extra_pnginfo is None:
            extra_pnginfo = {}

        extra_pnginfo["workflow"] = self.original_prompt

        prompt = {
            "1": {
              "inputs": {
                "clip_name": "qwen_3_06b_base.safetensors"
              },
              "class_type": "CLIPLoader"
            },
            "2": {
              "inputs": {
                "vae_name": "qwen_image_vae.safetensors"
              },
              "class_type": "VAELoader"
            },
            "3": {
              "inputs": {
                "unet_name": "anima_turboV10.safetensors"
              },
              "class_type": "UNETLoader"
            },
            "4": {
              "inputs": {
                "batch_size": 1,
                "width": img.width,
                "height": img.height
              },
              "class_type": "EmptyLatentImage"
            },
            "5": {
              "inputs": {
                "seed": self.civitai_seed,
                "steps": self.civitai_steps,
                "cfg": self.civitai_cfg,
                "sampler_name": self.civitai_sampler,
                "scheduler": "beta",
                "model": ["3",0],
                "positive": ["6",0],
                "negative": ["7",0],
                "latent_image": ["4",0]
              },
              "class_type": "KSampler"
            },
            "6": {
              "inputs": {
                "clip": ["1",0],
                "text": self.civitai_positive
              },
              "class_type": "CLIPTextEncode"
            },
            "7": {
              "inputs": {
                "clip": ["1",0],
                "text": self.civitai_negative
              },
              "class_type": "CLIPTextEncode"
            },
            "8": {
              "inputs": {
                "samples": ["5",0],
                "vae": ["2",0]
              },
              "class_type": "VAEDecode"
            }
        }

    return super().writeImage(
      image_path,
      img,
      prompt,
      save_metadata,
      extra_pnginfo,
      quality
    )


NODE_CLASS_MAPPINGS = {
  "SaveImageExtendedCivitai": SaveImageExtendedCivitai,
}

NODE_DISPLAY_NAME_MAPPINGS = {
  "SaveImageExtendedCivitai": "💾 Save Image Extended CivitAI",
}