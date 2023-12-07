import torch
import os

import comfy.utils
import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention
import folder_paths

from torch import nn
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as TT

from .resampler import Resampler
from .util import contrast_adaptive_sharpening, zeroed_hidden_states, image_add_noise

# set the models directory backward compatible
GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
MODELS_DIR = GLOBAL_MODELS_DIR if os.path.isdir(GLOBAL_MODELS_DIR) else os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["ipadapter"] = ([MODELS_DIR], folder_paths.supported_pt_extensions)
else:
    folder_paths.folder_names_and_paths["ipadapter"][1].update(folder_paths.supported_pt_extensions)

class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class To_KV(nn.Module):
    def __init__(self, state_dict):
        super().__init__()

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
            self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    if key not in to["patches_replace"]["attn2"]:
        patch = CrossAttentionPatch(**patch_kwargs)
        to["patches_replace"]["attn2"][key] = patch
    else:
        to["patches_replace"]["attn2"][key].set_new_condition(**patch_kwargs)


class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False, is_plus=False, is_full=False):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full

        self.image_proj_model = self.init_proj() if not is_plus else self.init_proj_plus()
        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(ipadapter_model["ip_adapter"])

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
        if self.is_full:
            image_proj_model = MLPProjModel(
                cross_attention_dim=self.cross_attention_dim,
                clip_embeddings_dim=self.clip_embeddings_dim
            )
        else:
            image_proj_model = Resampler(
                dim=self.cross_attention_dim,
                depth=4,
                dim_head=64,
                heads=20 if self.is_sdxl else 12,
                num_queries=self.clip_extra_context_tokens,
                embedding_dim=self.clip_embeddings_dim,
                output_dim=self.output_cross_attention_dim,
                ff_mult=4
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed):
        image_prompt_embeds = self.image_proj_model(clip_embed)
        uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

class CrossAttentionPatch:
    # forward for patching
    def __init__(self, weight, ipadapter, device, dtype, number, cond, uncond, weight_type, sigma_start=0.0, sigma_end=1.0, attn_group=0):
        self.weights = [weight]
        self.ipadapters = [ipadapter]
        self.conds = [cond]
        self.unconds = [uncond]
        self.device = 'cuda' if 'cuda' in device.type else 'cpu'
        self.dtype = dtype if 'cuda' in self.device else torch.bfloat16
        self.number = number
        self.weight_type = [weight_type]
        self.sigma_start = [sigma_start]
        self.sigma_end = [sigma_end]
        self.k_key = str(self.number*2+1) + "_to_k_ip"
        self.v_key = str(self.number*2+1) + "_to_v_ip"
        self.groups = [attn_group]
    
    def set_new_condition(self, weight, ipadapter, device, dtype, number, cond, uncond, weight_type, sigma_start=0.0, sigma_end=1.0, attn_group=0):
        self.weights.append(weight)
        self.ipadapters.append(ipadapter)
        self.conds.append(cond)
        self.unconds.append(uncond)
        self.device = 'cuda' if 'cuda' in device.type else 'cpu'
        self.dtype = dtype if 'cuda' in self.device else torch.bfloat16
        self.weight_type.append(weight_type)
        self.sigma_start.append(sigma_start)
        self.sigma_end.append(sigma_end)
        self.groups.append(attn_group)

    def __call__(self, n, context_attn2, value_attn2, extra_options):
        org_dtype = n.dtype
        cond_or_uncond = extra_options["cond_or_uncond"]
        sigma = extra_options["sigmas"][0].item() if 'sigmas' in extra_options else 999999999.9

        with torch.autocast(device_type=self.device, dtype=self.dtype):
            # create a list of adapters that need to be applied
            tasks = list(zip(self.weights, self.conds, self.unconds,
                        self.ipadapters, self.weight_type,
                        self.sigma_start, self.sigma_end, self.groups))
            # now filter out tasks that are not in the sigma range
            tasks = [task for task in tasks if sigma <= task[5] and sigma >= task[6]]
            # sort tasks by groups, so that once each group is done we can call crossattention and free memory
            tasks.sort(key=lambda x: x[7])
            # get the last index of each group
            group_end_indices = [i for i, x in enumerate(tasks[:-1]) if x[7] != tasks[i+1][7]] + [len(tasks)-1]

            q = n
            k = []
            v = []
            b = q.shape[0] // len(cond_or_uncond)
            out = 0

            # if text is separate from image, or if there are no images, do the text group first
            if not len(tasks) or tasks[0][7] != 0:
                out = optimized_attention(q, context_attn2, value_attn2, extra_options["n_heads"])
            else:
                k = [context_attn2]
                v = [value_attn2]

            for i, (weight, cond, uncond, ipadapter, weight_type, _, _, group) in enumerate(tasks):
                # cond = (n_images, n_tokens, d_clip)
                # we want to reshape to (1, n_images*n_tokens, d_clip)
                (ni, nt, _) = cond.shape
                cond = cond.reshape(1, -1, cond.shape[-1])
                uncond = uncond.reshape(1, -1, uncond.shape[-1])
                # we also need to reshape the weight, which is (n_images, 1) to (1, n_images*n_tokens, 1)
                # in particular, we need to repeat [1, 2] -> [1, 1, 2, 2] instead of [1, 2, 1, 2] (the latter is broadcasting)
                weight = torch.repeat_interleave(
                        weight.reshape(1, -1, 1),
                        repeats = nt, dim = 1).to(dtype=self.dtype, device=self.device)

                # do we need to pass these through separately ? 
                # no, they're linear
                k_cond = ipadapter.ip_layers.to_kvs[self.k_key](cond).repeat(b, 1, 1)
                k_uncond = ipadapter.ip_layers.to_kvs[self.k_key](uncond).repeat(b, 1, 1)
                v_cond = ipadapter.ip_layers.to_kvs[self.v_key](cond).repeat(b, 1, 1)
                v_uncond = ipadapter.ip_layers.to_kvs[self.v_key](uncond).repeat(b, 1, 1)
    
                # batch and weight the condition
                if weight_type.startswith("linear"):
                    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0) * weight
                    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0) * weight
                else:
                    ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
                    ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

                    if weight_type.startswith("channel"):
                        # code by Lvmin Zhang at Stanford University as also seen on Fooocus IPAdapter implementation
                        # please read licensing notes https://github.com/lllyasviel/Fooocus/blob/main/fooocus_extras/ip_adapter.py#L225
                        # we need to slice before we mean
                        # ip_v is shape [b, n_images * n_tokens, d_attn]
                        # we want to mean over the tokens of each image
                        # so we need to slice into [b, n_images, n_tokens, d_attn], mean over tokens, then repeat 
                        ip_v_slice = ip_v.reshape(b, ni, nt, -1)
                        ip_v_mean = torch.mean(ip_v_slice, dim=2).repeat(1, nt, 1)
                        ip_v_offset = ip_v - ip_v_mean
                        _, _, C = ip_k.shape
                        channel_penalty = float(C) / 1280.0
                        W = weight * channel_penalty
                        ip_k = ip_k * W
                        ip_v = ip_v_offset + ip_v_mean * W
                # add the condition to the current task
                k.append(ip_k)
                v.append(ip_v)
                # if we are at the end of the group, do the cross attention
                if i in group_end_indices:
                    k = torch.cat(k, dim=1)
                    v = torch.cat(v, dim=1)
                    out += optimized_attention(q, k, v, extra_options["n_heads"])
                    k = []
                    v = []

        return out.to(dtype=org_dtype)

class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (folder_paths.get_filename_list("ipadapter"), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"

    CATEGORY = "ipadapter"

    def load_ipadapter_model(self, ipadapter_file):
        ckpt_path = folder_paths.get_full_path("ipadapter", ipadapter_file)

        model = comfy.utils.load_torch_file(ckpt_path, safe_load=True)

        if ckpt_path.lower().endswith(".safetensors"):
            st_model = {"image_proj": {}, "ip_adapter": {}}
            for key in model.keys():
                if key.startswith("image_proj."):
                    st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
                elif key.startswith("ip_adapter."):
                    st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            model = st_model
                    
        if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
            raise Exception("invalid IPAdapter model {}".format(ckpt_path))

        return (model,)

class IPAdapterApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.01 }),
                "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "attn_group": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, ipadapter, model, weight, clip_vision=None, image=None,
                        weight_type="original", noise=None, embeds=None,
                        start_at=0.0, end_at=1.0, attn_group=0):
        self.dtype = model.model.diffusion_model.dtype
        self.device = comfy.model_management.get_torch_device()
        self.weight = weight
        self.is_full = "proj.0.weight" in ipadapter["image_proj"]
        self.is_plus = self.is_full or "latents" in ipadapter["image_proj"]

        output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_sdxl = output_cross_attention_dim == 2048
        cross_attention_dim = 1280 if self.is_plus and self.is_sdxl else output_cross_attention_dim
        clip_extra_context_tokens = 16 if self.is_plus else 4

        if embeds is not None:
            (clip_embed, clip_embed_weights, clip_embed_zeroed) = embeds
        else:
            if image.shape[1] != image.shape[2]:
                print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

            clip_embed = clip_vision.encode_image(image)
            clip_embed_weights = torch.ones(image.shape[0], 1, 1).to(self.device, dtype=self.dtype)
            neg_image = image_add_noise(image, noise) if noise > 0 else None
            
            if self.is_plus:
                clip_embed = clip_embed.penultimate_hidden_states
                if noise > 0:
                    clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
                else:
                    clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
            else:
                clip_embed = clip_embed.image_embeds
                if noise > 0:
                    clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
                else:
                    clip_embed_zeroed = torch.zeros_like(clip_embed)

        clip_embeddings_dim = clip_embed.shape[-1]

        self.ipadapter = IPAdapter(
            ipadapter,
            cross_attention_dim=cross_attention_dim,
            output_cross_attention_dim=output_cross_attention_dim,
            clip_embeddings_dim=clip_embeddings_dim,
            clip_extra_context_tokens=clip_extra_context_tokens,
            is_sdxl=self.is_sdxl,
            is_plus=self.is_plus,
            is_full=self.is_full,
        )
        
        self.ipadapter.to(self.device, dtype=self.dtype)

        image_prompt_embeds, uncond_image_prompt_embeds = self.ipadapter.get_image_embeds(clip_embed.to(self.device, self.dtype), clip_embed_zeroed.to(self.device, self.dtype))
        image_prompt_embeds = image_prompt_embeds.to(self.device, dtype=self.dtype)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.to(self.device, dtype=self.dtype)

        work_model = model.clone()

        sigma_start = model.model.model_sampling.percent_to_sigma(start_at)
        sigma_end = model.model.model_sampling.percent_to_sigma(end_at)

        patch_kwargs = {
            "number": 0,
            "weight": self.weight * clip_embed_weights,
            "ipadapter": self.ipadapter,
            "device": self.device,
            "dtype": self.dtype,
            "cond": image_prompt_embeds,
            "uncond": uncond_image_prompt_embeds,
            "weight_type": weight_type,
            "sigma_start": sigma_start,
            "sigma_end": sigma_end,
            "attn_group": attn_group,
        }

        if not self.is_sdxl:
            for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("input", id))
                patch_kwargs["number"] += 1
            for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
                set_model_patch_replace(work_model, patch_kwargs, ("output", id))
                patch_kwargs["number"] += 1
            set_model_patch_replace(work_model, patch_kwargs, ("middle", 0))
        else:
            for id in [4,5,7,8]: # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("input", id, index))
                    patch_kwargs["number"] += 1
            for id in range(6): # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(work_model, patch_kwargs, ("output", id, index))
                    patch_kwargs["number"] += 1
            for index in range(10):
                set_model_patch_replace(work_model, patch_kwargs, ("middle", 0, index))
                patch_kwargs["number"] += 1

        return (work_model, )

class PrepImageForClipVision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "crop_position": (["top", "bottom", "left", "right", "center", "pad", "stretch"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter"

    def prep_image(self, image, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
        _, oh, ow, _ = image.shape
        output = image.permute([0,3,1,2])

        if "pad" in crop_position:
            target_length = max(oh, ow)
            pad_l = (target_length - ow) // 2
            pad_r = (target_length - ow) - pad_l
            pad_t = (target_length - oh) // 2
            pad_b = (target_length - oh) - pad_t
            output = F.pad(output, (pad_l, pad_r, pad_t, pad_b), value=0, mode="constant")
        elif "stretch" not in crop_position:
            crop_size = min(oh, ow)
            x = (ow-crop_size) // 2
            y = (oh-crop_size) // 2
            if "top" in crop_position:
                y = 0
            elif "bottom" in crop_position:
                y = oh-crop_size
            elif "left" in crop_position:
                x = 0
            elif "right" in crop_position:
                x = ow-crop_size
            
            x2 = x+crop_size
            y2 = y+crop_size

            # crop
            output = output[:, :, y:y2, x:x2]

        # resize (apparently PIL resize is better than tourchvision interpolate)
        imgs = []
        for i in range(output.shape[0]):
            img = TT.ToPILImage()(output[i])
            img = img.resize((224,224), resample=Image.Resampling[interpolation])
            imgs.append(TT.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
       
        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)
        
        output = output.permute([0,2,3,1])

        return (output,)

class IPAdapterEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip_vision": ("CLIP_VISION",),
            "image_1": ("IMAGE",),
            "ipadapter_plus": ("BOOLEAN", { "default": False }),
            "noise": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
            "weight_1": ("FLOAT", { "default": 1.0, "min": 0, "max": 1.0, "step": 0.01 }),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "weight_2": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01 }),
                "weight_3": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01 }),
                "weight_4": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01 }),
            }
        }

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "preprocess"
    CATEGORY = "ipadapter"

    def preprocess(self, clip_vision, image_1, ipadapter_plus, noise, weight_1, image_2=None, image_3=None, image_4=None, weight_2=1.0, weight_3=1.0, weight_4=1.0):
        image = image_1
        weight = [weight_1]*image_1.shape[0]
        
        if image_2 is not None:
            if image_1.shape[1:] != image_2.shape[1:]:
                image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center").movedim(1,-1)
            image = torch.cat((image, image_2), dim=0)
            weight += [weight_2]*image_2.shape[0]
        if image_3 is not None:
            if image.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center").movedim(1,-1)
            image = torch.cat((image, image_3), dim=0)
            weight += [weight_3]*image_3.shape[0]
        if image_4 is not None:
            if image.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image.shape[2], image.shape[1], "bilinear", "center").movedim(1,-1)
            image = torch.cat((image, image_4), dim=0)
            weight += [weight_4]*image_4.shape[0]
        
        clip_embed = clip_vision.encode_image(image)
        neg_image = image_add_noise(image, noise) if noise > 0 else None
        
        if ipadapter_plus:
            clip_embed = clip_embed.penultimate_hidden_states
            if noise > 0:
                clip_embed_zeroed = clip_vision.encode_image(neg_image).penultimate_hidden_states
            else:
                clip_embed_zeroed = zeroed_hidden_states(clip_vision, image.shape[0])
        else:
            clip_embed = clip_embed.image_embeds
            if noise > 0:
                clip_embed_zeroed = clip_vision.encode_image(neg_image).image_embeds
            else:
                clip_embed_zeroed = torch.zeros_like(clip_embed)

        weight = torch.tensor(weight).unsqueeze(-1) if not ipadapter_plus else torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)
        
        output = (clip_embed, weight, clip_embed_zeroed)

        return ( output, )

class IPAdapterApplyEncoded(IPAdapterApply):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ipadapter": ("IPADAPTER", ),
                "embeds": ("EMBEDS",),
                "model": ("MODEL", ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.01 }),
                "weight_type": (["original", "linear", "channel penalty"], ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "attn_group": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

class IPAdapterSaveEmbeds:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("EMBEDS",),
            "filename_prefix": ("STRING", {"default": "embeds/IPAdapter"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter"

    def save(self, embeds, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}_.ipadpt"
        file = os.path.join(full_output_folder, file)

        torch.save(embeds, file)
        return (None, )


class IPAdapterLoadEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [os.path.relpath(os.path.join(root, file), input_dir) for root, dirs, files in os.walk(input_dir) for file in files if file.endswith('.ipadpt')]
        return {"required": {"embeds": [sorted(files), ]}, }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "ipadapter"

    def load(self, embeds):
        path = folder_paths.get_annotated_filepath(embeds)
        output = torch.load(path).cpu()

        return (output, )
