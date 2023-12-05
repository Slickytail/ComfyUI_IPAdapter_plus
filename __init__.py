from .IPAdapterPlus import IPAdapterModelLoader, IPAdapterApply, IPAdapterApplyEncoded, PrepImageForClipVision
from .IPAdapterPlus import IPAdapterEncoder, IPAdapterSaveEmbeds, IPAdapterLoadEmbeds

NODE_CLASS_MAPPINGS = {
    "IPAdapterModelLoader": IPAdapterModelLoader,
    "IPAdapterApply": IPAdapterApply,
    "IPAdapterApplyEncoded": IPAdapterApplyEncoded,
    "PrepImageForClipVision": PrepImageForClipVision,
    "IPAdapterEncoder": IPAdapterEncoder,
    "IPAdapterSaveEmbeds": IPAdapterSaveEmbeds,
    "IPAdapterLoadEmbeds": IPAdapterLoadEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IPAdapterModelLoader": "Load IPAdapter Model",
    "IPAdapterApply": "Apply IPAdapter",
    "IPAdapterApplyEncoded": "Apply IPAdapter from Encoded",
    "PrepImageForClipVision": "Prepare Image For Clip Vision",
    "IPAdapterEncoder": "Encode IPAdapter Image",
    "IPAdapterSaveEmbeds": "Save IPAdapter Embeds",
    "IPAdapterLoadEmbeds": "Load IPAdapter Embeds",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
