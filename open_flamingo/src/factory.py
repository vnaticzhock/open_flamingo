from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import hf_hub_download
import open_clip 
import torch
from open_clip import transformer
import torch.nn.functional as F

from .flamingo import Flamingo
from .flamingo_lm import FlamingoLMMixin
from .utils import extend_instance

def LNormforward(self, x: torch.Tensor):
    print('substitue layer norm to use fp16')
    return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

transformer.LayerNormFp32.forward = LNormforward

def create_model_and_transforms(
    clip_vision_encoder_path: str,
    clip_vision_encoder_pretrained: str,
    lang_encoder_path: str,
    tokenizer_path: str,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    inference: bool = False,
    precision: str = 'auto', ## 'auto', 'fp16', 'int8'
    device: str = 'cpu', ## 'cpu', 'cuda'
    checkpoint_path: str = None,
    **flamingo_kwargs,
):
    """
    Initialize a Flamingo model from a pretrained vision encoder and language encoder.
    Appends special tokens to the tokenizer and freezes backbones.

    Args:
        clip_vision_encoder_path (str): path to pretrained clip model (e.g. "ViT-B-32")
        clip_vision_encoder_pretrained (str): name of pretraining dataset for clip model (e.g. "laion2b_s32b_b79k")
        lang_encoder_path (str): path to pretrained language encoder
        tokenizer_path (str): path to pretrained tokenizer
        cross_attn_every_n_layers (int, optional): determines how often to add a cross-attention layer. Defaults to 1.
        use_local_files (bool, optional): whether to use local files. Defaults to False.
        decoder_layers_attr_name (str, optional): name of the decoder layers attribute. Defaults to None.
    Returns:
        Flamingo: Flamingo model from pretrained vision and language encoders
        Image processor: Pipeline to preprocess input images
        Tokenizer: A tokenizer for the language model
    """
    print('1/4 Loading open clip...')
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained,
        precision=precision, device=device,
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    print('2/4 Loading tokenizer...')
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=use_local_files
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    torch_dtype = torch.float32
    quantization_config = None

    if precision == 'fp16':
        torch_dtype = torch.float16
    elif precision == 'int8':
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    ## language encoder fp16
    print('3/4 Loading... causal lm')
    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path, local_files_only=use_local_files,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config
    )

    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    print('4/4 Loading... flamingo')
    model = Flamingo(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    # Unfreeze perceiver, gated_cross_attn_layers, and LM input embeddings
    model.perceiver.requires_grad_(not inference)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(not inference)
    model.lang_encoder.get_input_embeddings().requires_grad_(not inference)

    if checkpoint_path is not None:
        checkpoint_path = hf_hub_download(checkpoint_path, 'checkpoint.pt')
        model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    else:
        print("WARNING: No checkpoint path provided. Initializing model randomly.")

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    ## try to force everything in fp16 and gpu
    model.perceiver = model.perceiver.to(device)
    model.lang_encoder = model.lang_encoder.to(device)
    model.to(device)
    if precision == 'fp16':
        model.vision_encoder = model.vision_encoder.half() # this remains on fp32, don't know why
        model.lang_encoder = model.lang_encoder.half()
        model.perceiver = model.perceiver.half()

    return model, image_processor, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
}
