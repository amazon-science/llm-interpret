import transformers
import torch
import os
import random
import pickle
from lm_eval.base import BaseLM
# from transformers.deepspeed import HfDeepSpeedConfig
# import deepspeed

class HFLM(BaseLM):
    def __init__(
        self,
        device="cuda",
        pretrained="facebook/opt-125m",
        revision="main",
        subfolder=None,
        tokenizer=None,
        batch_size=1,
        model_cache_dir = None,
        tokenizer_cache_dir = None,
        mask_single_head=0,
        mask_heads=0,
        mask_fc=0,
        mask_iterative_fc=0,
        head_percent_mask=0,
        head_importance_path=None,
        fc_percent_mask=0,
        fc_importance_path=None,
    ):
        super().__init__()

        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, int)

        if device:
            if device not in ["cuda", "cpu"]:
                device = int(device)
            self._device = torch.device(device)
            print(f"Using device '{device}'")
        else:
            print("Device not specified")
            print(f"Cuda Available? {torch.cuda.is_available()}")
            self._device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        device_map = 'auto'
        if '66b' in pretrained:
            device_map = eval(open('66b_device_map.txt', 'r').readlines()[0])
            for key in device_map:
                if 'layers' in key:
                    layer_num = int(key.split('.')[-1])
                    if layer_num <= 3:
                        device_map[key] = 0
                    elif layer_num >= 60:
                        device_map[key] = 'cpu'
                    else:    
                        device_map[key] = ((layer_num - 4) // 8) + 1

        self.opt = transformers.AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map = device_map,
            cache_dir = model_cache_dir,
            torch_dtype=torch.float16
        )

        self.opt.get_decoder().embed_tokens.weight.requires_grad = False
        self.opt.get_decoder().embed_positions.weight.requires_grad = False
        # self.opt = transformers.AutoModelForCausalLM.from_pretrained(
        #     pretrained,
        #     cache_dir = model_cache_dir,
        # )
        # self.ds_engine = deepspeed.initialize(model=self.opt, model_parameters = self.opt.parameters(), config_params=ds_config)[0]
        # self.ds_engine.module.eval()  # inference

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained,
            cache_dir = tokenizer_cache_dir if tokenizer_cache_dir else 'tokenizer_cache/',
            use_fast = False
        ) if tokenizer is None else tokenizer

        assert isinstance(
            self.tokenizer,
            (
                transformers.GPT2Tokenizer,
                transformers.GPT2TokenizerFast,
                transformers.T5Tokenizer,
                transformers.T5TokenizerFast,
            ),
        ), "this tokenizer has not been checked for compatibility yet!"

        self.vocab_size = self.tokenizer.vocab_size

        # multithreading and batching
        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        num_hidden_layers = self.opt.config.num_hidden_layers
        num_heads = self.opt.config.num_attention_heads
        self.head_mask = torch.ones(num_hidden_layers * num_heads, dtype = torch.half)
        self.fc_mask = torch.ones(num_hidden_layers, dtype = torch.half)

        if int(mask_heads):
            with open(head_importance_path, 'rb') as f:
                importance = pickle.load(f)
            _, head_indices = torch.sort(importance.view(-1))
            head_indices = list(head_indices.numpy())
            head_indices = head_indices[: int(head_percent_mask) * len(head_indices) // 100]
            self.head_mask[head_indices] = 0. 
        elif int(mask_single_head): #Only performing it on OPT125M
            self.head_mask[int(mask_single_head)-1] = 0.

        self.head_mask = self.head_mask.view(num_hidden_layers, num_heads).contiguous()
        
        if mask_fc:
            self.fc_mask[int(mask_fc)] = 0.
        elif int(mask_iterative_fc):
            with open(fc_importance_path, 'rb') as f:
                fc_indices = list(pickle.load(f))
            fc_indices = fc_indices[: int(fc_percent_mask) * len(fc_indices) // 100] 
            self.fc_mask[fc_indices] = 0.


    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.opt.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def get_tokenizer(self):
        return self.tokenizer

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps, attn_mask = None, labels = None):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """

        if labels == None:
            with torch.no_grad():
                return self.opt(input_ids = inps, head_mask = self.head_mask, fc_mask = self.fc_mask)[0][:, :, :50265]
                # rank = int(os.getenv("LOCAL_RANK", "0"))
                # return self.ds_engine.module(input_ids = inps, head_mask = self.head_mask.to(rank))[0][:, :, :50265]
        else:
            return self.opt(input_ids = inps, attention_mask = attn_mask, labels = labels).loss
            # return self.ds_engine.module(input_ids = inps, attention_mask = attn_mask, labels = labels).loss
            
    def _model_generate(self, context, max_length, eos_token_id):
        return self.opt.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )


# for backwards compatibility
OPTLM = HFLM