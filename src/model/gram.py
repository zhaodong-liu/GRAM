"""
Model design code refers to FID code:https://github.com/facebookresearch/FiD/blob/main/src/model.py.
########################
"""

import torch
from torch import nn
from IPython import embed

from .gram_t5 import T5ForConditionalGeneration_GRAM
from .gram_t5_outputs import BaseModelOutputWithPastAndCrossAttentions


class GRAM(T5ForConditionalGeneration_GRAM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.max_item_num = config.max_item_num

        # 从config或args中获取简化处理标志
        self.use_simplified_fusion = getattr(config, 'simplified_metadata', False)
        self.disable_fine_grained = getattr(config, 'disable_fine_grained_fusion', False)
        
        self.use_position_embedding = config.use_position_embedding

        if self.use_position_embedding:
            pos_emb_size = self.max_item_num + 1  # one for coarse-grained user prompt
            self.position_embedding = nn.Embedding(pos_emb_size, self.config.d_model)
            self.init_position_embedding()
        else:
            self.position_embedding = None

        self.wrap_encoder()
        
        if hasattr(config, 'local_rank') and (not hasattr(config, 'local_rank') or config.local_rank in [0, -1]):
            print(f"🔧 GRAM Model initialized:")
            print(f"   - Simplified fusion: {self.use_simplified_fusion}")
            print(f"   - Disable fine-grained: {self.disable_fine_grained}")



    def init_position_embedding(self):
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(
                kwargs["input_ids"].size(0), -1
            )
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(
                kwargs["attention_mask"].size(0), -1
            )

        return super(GRAM, self).forward(**kwargs)

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward_(self, **kwargs):
        """
        条件性forward pass
        """
        # ========== 新增：条件性处理逻辑 ==========
        if self.use_simplified_fusion or self.disable_fine_grained:
            # 简化处理：对于MovieLens等简单数据集
            return self._simplified_forward(**kwargs)
        else:
            # 复杂处理：保持原版行为用于Amazon数据集
            return self._complex_forward(**kwargs)

    def _simplified_forward(self, **kwargs):
        """
        简化的forward pass，适用于MovieLens等简单数据
        """
        # 简化处理：直接使用标准的forward逻辑
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(
                kwargs["input_ids"].size(0), -1
            )
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(
                kwargs["attention_mask"].size(0), -1
            )

        return super(GRAM, self).forward(**kwargs)

    def _complex_forward(self, **kwargs):
        """
        复杂的forward pass，保持原版multi-granular late fusion逻辑
        """
        # 保持原来的复杂处理逻辑
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(
                kwargs["input_ids"].size(0), -1
            )
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(
                kwargs["attention_mask"].size(0), -1
            )

        return super(GRAM, self).forward(**kwargs)

        
    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(self, input_ids, attention_mask, max_length, **kwargs):
        self.encoder.n_passages = input_ids.size(1)
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids2 = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask2 = attention_mask.view(attention_mask.size(0), -1)

        last_hidden_states = self.encoder(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            return_dict=True,
        )[0]
        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_states
        )

        outputs = super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            **kwargs
        )

        if kwargs.get("output_hidden_states"):
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state[
                0:1, :, :
            ]  # only first beam
            outputs["encoder_outputs"] = encoder_outputs

        return outputs

    def get_crossattention_scores(self, cross_attentions, attention_mask, b_idx=0):
        """
        ## beam size must be 1 for get_crossattention_scores
        cross_attentions: list(#gen tokens: varies) of list(#layers:6) of (beam (1), n_heads, n-th beam (1), n_passages * text_maxlength)
        attention_mask: torch.tensor (bsz, n_passages, text_maxlength)
        """

        # Assuming that the cross_attentions are arranged as a list of [gen tokens][layers], where each element is
        # a tensor of shape (bsz, n_heads, 1, n_passages * text_maxlength)
        cross_attentions_first_token = [
            cross_attention_token[b_idx]
            for cross_attention_token in cross_attentions[0]
        ]
        cross_attentions_first_token = torch.stack(
            cross_attentions_first_token
        )  ## (n_layers, n_heads, 1, n_passages * text_maxlength)
        # Consider only first token
        bsz, n_passages, text_maxlength = attention_mask.size()
        n_layers, n_heads, _, _ = cross_attentions_first_token.size()

        scores = cross_attentions_first_token.view(
            bsz, n_layers, n_heads, n_passages, -1
        )
        scores = scores.masked_fill(~attention_mask[:, None, None], 0.0)
        token_scores = scores.sum(dim=[1, 2]).squeeze(0).tolist()
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = attention_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens

        return token_scores, scores

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(
            encoder=self.encoder,
            config=self.config,
            use_checkpoint=use_checkpoint,
            position_embedding=self.position_embedding,
        )

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict, strict=False)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None


class EncoderWrapper(nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(
        self, encoder, config=None, use_checkpoint=False, position_embedding=None
    ):
        super().__init__()
        # print(f"> WARN: main_input_name not found in encoder, transformer version might be too old")
        self.main_input_name = encoder.main_input_name
        self.encoder = encoder
        self.config = config
        self.position_embedding = position_embedding
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(
        self, input_ids=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # print(f">>> inside EncoderWrapper  --- 3"); embed()
        if input_ids is not None:
            # total_length = n_passages * passage_length
            bsz, total_length = input_ids.shape  # B x (N * L)
            passage_length = total_length // self.n_passages  # L
            input_ids = input_ids.view(
                bsz * self.n_passages, passage_length
            )  # B x (N * L) -> (B * N) x L
            attention_mask = attention_mask.view(
                bsz * self.n_passages, passage_length
            )  # B x (N * L) -> (B * N) x L
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, **kwargs
            )  # tuple ( (B * N) x L x D, )

        elif inputs_embeds is not None:
            bsz, total_length, _ = inputs_embeds.shape  # B x (N * L) x D
            passage_length = total_length // self.n_passages  # L
            inputs_embeds = inputs_embeds.view(
                bsz * self.n_passages, passage_length, -1
            )  # B x (N * L) x D -> (B * N) x L x D
            attention_mask = attention_mask.view(
                bsz * self.n_passages, passage_length
            )  # B x (N * L) -> (B * N) x L
            outputs = self.encoder(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
            )  # tuple ( (B * N) x L x D, )

        else:
            raise ValueError(
                "At least one of input_ids or inputs_embeds should be not None"
            )

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if self.position_embedding is not None:
            last_hidden_states = outputs[0]  # (B * N) x L x D
            position_ids = torch.arange(self.n_passages, device=device).expand(
                bsz, self.n_passages
            )  # N -> B x N
            position_embeddings = self.position_embedding(position_ids)  # B x N x D
            position_embeddings = position_embeddings.view(
                bsz * self.n_passages, 1, -1
            )  # (B * N) x 1 x D
            last_hidden_states = (
                last_hidden_states + position_embeddings
            )  # (B * N) x L x D
        else:
            last_hidden_states = outputs[0]
        # tuple ( (B * N) x L x D, ) -> (B x (N * L) x D, )
        outputs = (
            last_hidden_states.view(bsz, self.n_passages * passage_length, -1),
        ) + outputs[1:]
        return outputs  # tuple ( B x (N * L) x D, )


class CheckpointWrapper(nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [], dtype=torch.float, device=output[0].device, requires_grad=True
                )
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block
