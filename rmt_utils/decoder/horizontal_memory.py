import torch
from typing import Union, Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast

def gpt_neox_horizontal_memory_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    head_mask: Optional[torch.FloatTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    rmt_parent=None
) -> Union[Tuple, BaseModelOutputWithPast]:
    r"""
    past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
        Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
        If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
        don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
        `decoder_input_ids` of shape `(batch_size, sequence_length)`.
    use_cache (`bool`, *optional*):
        If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
        `past_key_values`).
    """
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    batch_size, seq_length = input_shape

    if past_key_values is None:
        past_length = 0
        past_key_values = tuple([None] * self.config.num_hidden_layers)
    else:
        past_length = past_key_values[0][0].size(-2)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    # Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
    # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
    head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

    if inputs_embeds is None:
        inputs_embeds = self.embed_in(input_ids)

    hidden_states = self.emb_dropout(inputs_embeds)

    if self.gradient_checkpointing and self.training:
        if use_cache:
            # logger.warning(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            # )
            use_cache = False

    presents = () if use_cache else None
    all_attentions = () if output_attentions else None
    all_hidden_states = () if output_hidden_states else None
    for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            raise NotImplementedError
            # def create_custom_forward(module):
            #     def custom_forward(*inputs):
            #         # None for layer_past
            #         return module(*inputs, use_cache, None, output_attentions)

            #     return custom_forward

            # outputs = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(layer),
            #     hidden_states,
            #     attention_mask,
            #     position_ids,
            #     head_mask[i],
            # )
        else:
            num_mem_tokens = rmt_parent.memory_cell.num_mem_tokens
            if i in rmt_parent.memory_storage:
                layer_memory = rmt_parent.memory_storage[i]
                if layer_memory.shape[0] == 1:
                    raise NotImplementedError
                    # layer_memory = layer_memory.repeat(hidden_states.shape[0], 1, 1)

                hidden_states[:, :num_mem_tokens] = layer_memory
                    
            outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                head_mask=head_mask[i],
                layer_past=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
        hidden_states = outputs[0]

        ### set layer memory 
        ### for all layers except the first one
        if i > 0:
            rmt_parent.memory_storage[i] = hidden_states[:, -num_mem_tokens:]

        if use_cache is True:
            presents = presents + (outputs[1],)
        if output_attentions:
            all_attentions = all_attentions + (outputs[2 if use_cache else 1],)


    hidden_states = self.final_layer_norm(hidden_states)

    # Add last hidden state
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=presents,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
    )