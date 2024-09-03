
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.functional as F
from constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX, QUESTION_TOKEN_INDEX
import random,math
from Qformer import BertConfig, BertLMHeadModel
from blip2 import Blip2Base, disabled_train
import einops
import contextlib

class MLLMCatConfig(LlamaConfig):
    model_type = "MLLMCat"


class Perciver(nn.Module):

    def __init__(self):
        super().__init__()
        self.attn_a = nn.MultiheadAttention(1024,4,0.1,batch_first=False)
        self.attn_v = nn.MultiheadAttention(1024,4,0.1,batch_first=False)

        self.attn_t2a = nn.MultiheadAttention(4096,4,0.1,batch_first=False)
        self.attn_t2v = nn.MultiheadAttention(4096,4,0.1,batch_first=False)

        self.attn_a2t = nn.MultiheadAttention(4096,4,0.1,batch_first=False)
        self.attn_v2t = nn.MultiheadAttention(4096,4,0.1,batch_first=False)

        self.fc_t2a = nn.Linear(1024,4096)
        self.fc_t2v = nn.Linear(1024,4096)
        self.fc_a2t = nn.Linear(4096,1024)
        self.fc_v2t = nn.Linear(4096,1024)

        self.LayerNorm1 = nn.LayerNorm(4096)
        self.LayerNorm2 = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.1)


    def forward(
        self,
        txt_embedding=None,
        cur_image_features=None,
        cur_audio_features=None,
    ):
        cur_image_features = self.LayerNorm1(self.dropout(self.fc_t2v(self.attn_v(cur_image_features,cur_image_features,cur_image_features, attn_mask=None, key_padding_mask=None)[0])))
        cur_audio_features = self.LayerNorm1(self.dropout(self.fc_t2a(self.attn_a(cur_audio_features,cur_audio_features,cur_audio_features, attn_mask=None, key_padding_mask=None)[0])))
        visual_ground = F.tanh(self.attn_t2v(txt_embedding,cur_image_features,cur_image_features, attn_mask=None, key_padding_mask=None)[0])
        audio_ground = F.tanh(self.attn_t2a(txt_embedding,cur_audio_features,cur_audio_features, attn_mask=None, key_padding_mask=None)[0])

        new_visual_ground = self.LayerNorm2(self.dropout(self.fc_v2t(F.tanh(self.attn_v2t(cur_image_features,visual_ground,visual_ground, attn_mask=None, key_padding_mask=None)[0]))))
        new_audio_ground = self.LayerNorm2(self.dropout(self.fc_a2t(F.tanh(self.attn_a2t(cur_audio_features,audio_ground,audio_ground, attn_mask=None, key_padding_mask=None)[0]))))

        return new_visual_ground, new_audio_ground


class CA(Blip2Base):
    def __init__(self, num_query_token = 32, num_features = 1408, q_former_model="/home/qilang/PythonProjects/Video-LLaMA-main/ckpt/blip2_pretrained_flant5xxl.pth"):
              
        super(CA, self).__init__()

        self.perceiver = Perciver()

        self.qf_proj_video = nn.Linear(
            1024, num_features
        )
        self.qf_proj_audio = nn.Linear(
            1024, num_features
        )

        self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, num_features,
            )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        
        self.load_from_pretrained(url_or_filename=q_former_model)
        self.audio_Qformer, self.audio_query_tokens = self.init_Qformer(
                num_query_token, num_features,
            )
        self.audio_Qformer.cls = None
        self.audio_Qformer.bert.embeddings.word_embeddings = None
        self.audio_Qformer.bert.embeddings.position_embeddings = None
        for layer in self.audio_Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)
        
        self.llama_proj_video = nn.Linear(
            self.Qformer.config.hidden_size, 4096
        )
        self.llama_proj_audio = nn.Linear(
            self.audio_Qformer.config.hidden_size, 4096
        )

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def init_Qformer(cls, num_query_token, vision_width,num_hidden_layers =4):
        encoder_config = BertConfig.from_pretrained("/home/qilang/bert-base-uncased/", local_files_only=True)
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens
    
    def forward(self, txt_embeds, image_embeds, audio_embeds):
        # Q-former
        device = image_embeds.device

        image_embeds, audio_embeds = self.perceiver(txt_embeds,image_embeds,audio_embeds)

        image_embeds = self.qf_proj_video(image_embeds).unsqueeze(0)
        audio_embeds = self.qf_proj_audio(audio_embeds).unsqueeze(0)
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens
            query_tokens = query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            inputs_llama_video = self.llama_proj_video(query_output.last_hidden_state)

            audio_atts = torch.ones(audio_embeds.size()[:-1], dtype=torch.long).to(device)

            audio_query_tokens = self.audio_query_tokens
            audio_query_tokens = audio_query_tokens.expand(audio_embeds.shape[0], -1, -1)
            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens,
                encoder_hidden_states=audio_embeds,
                encoder_attention_mask=audio_atts,
                return_dict=True,
            )
            inputs_llama_audio = self.llama_proj_audio(audio_query_output.last_hidden_state)

        return inputs_llama_video,inputs_llama_audio

class MLLMCatLlamaModel(LlamaModel):
    config_class = MLLMCatConfig

    def __init__(self, config: LlamaConfig):
        super(MLLMCatLlamaModel, self).__init__(config)
        
    def initialize_av_modules(self, model_args):
        pretrain_CA = model_args.pretrain_CA
        pretrain_mm_mlp_adapter_v = model_args.pretrain_mm_mlp_adapter_v
        pretrain_mm_mlp_adapter_a = model_args.pretrain_mm_mlp_adapter_a

        if not hasattr(self, 'mm_projector_v'):
            self.mm_projector_v = nn.Linear(1024, self.config.hidden_size)
        if not hasattr(self, 'mm_projector_a'):
            self.mm_projector_a = nn.Linear(1024, self.config.hidden_size)

        if not hasattr(self, 'CA'):
            self.CA = CA()

        if pretrain_mm_mlp_adapter_v is not None and pretrain_mm_mlp_adapter_a is not None:
            mm_projector_weights_v = torch.load(pretrain_mm_mlp_adapter_v, map_location='cpu')
            mm_projector_weights_a = torch.load(pretrain_mm_mlp_adapter_a, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector_v.load_state_dict(get_w(mm_projector_weights_v, 'mm_projector_v'))
            self.mm_projector_a.load_state_dict(get_w(mm_projector_weights_a, 'mm_projector_a'))
    
        if pretrain_CA is not None:
            CA_weights = torch.load(pretrain_CA, map_location='cpu')
            def load_ca(weights):
                new_weights_set = {}
                for k, v in weights.items():
                    new_k = k.replace('base_model.model.model.CA.','')
                    new_weights_set.update({new_k:v})
                checkpoint = torch.load('/home/qilang/PythonProjects/Video-LLaMA-main/ckpt/blip2_pretrained_flant5xxl.pth', map_location="cpu")
                state_dict = checkpoint["model"]
                new_weights_set.update({'Qformer.bert.embeddings.position_ids':state_dict['Qformer.bert.embeddings.position_ids']})
                new_weights_set.update({'audio_Qformer.bert.embeddings.position_ids':state_dict['Qformer.bert.embeddings.position_ids']})
                return new_weights_set

            self.CA.load_state_dict(load_ca(CA_weights))
            print('load success')

class MLLMCatLlamaForCausalLM(LlamaForCausalLM):
    config_class = MLLMCatConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = MLLMCatLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, audios
    ):
        if past_key_values is not None and images is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels
                
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0: #如果里面没有image token
                # multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                # cur_input_embeds = cur_input_embeds + (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0] #找到image token的位置（索引）
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            #Multi-Inputs
            cur_image_features = images[cur_image_idx]
            cur_audio_features = audios[cur_image_idx]
            # image_size = images[cur_image_idx].size(0)
            # visual_ground = F.tanh(self.get_model().attn_a2v(cur_image_features,cur_audio_features,cur_audio_features, attn_mask=None, key_padding_mask=None)[0])
            # deeped_audio = cur_audio_features.repeat(image_size,1) + visual_ground

            global_visual = self.get_model().mm_projector_v(cur_image_features.mean(0).unsqueeze(0))
            global_audio = self.get_model().mm_projector_a(cur_audio_features.mean(0).unsqueeze(0))

            #Q-former
            question_token_start = torch.where(cur_input_ids == QUESTION_TOKEN_INDEX)[0][0] #找到question token的位置（索引）
            question_token_end = torch.where(cur_input_ids == QUESTION_TOKEN_INDEX)[0][1]
            txt_embedding = self.get_model().embed_tokens(cur_input_ids[question_token_start+1:question_token_end])
            # # txt_embedding = self.get_model().shared_projector(txt_embedding)
            cur_image_features = images[cur_image_idx]
            cur_audio_features = audios[cur_image_idx]
            inputs_llama_video,inputs_llama_audio = self.get_model().CA(txt_embedding, cur_image_features,cur_audio_features)
            inputs_llama_video = inputs_llama_video.to(torch.bfloat16).squeeze(0)
            inputs_llama_audio = inputs_llama_audio.to(torch.bfloat16).squeeze(0)
            
            while image_token_indices.numel() > 0: #当有image token时 (返回token的个数)

                image_token_start = image_token_indices[0] #image token开始的地方

                #新加方法需要remove -300
                new_tensor1 = torch.cat((cur_input_ids[:question_token_start], cur_input_ids[question_token_start+1:question_token_end]))
                new_tensor2 = torch.cat((new_tensor1,cur_input_ids[question_token_end+1:]))
                cur_input_ids = new_tensor2
                if labels is not None:
                    new_tensor1 = torch.cat((cur_labels[:question_token_start], cur_labels[question_token_start+1:question_token_end]))
                    new_tensor2 = torch.cat((new_tensor1,cur_labels[question_token_end+1:]))
                    cur_labels = new_tensor2

                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-2]))
                # cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start])) #token转embedding （image token之前的）
                cur_new_input_embeds.append(inputs_llama_video) #中间插入ground特征
                cur_new_input_embeds.append(inputs_llama_audio)
                cur_new_input_embeds.append(global_visual)
                cur_new_input_embeds.append(global_audio)

                if labels is not None:
                    cur_new_labels.append(cur_labels[:image_token_start-2]) #文本的遮罩
                    cur_new_labels.append(torch.full((inputs_llama_video.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) 
                    cur_new_labels.append(torch.full((inputs_llama_audio.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) 
                    cur_new_labels.append(torch.full((global_visual.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) 
                    cur_new_labels.append(torch.full((global_audio.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype)) 
                    #label也插上全未-100的tensor 与input对应 input的列表0：145，4096 1：60，4096 则label的列表0：145 1：60
                    cur_labels = cur_labels[image_token_start-1:] #剩下的114个-100
                cur_image_idx += 1
                cur_input_ids = cur_input_ids[image_token_start-1:] #image token后半段
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids)) #image token后半段的token转embedding
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds] #整个embeding输入 分为3段 第一段为image前文本 第二段为image 第三段为image后文本
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0) #最后全部整合
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def get_logits(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images,audios)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return logits,loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        audios: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images,audios)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "audios": kwargs.get("audios", None),
            }
        )
        return model_inputs

AutoConfig.register("MLLMCat", MLLMCatConfig)
AutoModelForCausalLM.register(MLLMCatConfig, MLLMCatLlamaForCausalLM)