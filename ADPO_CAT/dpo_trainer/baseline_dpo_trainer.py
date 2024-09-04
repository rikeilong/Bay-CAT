import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from transformers import PreTrainedModel

from .base_dpo_trainer import BaseDPOTrainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Dpo_Trainer(BaseDPOTrainer):
    
    def dpo_loss(
        self,
        policy_logps,
        ref_logps,
        beta: float,
        reference_free: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        policy_chosen_logps, policy_rejected_logps = policy_logps.chunk(2, dim=0)
        reference_chosen_logps, reference_rejected_logps = ref_logps.chunk(2, dim=0)
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        if reference_free:
            ref_logratios = 0

        logits = pi_logratios - ref_logratios

        losses = -F.logsigmoid(beta * logits)
        chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards, \
                policy_chosen_logps, policy_rejected_logps, \
                reference_chosen_logps, reference_rejected_logps
    
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        total_loss = 0
        for batch_idx in range(len(inputs["images"])):
            images = inputs["images"][batch_idx].unsqueeze(0)
            audios = inputs['audios'][batch_idx].unsqueeze(0)
            rejected_input_ids = inputs["input_ids"][batch_idx].unsqueeze(0)
            rejected_labels = inputs["labels"][batch_idx].unsqueeze(0)
            rejected_attention_mask = inputs['attention_mask'][batch_idx].unsqueeze(0)
            chosen_input_ids = inputs["chosen_input_ids"][batch_idx].unsqueeze(0)
            chosen_labels = inputs["chosen_labels"][batch_idx].unsqueeze(0)
            chosen_attention_mask = inputs['chosen_attention_mask'][batch_idx].unsqueeze(0)

            lens = abs(len(rejected_input_ids[0]) - len(chosen_input_ids[0]))
            l_s = torch.zeros(lens).to(torch.int8).unsqueeze(0).to(images.device)
            att_s = l_s.bool().to(images.device)
            if len(rejected_input_ids[0]) > len(chosen_input_ids[0]):
                chosen_input_ids = torch.cat([chosen_input_ids,l_s],dim=1)
                chosen_labels = torch.cat([chosen_labels,l_s],dim=1)
                chosen_attention_mask = torch.cat([chosen_attention_mask,att_s],dim=1)
            else:
                rejected_input_ids = torch.cat([rejected_input_ids,l_s],dim=1)
                rejected_labels = torch.cat([rejected_labels,l_s],dim=1)
                rejected_attention_mask = torch.cat([rejected_attention_mask,att_s],dim=1)

            total_input_ids = torch.cat([rejected_input_ids,chosen_input_ids],dim=0)
            total_labels = torch.cat([rejected_labels,chosen_labels],dim=0)
            total_attention_mask = torch.cat([rejected_attention_mask,chosen_attention_mask],dim=0)


            images ,audios = images.to(torch.bfloat16).repeat(2,1,1),audios.to(torch.bfloat16).repeat(2,1,1)
        
            policy_logits,loss = self.model.get_logits(
                input_ids = total_input_ids,
                attention_mask = total_attention_mask,
                labels = total_labels,
                images = images,
                audios = audios
            )  # [B, L, V]

            policy_logps = _get_batch_logps(
                policy_logits,
                total_labels.long(),
                False
            )

            with torch.no_grad():
                ref_logits,loss = self.ref_model.get_logits(
                    input_ids = total_input_ids,
                    attention_mask = total_attention_mask,
                    labels = total_labels,
                    images = images,
                    audios = audios
                )  # [B, L, V]

                ref_logps = _get_batch_logps(
                    ref_logits,
                    total_labels.long(),
                    False
                )
                
            losses, chosen_rewards, rejected_rewards, \
            policy_chosen_logps, policy_rejected_logps, \
            reference_chosen_logps, reference_rejected_logps = self.dpo_loss(
                policy_logps=policy_logps, ref_logps=ref_logps, beta=self.beta, reference_free=False
            )
            loss = losses.mean()
            total_loss += loss
        
        loss = total_loss / 2

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        prefix, metrics = "train_", {}
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().mean()
        metrics[f"policy_{prefix}logps/rejected"] = policy_rejected_logps.detach().cpu().mean()
        metrics[f"policy_{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/rejected"] = reference_rejected_logps.detach().cpu().mean()
        metrics[f"referece_{prefix}logps/chosen"] = reference_chosen_logps.detach().cpu().mean()
        
        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss


def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    # if logits.shape[:-1] != labels.shape:
    # if logits.shape[1] != labels.shape:
    #     raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    label_pad_token_id = -100
    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)