"""
Multimodal ASD Classification Model
This module implements a multimodal neural network for ASD classification using both text and audio data.
"""

from transformers import RobertaForSequenceClassification, RobertaTokenizerFast

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
_HIDDEN_STATES_START_POSITION = 1

class WhisperEncoder(transformers.WhisperForAudioClassification):
    """Whisper encoder for audio feature extraction."""

    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
    

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if self.config.use_weighted_layer_sum:
            output_hidden_states = True
        elif output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        if self.config.use_weighted_layer_sum:
            hidden_states = encoder_outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = encoder_outputs[0]

        hidden_states = self.projector(hidden_states)
        pooled_output = hidden_states.mean(dim=1)

        return pooled_output



class AttentionWhisper(nn.Module):
    """Attention mechanism for audio processing."""
    def __init__(self):
        super(AttentionWhisper, self).__init__()
        self.M = 256 
        self.L = 256
        self.ATTENTION_BRANCHES = 1

        self.feature_extractor = WhisperEncoder.from_pretrained("openai/whisper-base")


        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L), # matrix V
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES) # matrix w (or vector w if self.ATTENTION_BRANCHES==1)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.M*self.ATTENTION_BRANCHES, 1),
        #     nn.Sigmoid()
        # )


    def forward(self, x):


        H=[]
        for  bag in x:
            H.append(self.feature_extractor(bag))
        H = torch.stack(H)
        A = self.attention(H)  # KxATTENTION_BRANCHES
        A = F.softmax(A, dim=1)  # softmax over K


        A = A.transpose(1,2)  # ATTENTION_BRANCHESxK
        Z = torch.bmm(A, H)  # ATTENTION_BRANCHESxM

        # Y_prob = self.classifier(Z)
        # Y_hat = torch.ge(Y_prob, 0.5).float()

        return Z

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A


class RobertaClassificationHead(nn.Module):
    """Classification head for RoBERTa model."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x



class MultimodalASDClassifier(RobertaForSequenceClassification):
    """Main multimodal classifier for ASD detection."""
    
    def __init__(self, config):
        super().__init__(config)

        self.audio_feature_extractor = AttentionWhisper()
        config.hidden_size=config.hidden_size+self.audio_feature_extractor.M
        self.classifier = RobertaClassificationHead(config)

    def audio_feature_extractor_load_pretrained(self, ):
        self.audio_feature_extractor = AttentionWhisper()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_features = self.audio_feature_extractor(audio_data)

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        # print(audio_features.shape)
        sequence_output = torch.cat((sequence_output[:,0,:], audio_features[:,0,:]), dim=1)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    



class CustomSequenceClassifierOutput(SequenceClassifierOutput):
    def __init__(self, loss=None, ce_loss=None, uncertainty_loss=None,language_delay_logits=None, language_delay_loss=None,logits=None,variance_text=None,variance_audio=None, variance=None, hidden_states=None,uncertainty_text_loss=None,uncertainty_audio_loss=None, attentions=None):
        super().__init__(loss=loss, logits=logits, hidden_states=hidden_states, attentions=attentions)
        self.ce_loss = ce_loss
        self.uncertainty_loss = uncertainty_loss
        self.variance = variance
        self.uncertainty_text_loss = uncertainty_text_loss
        self.variance_text = variance_text
        self.uncertainty_loss = uncertainty_audio_loss
        self.variance = variance_audio
        self.language_delay_loss = language_delay_loss
        self.language_delay_logits = language_delay_logits  




class CustomRobertaUncertainty(RobertaForSequenceClassification):
    
    def __init__(self,config):
        super().__init__(config)

        self.audio_feature_extractor = AttentionWhisper()
        config.hidden_size=config.hidden_size+self.audio_feature_extractor.M
        self.classifier = RobertaClassificationHead(config)
        self.variance_head = nn.Linear(config.hidden_size, 1)

    def audio_feature_extractor_load_pretrained(self, ):
        self.audio_feature_extractor = AttentionWhisper()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_features = self.audio_feature_extractor(audio_data)

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        # print(audio_features.shape)
        sequence_output = torch.cat((sequence_output[:,0,:], audio_features[:,0,:]), dim=1)
        logits = self.classifier(sequence_output)
        variance = torch.exp(self.variance_head(sequence_output))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                uncertainty_loss = 0.5 * torch.log(variance) + 0.5 / variance
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = ce_loss+ uncertainty_loss.mean()
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                uncertainty_loss = 0.5 * torch.log(variance) + 0.5 / variance
                ce_loss = loss_fct(logits, labels)
                loss = ce_loss+ uncertainty_loss.mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CustomSequenceClassifierOutput(
            loss=loss,
            ce_loss=ce_loss,
            uncertainty_loss=uncertainty_loss,
            logits=logits,
            variance=variance,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )








class CustomRobertaUncertainty2(RobertaForSequenceClassification):
    
    def __init__(self,config):
        super().__init__(config)

        self.audio_feature_extractor = AttentionWhisper()
        self.variance_text_head = nn.Linear(config.hidden_size, 1)
        self.variance_audio_head = nn.Linear(self.audio_feature_extractor.M, 1)
        config.hidden_size=config.hidden_size+self.audio_feature_extractor.M
        self.classifier = RobertaClassificationHead(config)


    def audio_feature_extractor_load_pretrained(self, ):
        self.audio_feature_extractor = AttentionWhisper()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_features = self.audio_feature_extractor(audio_data)

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        # print(audio_features.shape)
        variance_text = torch.exp(self.variance_text_head(sequence_output[:,0,:]))
        variance_audio = torch.exp(self.variance_audio_head( audio_features[:,0,:]))
        sequence_output = torch.cat((sequence_output[:,0,:], audio_features[:,0,:]), dim=1)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                uncertainty_text_loss = 0.5 * torch.log(variance_text) + 0.5 / variance_text
                uncertainty_audio_loss = 0.5 * torch.log(variance_audio) + 0.5 / variance_audio
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = ce_loss+ (uncertainty_text_loss.mean()+uncertainty_audio_loss.mean())/2
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                uncertainty_text_loss = 0.5 * torch.log(variance_text) + 0.5 / variance_text
                uncertainty_audio_loss = 0.5 * torch.log(variance_audio) + 0.5 / variance_audio
                ce_loss = loss_fct(logits, labels)
                loss = ce_loss+ (uncertainty_text_loss.mean()+uncertainty_audio_loss.mean())/2

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CustomSequenceClassifierOutput(
            loss=loss,
            ce_loss=ce_loss,
            uncertainty_text_loss=uncertainty_text_loss,
            uncertainty_audio_loss=uncertainty_audio_loss,
            logits=logits,
            variance_text=variance_text,
            variance_audio=variance_audio,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





class CustomRobertaUncertaintyDelay(RobertaForSequenceClassification):
    
    def __init__(self,config):
        super().__init__(config)

        self.audio_feature_extractor = AttentionWhisper()
        self.variance_text_head = nn.Linear(config.hidden_size, 1)
        self.variance_audio_head = nn.Linear(self.audio_feature_extractor.M, 1)
        config.hidden_size=config.hidden_size+self.audio_feature_extractor.M
        self.classifier = RobertaClassificationHead(config)
        self.language_delay_head = nn.Linear(config.hidden_size, 2)


    def audio_feature_extractor_load_pretrained(self, ):
        self.audio_feature_extractor = AttentionWhisper()


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        delay_labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        audio_features = self.audio_feature_extractor(audio_data)

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        # print(audio_features.shape)
        variance_text = torch.exp(self.variance_text_head(sequence_output[:,0,:]))
        variance_audio = torch.exp(self.variance_audio_head( audio_features[:,0,:]))
        sequence_output = torch.cat((sequence_output[:,0,:], audio_features[:,0,:]), dim=1)
        logits = self.classifier(sequence_output)
        language_delay_logits = self.language_delay_head(sequence_output)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                uncertainty_text_loss = 0.5 * torch.log(variance_text) + 0.5 / variance_text
                uncertainty_audio_loss = 0.5 * torch.log(variance_audio) + 0.5 / variance_audio
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                language_delay_loss = loss_fct(language_delay_logits.view(-1, self.num_labels), delay_labels.view(-1))
                loss = (language_delay_loss+ce_loss)/2+ (uncertainty_text_loss.mean()+uncertainty_audio_loss.mean())/2
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                uncertainty_text_loss = 0.5 * torch.log(variance_text) + 0.5 / variance_text
                uncertainty_audio_loss = 0.5 * torch.log(variance_audio) + 0.5 / variance_audio
                ce_loss = loss_fct(logits, labels)
                language_delay_loss = loss_fct(language_delay_logits.view(-1, self.num_labels), delay_labels.view(-1))

                loss =(language_delay_loss+ ce_loss)/2+ (uncertainty_text_loss.mean()+uncertainty_audio_loss.mean())/2

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return {'loss':loss,
            'ce_loss':ce_loss,
            'uncertainty_text_loss':uncertainty_text_loss,
            'uncertainty_audio_loss':uncertainty_audio_loss,
            'language_delay_loss':language_delay_loss,
            'logits':logits,
            'language_delay_logits':language_delay_logits,
            'variance_text':variance_text,
            'variance_audio':variance_audio,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,}
        
        
        
        
        
        CustomSequenceClassifierOutput(
            loss=loss,
            ce_loss=ce_loss,
            uncertainty_text_loss=uncertainty_text_loss,
            uncertainty_audio_loss=uncertainty_audio_loss,
            language_delay_loss=language_delay_loss,
            logits=logits,
            language_delay_logits=language_delay_logits,
            variance_text=variance_text,
            variance_audio=variance_audio,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )








class CustomtextUncertainty(RobertaForSequenceClassification):
    
    def __init__(self,config):
        super().__init__(config)

        self.classifier = RobertaClassificationHead(config)
        self.variance_head = nn.Linear(config.hidden_size, 1)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        # print(audio_features.shape)
        sequence_output = sequence_output[:,0,:]
        logits = self.classifier(sequence_output)
        variance = torch.exp(self.variance_head(sequence_output))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                uncertainty_loss = 0.5 * torch.log(variance) + 0.5 / variance
                ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss = ce_loss+ uncertainty_loss.mean()
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                uncertainty_loss = 0.5 * torch.log(variance) + 0.5 / variance
                ce_loss = loss_fct(logits, labels)
                loss = ce_loss+ uncertainty_loss.mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CustomSequenceClassifierOutput(
            loss=loss,
            ce_loss=ce_loss,
            uncertainty_loss=uncertainty_loss,
            logits=logits,
            variance=variance,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )





