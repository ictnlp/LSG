from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
import pdb
from argparse import Namespace, ArgumentParser
import numpy as np
import torch.nn.functional as F
import sys
import re

from audio import log_mel_spectrogram, get_T_after_cnn, pad_or_trim

import codecs
import pdb
import torch

# template for Qwen_Audio
translation_template = '<|startoftranscript|><|{}|><|translate|><|{}|><|notimestamps|><|itn|>'

torch.manual_seed(1234)

@entrypoint
class Qwen_audio_LSG(SpeechToTextAgent):
    def __init__(self, args: Namespace):
        super().__init__(args)
        
        # the parameter for confidence
        self.alpha = args.threshold
        # low and top boundary
        self.low_bound = args.low_bound
        self.top_bound = args.top_bound
        # the language pair
        self.languge_pair = args.lang_pair
        # the parameter for delta
        self.delta = args.decision_ratio
        self.source_lang, self.target_lang = (args.lang_pair).split('_')
        # the parameter for Qwen2-Audio
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="cuda", trust_remote_code=True).eval()
        self.generation_config = GenerationConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        # generation config
        self.generation_config.num_beams = 1
        self.generation_config.do_sample = False
        self.sp_prompt = translation_template.format(self.source_lang, self.target_lang)
        self.source_size = args.source_size

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add customized command line arguments"""
        parser.add_argument("--threshold", type=float)
        parser.add_argument("--low_bound", type=int)
        parser.add_argument("--top_bound", type=int)
        parser.add_argument("--decision_ratio", type=float, default=0.3)
        parser.add_argument("--lang_pair", type=str)
        parser.add_argument("--model_dir", type=str)
        parser.add_argument("--source_size", type=int, default=320)
    
    # Transform the waveform to mel spectrum
    def generate_audio_info(self, audio):
        L = (audio.shape[0] if audio.shape[0] <= 480000 else 480000)  # max_length < 30s
        mel_len = L // 160
        audio = pad_or_trim(audio.flatten())
        mel = log_mel_spectrogram(audio)
        audio_len_after_cnn = get_T_after_cnn(mel_len)
        audio_token_num = (audio_len_after_cnn - 2) // 2 + 1
        audio_len = [audio_len_after_cnn, audio_token_num]

        return mel, audio_len, audio_token_num + 2
    # Prepare for Inputs
    def get_model_inputs(self, audio):
        audio_urls = ['audio.wav']

        audios, audio_lens, audio_span_tokens = [], [], []
        self_policy_dict = self.generate_audio_info(audio)

        # append audio information
        audios.append(self_policy_dict[0])
        audio_lens.append(self_policy_dict[1])
        audio_span_tokens.append(self_policy_dict[2])
        
        # audio information
        input_audio_lengths = torch.IntTensor(audio_lens)
        input_audios = torch.stack(audios, dim=0)

        audio_info =  {"input_audios": input_audios,
                        "input_audio_lengths": input_audio_lengths,
                        "audio_span_tokens": audio_span_tokens,
                        "audio_urls": audio_urls
                    }
        
        # prepare the input for the audio LLMs
        query = f"<audio>{audio_urls[0]}</audio>{self.sp_prompt}" + ' '.join(self.states.target)
        inputs = self.tokenizer(query, return_tensors='pt', audio_info=audio_info)
        inputs = inputs.to(self.model.device)
        input_length = inputs['input_ids'].size(-1)

        return inputs, input_length, audio_info
    
    def policy(self):
        if len(self.states.source) <= 0:
            return ReadAction()
        
        source_len = round(len(self.states.source) / 16000 / (self.source_size/1000))
        target_len = len(self.states.target)
        
        if self.states.source_finished == False and (self.low_bound > (source_len - target_len)):
            return ReadAction()
        
        # judge whether the input is finished
        if self.states.source_finished:
            audio = torch.tensor(self.states.source, device=self.model.device, dtype=torch.float32)

            inputs, input_length, audio_info = self.get_model_inputs(audio)

            self.generation_config.max_new_tokens = 128
            self.generation_config.output_scores=False
            self.generation_config.return_dict_in_generate=False

            pred = self.model.generate(**inputs, audio_info=audio_info, generation_config=self.generation_config)
            response = self.tokenizer.decode(pred.cpu()[0][input_length:], skip_special_tokens=True,audio_info=audio_info)

            return WriteAction(response.strip(), finished=True)
        
        # wait1 speech input process
        #wait1_audio_urls = ['audio.wav']
        # self_policy speech input process
        audio = torch.tensor(self.states.source, device=self.model.device, dtype=torch.float32)
        audio_inputs, audio_input_length, audio_info = self.get_model_inputs(audio)
        
        self.generation_config.max_new_tokens = 10
        self.generation_config.output_scores=True
        self.generation_config.return_dict_in_generate=True
        # Predict the current inputs
        pred = self.model.generate(**audio_inputs, audio_info=audio_info, generation_config=self.generation_config)
        logits = F.softmax(torch.stack(pred.scores, dim=1)[0], dim=-1)
        max_logits = torch.max(logits, dim=-1)[0]
        pred_int = pred.sequences[0][audio_input_length: -1]

        # Prepare the wait-1 inputs
        wait1_audio = torch.tensor(self.states.source[: int((target_len+1)*(self.source_size / 1000.0)*16000)], device=self.model.device, dtype=torch.float)
        wait1_inputs, wait1_input_length, wait1_audio_info = self.get_model_inputs(wait1_audio)
        wait1_inputs['input_ids'] = torch.cat([wait1_inputs['input_ids'], pred_int.unsqueeze(0)], dim=-1)
        
        wait1_pred = self.model(input_ids=wait1_inputs['input_ids'], audio_info=wait1_audio_info).logits[0, -(pred_int.size(-1)+1):]
        wait1_logits = F.softmax(wait1_pred, dim=-1)
        # Compare kl_logits and decide the next action
        kl_div_logits = F.kl_div(logits.log(), wait1_logits, reduction='none').sum(dim=-1)
        logit_mask = torch.cumprod((max_logits >= self.alpha) | (kl_div_logits >= self.delta), dim=-1).sum()

        generate_len = min(max(logit_mask, max(source_len - (target_len + self.top_bound), 0)), source_len - (target_len + self.low_bound))
        generate_len = min(generate_len, logits.size(-1))

        if generate_len <= 0:
            return ReadAction()
        
        response = self.tokenizer.decode(pred.sequences[0].cpu()[audio_input_length:audio_input_length+generate_len], skip_special_tokens=True,audio_info=audio_info)

        if generate_len < (logits.size(-1) - 1) and pred.sequences[0][-1] == 151643:
            response_second = self.tokenizer.decode(pred.sequences[0].cpu()[audio_input_length:audio_input_length+generate_len+1], skip_special_tokens=True,audio_info=audio_info)
            if response_second.strip().split(' ') > response.strip().split(' '):

                tmp_response = re.sub(r'[0-9.]', '', response.strip()).strip()
                if tmp_response == '':
                    ReadAction()
                else:
                    return WriteAction(tmp_response, finished=False)
            else:
                tmp_response = re.sub(r'[0-9.]', '', ' '.join(response.strip().split(' ')[:-1])).strip()
                if tmp_response == '':
                    return ReadAction()
                else:
                    return WriteAction(tmp_response, finished=False)
        
        if logits.size(-1) >= self.generation_config.max_new_tokens:
            tmp_response = re.sub(r'[0-9.]', '', ' '.join(response.strip().split(' ')[:-1])).strip()
            if tmp_response == '':
                return ReadAction()
            return WriteAction(tmp_response, finished=False)
        else:
            tmp_response = re.sub(r'[0-9.]', '', response.strip()).strip()
            if tmp_response == '':
                return ReadAction()
            return WriteAction(tmp_response, finished=False)