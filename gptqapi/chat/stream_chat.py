import torch
from typing import Any, Dict, Generator, List, Optional, Tuple
from threading import Thread
from transformers import AutoTokenizer, GenerationConfig, TextIteratorStreamer
from auto_gptq import AutoGPTQForCausalLM

from gptqapi.extras.misc import dispatch_model, get_logits_processor
from gptqapi.extras.template import get_template_and_fix_tokenizer

SYSTEM_PROMPT=(
    "Anda adalah asisten yang suka membantu, penuh hormat, dan jujur. "
    "Selalu jawab semaksimal mungkin, sambil tetap aman. Jawaban Anda tidak boleh berisi konten berbahaya, "
    "tidak etis, rasis, seksis, toxic, atau ilegal. "
    "Harap pastikan bahwa tanggapan Anda tidak memihak secara sosial dan bersifat positif."
    "\n"
    "Jika sebuah pertanyaan tidak masuk akal, atau tidak koheren secara faktual, jelaskan alasannya daripada menjawab sesuatu yang tidak benar. "
    "Jika Anda tidak mengetahui jawaban atas sebuah pertanyaan, mohon jangan membagikan informasi palsu."
)

def load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, padding_side="left")
    model = AutoGPTQForCausalLM.from_quantized(model_name,
                                               device="cuda:0", 
                                               inject_fused_mlp=True,
                                               inject_fused_attention=True,
                                               trust_remote_code=True)
    return model, tokenizer

class ChatModel:

    def __init__(self, model_name, args=Optional[Dict[str, Any]]):
        global SYSTEM_PROMPT
        self.model, self.tokenizer = load_model_and_tokenizer(model_name)
        self.tokenizer.padding_side = "left"
        self.model = dispatch_model(self.model)
        self.template = get_template_and_fix_tokenizer("llama2", self.tokenizer)
        self.system_prompt = args.get("system_prompt", SYSTEM_PROMPT)
        if len(self.system_prompt.strip()) == 0:
            self.system_prompt = SYSTEM_PROMPT

    def process_args(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[Dict[str, Any], int]:
        system = system or self.system_prompt

        prompt, _ = self.template.encode_oneturn(
            tokenizer=self.tokenizer, query=query, resp="", history=history, system=system
        )
        input_ids = torch.tensor([prompt], device=self.model.device)
        prompt_length = len(input_ids[0])

        do_sample = input_kwargs.pop("do_sample", None)
        temperature = input_kwargs.pop("temperature", None)
        top_p = input_kwargs.pop("top_p", None)
        top_k = input_kwargs.pop("top_k", None)
        repetition_penalty = input_kwargs.pop("repetition_penalty", None)
        max_length = input_kwargs.pop("max_length", None)
        max_new_tokens = input_kwargs.pop("max_new_tokens", None)

        generating_args = {}
        generating_args.update(dict(
            do_sample=do_sample if do_sample is not None else generating_args.get("do_sample"),
            temperature=temperature or generating_args.get("temperature"),
            top_p=top_p or generating_args.get("top_p"),
            top_k=top_k or generating_args.get("top_k"),
            repetition_penalty=repetition_penalty or generating_args.get("repetition_penalty"),
            eos_token_id=[self.tokenizer.eos_token_id] + self.tokenizer.additional_special_tokens_ids,
            pad_token_id=self.tokenizer.pad_token_id
        ))

        if max_length:
            generating_args.pop("max_new_tokens", None)
            generating_args["max_length"] = max_length

        if max_new_tokens:
            generating_args.pop("max_length", None)
            generating_args["max_new_tokens"] = max_new_tokens

        gen_kwargs = dict(
            inputs=input_ids,
            generation_config=GenerationConfig(**generating_args),
            logits_processor=get_logits_processor()
        )

        return gen_kwargs, prompt_length

    @torch.inference_mode()
    def chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Tuple[str, Tuple[int, int]]:
        gen_kwargs, prompt_length = self.process_args(query, history, system, **input_kwargs)
        generation_output = self.model.generate(**gen_kwargs)
        outputs = generation_output.tolist()[0][prompt_length:]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        response_length = len(outputs)
        return response, (prompt_length, response_length)

    @torch.inference_mode()
    def stream_chat(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
        system: Optional[str] = None,
        **input_kwargs
    ) -> Generator[str, None, None]:
        gen_kwargs, _ = self.process_args(query, history, system, **input_kwargs)
        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        yield from streamer
