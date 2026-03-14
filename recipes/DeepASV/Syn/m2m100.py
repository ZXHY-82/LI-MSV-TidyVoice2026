import os, sys
sys.path.append('../../..')
sys.path.append(os.path.split(__file__)[0])
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import numpy as np
import torch

device = "cuda:4" if torch.cuda.is_available() else "cpu"

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M").to(device)
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
tokenizer.src_lang = "en"


qwen_langs = ['zh', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it']
en2langs = {}

# dur3-10_text.list:  English sentences from LibriTTS with a duration of 3 to 10 seconds
with open('/work/zl389/workspace/LLM_ASV/data/Qwen_TTS/dur3-10_text.list', 'r') as f:
    batch_size = 100
    all_en_texts = [line.strip() for line in f]
    for lang in qwen_langs:
        lang2text = {}
        lang2text[lang] = []
        for i in tqdm(range(0, len(all_en_texts), batch_size)):
            batch_texts = all_en_texts[i:i + batch_size]
            encoded_en = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
            generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id(lang))
            res = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for re in res:
                lang2text[lang].append(re)
        np.save(f'./{lang}2text.npy', lang2text)

