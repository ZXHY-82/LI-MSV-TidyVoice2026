import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import numpy as np
import random
import librosa
from tqdm import tqdm
import os
random.seed(0)

def resample(wav, orig_sr, target_sr):
    if orig_sr == target_sr:
        return wav

    new_wav = librosa.resample(
        wav.astype(float),
        orig_sr=orig_sr,
        target_sr=target_sr
    )
    return new_wav


cuda_idx = 1

utt2wav = {}
with open('/work/zl389/workspace/LLM_ASV/data/Qwen_TTS/choise2sny.scp', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split()
        utt2wav[tmp[0]] = tmp[1]
utt2text = {}
with open('/work/zl389/workspace/LLM_ASV/data/Qwen_TTS/choise2sny.txt', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        utt2text[tmp[0]] = tmp[1]

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=f"cuda:{cuda_idx}",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)


qwen_langs = ['en', 'zh', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it']


text_dict = {}
total_text_num = 0
for lang in qwen_langs[1:]:
    tmp = np.load(f'/work/zl389/workspace/LLM_ASV/data/Qwen_TTS/{lang}2text.npy', allow_pickle=True).item()
    text_dict[lang] = tmp[lang]
    total_text_num = len(tmp[lang])

with open('/work/zl389/workspace/LLM_ASV/data/Qwen_TTS/dur3-10_text.list', 'r') as f:
    text_dict['en'] = []
    for line in f.readlines():
        text_dict['en'].append(line.strip())


out_dir = '/work/zl389/AudioData/TidyVoice/TidyVoice_Syn'

for utt in tqdm(utt2text):
    utt_split = utt.split('/')
    ref_audio = utt2wav[utt]
    ref_text = utt2text[utt]
    if ref_text == 'ERROR!':
        x_vector_only_mode=True
    else:
        x_vector_only_mode=False

    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=x_vector_only_mode,
    )
    text_list = []
    text_idx_list = []
    for lang in qwen_langs:
        texts = text_dict[lang]
        text_idx = random.sample(range(total_text_num), 1)[0]
        text_idx_list.append(text_idx)
        text_list.append(texts[text_idx])
    wavs, sr = model.generate_voice_clone(
        text=text_list,
        language=['english', 'chinese', 'japanese', 'korean','german', 'french', 'russian', 'portuguese', 'spanish', 'italian'],
        voice_clone_prompt=prompt_items,
        max_new_tokens=120,
    )

    for i in range(len(wavs)):
        wav = wavs[i]
        wav = resample(wav, sr, 16000)
        out_path = f'{out_dir}/{utt_split[0]}/{qwen_langs[i]}/{utt_split[2][:-4]}_{qwen_langs[i]}_{text_idx_list[i]}_Syn.wav'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        sf.write(out_path, wav, 16000)

