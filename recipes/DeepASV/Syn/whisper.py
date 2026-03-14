import os, sys
sys.path.append('../../..')
sys.path.append(os.path.split(__file__)[0])
sys.path.append('../../../deeplab/pretrained/audio2vector/module/transformers/src')
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm import tqdm

device = "cuda:1" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = '/work/zl389/workspace/LLM_ASV/deeplab/pretrained/audio2vector/ckpts/openai/whisper-large-v3'

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)


qwen_langs = ['zh-CN', 'en', 'ja', 'ko', 'de', 'fr', 'ru', 'pt', 'es', 'it']


utt2wav = {}
lang2utt = {}
other_utts = []
with open('/work/zl389/workspace/LLM_ASV/data/Qwen_TTS/choise2sny.scp', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split()
        utt = tmp[0]
        wav = tmp[1]
        utt2wav[utt] = wav
        lang = utt.split('/')[1]
        if lang in qwen_langs:
            if lang2utt.get(lang) == None:
                lang2utt[lang] = []
            lang2utt[lang].append(utt)
        else:
            other_utts.append(utt)

f_w = open('./choise2sny.txt', 'w', buffering=1000)

error_lang = {'zh-CN':'zh', 'hy-AM':'hy'}

for lang in lang2utt:
    utts = lang2utt[lang]
    if error_lang.get(lang) != None:
        lang = error_lang[lang]
    for i in tqdm(range(0, len(utts), 10)):
        batch_utts = utts[i:i + 10]
        wav_list = [utt2wav[u] for u in batch_utts]
        result = pipe(
            wav_list,
            generate_kwargs={"language": lang},
        )
        for idx, r in enumerate(result):
            f_w.write(f"{batch_utts[idx]}\t{r['text']}\n")

for utt in other_utts:
    f_w.write(f'{utt}\tERROR!\n')

