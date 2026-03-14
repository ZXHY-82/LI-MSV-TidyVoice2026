import os, sys
sys.path.append('..')
sys.path.append('../../..')
from deeplab.utils.corpus import *
import numpy as np
import torch

# You could use 'utils/get_embd_w2v.py' to extract embd.
# tv_embd: {'TV_spkid_idx': embd}
tv_embd = np.load('/work/zl389/workspace/LLM_ASV/data/prepare/tidyvoice.npy', allow_pickle=True).item()

tv_spk2embd = {}
for key in tv_embd.keys():
    tmp = key.split('_')
    spk = f'{tmp[0]}_{tmp[1]}' # spk = 'TV_spkid'
    if tv_spk2embd.get(spk) == None:
        tv_spk2embd[spk] = []
    tv_spk2embd[spk].append(tv_embd[key])


# /work/zl389/workspace/LLM_ASV/data/ASV_data/TV2026_train.spk2utt. Each Line:'TV_spkid wav_path'
tv_root = '/work/zl389/workspace/LLM_ASV/data/ASV_data'
spk2utt = load_audio_corpus(tv_root, ['TV2026_train'])
spk_ids = sorted(list(spk2utt.keys()))

tv_embd = []
for spk in spk_ids:
    embds = tv_spk2embd[spk]
    mean_feat = np.mean(np.stack(embds, axis=0), axis=0)
    tv_embd.append(mean_feat[0])
tv_tensor = torch.from_numpy(np.stack(tv_embd, axis=0))

# best ckpt from base_s2
ckpt_path = '/work/zl389/workspace/LLM_ASV/publish_code_2/recipes/DeepASV/results/checkpoints/base_s2_260314042050/ckpt_0002.pth'

ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)

classifier = ckpt_data['modules']['classifier']

weights = classifier['weight']
print(weights.shape)
weights = tv_tensor
print(weights.shape)
ckpt_data['modules']['classifier']['weight'] = weights
torch.save(ckpt_data, '/work/zl389/workspace/LLM_ASV/publish_code_2/recipes/DeepASV/results/checkpoints/base_s2_260314042050/ckpt_only_tv.pth')