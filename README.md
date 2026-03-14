# [Language-Invariant Multilingual Speaker Verification for the TidyVoice 2026 Challenge](https://arxiv.org/abs/2603.08092)

#### Note:

This project is built upon our previous repository [ZXHY-82/w2v-BERT-2.0_SV (github.com)](https://github.com/ZXHY-82/w2v-BERT-2.0_SV)

### Preparation Stage 

Download the W2V-BERT 2.0 pre-trained weights from Hugging Face and place them in the designated directory:

```
URL: https://huggingface.co/facebook/w2v-bert-2.0/blob/main/model.safetensors
Destination folder: deeplab/pretrained/audio2vector/ckpts/facebook/w2v-bert-2.0/
```

Environment Setup

```
conda create -y -n asv python=3.9

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

pip uninstall transformers

conda install -c conda-forge sox
```

![](./assets/framework.png)

### Train Stage

**Stage1: Large-Scale Speaker Model Pre-training**

```
#step1: Pre-trained model freeze training
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
--tag base_s1_ \
--is_distributed true \
--yaml conf/base_model/s1.yaml

#step2: Merging LoRA module parameters into the pre-trained model
cd utils
python3 lora_merge.py

#step3: Joint fine-tuning
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
--tag base_s2_ \
--is_distributed true \
--yaml conf/base_model/s2.yaml \
--pretrain results/checkpoints/base_s1_xxxxxxx/merge_lora.pth
```

**Stage2: Fine-tuning with Language Invariant Learning**

```
# step0: If you fine-tune the model using only the TidyVoice training set, please initializing the classifier using the average speaker embeddings extracted by the base model from the training data.
cd utils
python3 init_classifier.py

# step1: training language classifier
OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang.py \
--tag base_ft_language_ \
--is_distributed true \
--yaml conf/ft_base_model/s1.yaml \
--pretrain results/checkpoints/base_s2_xxxxxxx/ckpt_only_tv.pth

# step2: fine-tuning with language invariant learning
cd utils
python3 merge_classifier.py

OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang_grl.py \
--tag base_ft_language_grl_ \
--is_distributed true \
--yaml conf/ft_base_model/s2.yaml \
--pretrain results/checkpoints/base_ft_language_xxxxxxx/ckpt_update.pth
```

### Multilingual Synthetic Speech Generation

```
cd Syn

# For the text corpus, we use English sentences from LibriTTS, which are then translated into the target languages using the M2M100 multilingual translation model.
python3 m2m100.py

# Reference audio is processed with Whisper-large-v3 to obtain the corresponding transcript
python3 whisper.py

# Both the reference audio and text are fed into Qwen3-TTS to generate synthetic speech in the specified language and target text.
python3 tts.py
```

### Test stage

```
cd utils
python3 get_embd_w2v.py
```

![](./assets/table1.png)

## Citations

```
@article{li2026language,
  title={Language-Invariant Multilingual Speaker Verification for the TidyVoice 2026 Challenge},
  author={Li, Ze and Miao, Xiaoxiao and Liu, Juan and Li, Ming},
  journal={arXiv preprint arXiv:2603.08092},
  year={2026}
}
```

