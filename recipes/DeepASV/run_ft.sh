# step0: If you fine-tune the model using only the TidyVoice training set, please initializing the classifier using the average speaker embeddings extracted by the base model from the training data.
# cd utils
# python3 init_classifier.py

# # step1: training language classifier
# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
# torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang.py \
# --tag base_ft_language_ \
# --is_distributed true \
# --yaml conf/ft_base_model/s1.yaml \
# --pretrain results/checkpoints/base_s2_260314042050/ckpt_only_tv.pth

# # step2: fine-tuning with language invariant learning
# cd utils
# python3 merge_classifier.py

OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
torchrun --nnodes 1 --nproc_per_node=8 --master_port=12886 train_sf2_lang_grl.py \
--tag base_ft_language_grl_ \
--is_distributed true \
--yaml conf/ft_base_model/s2.yaml \
--pretrain results/checkpoints/base_ft_language_260314044238/ckpt_update.pth