# step1: Pre-trained model freeze training
# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
# torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
# --tag base_s1_ \
# --is_distributed true \
# --yaml conf/base_model/s1.yaml

# #step2: Merging LoRA module parameters into the pre-trained model
# cd utils
# python3 lora_merge.py

# #step3: Joint fine-tuning
# OMP_NUM_THREADS="16" CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"  \
# torchrun --nnodes 1 --nproc_per_node=8 --master_port=12885 train.py \
# --tag base_s2_ \
# --is_distributed true \
# --yaml conf/base_model/s2.yaml \
# --pretrain results/checkpoints/base_s1_260314041032/merge_lora.pth