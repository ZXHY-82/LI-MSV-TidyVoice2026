import torch

ckpt_path = '/work/zl389/workspace/LLM_ASV/publish_code_2/recipes/DeepASV/results/checkpoints/base_s2_260314042050/ckpt_only_tv.pth'

ckpt_data = torch.load(ckpt_path, map_location='cpu', weights_only=False)

classifier = ckpt_data['modules']['classifier']

ckpt_path2 = '/work/zl389/workspace/LLM_ASV/publish_code_2/recipes/DeepASV/results/checkpoints/base_ft_language_260314044238/ckpt_0003.pth'

ckpt_data2 = torch.load(ckpt_path2, map_location='cpu', weights_only=False)

ckpt_data2['modules']['classifier'] = classifier

torch.save(ckpt_data2, '/work/zl389/workspace/LLM_ASV/publish_code_2/recipes/DeepASV/results/checkpoints/base_ft_language_260314044238/ckpt_update.pth')
