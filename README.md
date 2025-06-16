# LAFITE (Language-Free Training for Text-to-Image Generation)

> 🧠 Official implementation of [LAFITE: Towards Language-Free Training for Text-to-Image Generation (CVPR 2022)](https://arxiv.org/abs/2111.13792)

**LAFITE** is the *first* text-to-image model that enables training **without any paired text captions**.  
It leverages the multimodal embedding space of CLIP to generate images based on pseudo text features extracted from images alone.

> Looking for newer methods? Try this [Shifted Diffusion](https://github.com/drboog/Shifted_Diffusion)

---

## 🔧 Setup & Dependencies

This repo is based on:
- [StyleGAN2-ADA (PyTorch)](https://github.com/NVlabs/stylegan2-ada-pytorch)
- [CLIP (OpenAI)](https://github.com/openai/CLIP)

Install the required dependencies listed in the original StyleGAN2-ADA and CLIP repos.

---

## 📂 Dataset Preparation

To convert your dataset:
```bash
python dataset_tool.py --source=./your_dataset/ --dest=./datasets/your_dataset.zip --width=256 --height=256 --transform=center-crop
```

**Required format**:
```
your_dataset/
├── 1.png
├── 1.txt     ← optional (ignored in language-free mode)
├── 2.png
├── 2.txt
└── ...
```

📦 Ready-made datasets (CLIP-ViT/B-32 encoded):
- [MS-COCO Train](https://drive.google.com/file/d/1b82BCh65XxwR-TiA8zu__wwiEHLCgrw2/view)
- [MS-COCO Val](https://drive.google.com/file/d/1qBy5rPfo1go4d-PjF_Gu0kESCZ9Nt1Ta/view)
- [LN-COCO, CUB, CelebA-HQ](#) *(see original repo for full list)*

---

## 🏋️‍♀️ Training

### With Ground-Truth Pairs (Supervised)
```bash
python train.py --gpus=4 --outdir=./outputs/ --temp=0.5 --itd=5 --itc=10 --gamma=10 \
--mirror=1 --data=./datasets/train.zip --test_data=./datasets/val.zip --mixing_prob=0.0
```

### With Pseudo Pairs (Language-Free)
```bash
python train.py --gpus=4 --outdir=./outputs/ --temp=0.5 --itd=10 --itc=10 --gamma=10 \
--mirror=1 --data=./datasets/train.zip --test_data=./datasets/val.zip --mixing_prob=1.0
```

> 🔎 Key hyperparameters:
> - `--itd`: discriminator iterations
> - `--itc`: contrastive iterations
> - `--gamma`: contrastive loss weight

---

## 🥪 Evaluation & Testing

### FID / IS Metrics
```bash
python calc_metrics.py --network=./model.pkl --metrics=fid50k_full,is50k \
--data=./train.zip --test_data=./val.zip
```

### Image Generation
Use `generate.ipynb` locally or try this [Colab Notebook](https://colab.research.google.com/github/pollinations/hive/blob/main/interesting_notebooks/LAFITE_generate.ipynb)

### SOA (Semantic Object Accuracy)
```bash
python generate_for_soa.py
# Refer to: https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis
```

---

## 📦 Pre-trained Models

- [LAFITE-G (MS-COCO, Language-free)](https://drive.google.com/file/d/1eNkuZyleGJ3A3WXTCIGYXaPwJ6NH9LRA/view)
- [LAFITE-NN (MS-COCO, Language-free)](https://drive.google.com/file/d/1WQnlCM4pQZrw3u9ZeqjeUNqHuYfiDEU3/view)
- [Supervised (MS-COCO)](https://drive.google.com/file/d/1tMD6MWydRDMaaM7iTOKsUK-Wv2YNDRRt/view)
- [Pre-trained on CC3M](https://drive.google.com/file/d/17ER7Yl02Y6yCPbyWxK_tGrJ8RKkcieKq/view)

---

## 📊 Highlights from Paper

- 🔹 **No captions needed**: Uses CLIP image encoder to synthesize pseudo-text features.
- 🔹 **Strong zero-shot performance**: Outperforms DALL-E & CogView on MS-COCO with only 1% model size.
- 🔹 **Plug & play fine-tuning**: Efficiently fine-tune on downstream datasets using only images.
- 🔹 **SoTA Results** on MS-COCO, CUB, CelebA-HQ, and LN-COCO with FID and IS.

---

## 📜 Citation
```bibtex
@article{zhou2021lafite,
  title={LAFITE: Towards Language-Free Training for Text-to-Image Generation},
  author={Zhou, Yufan and Zhang, Ruiyi and Chen, Changyou and Li, Chunyuan and Tensmeyer, Chris and Yu, Tong and Gu, Jiuxiang and Xu, Jinhui and Sun, Tong},
  journal={arXiv preprint arXiv:2111.13792},
  year={2021}
}
```

---

## 📬 Contact

For questions or collaboration, contact:  
📧 yufanzho@buffalo.edu
