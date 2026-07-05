# Offline Handwritten Mathematical Expression Recognition (HMER)

This project implements three models for offline handwritten mathematical expression recognition:
- **WAP** (Watch, Attend and Parse) — CNN + RNN with attention
- **CoMER** (Coverage-guided Masked Encoder-decoder with Refinement) — DenseNet + Transformer with Attention Refinement Module
- **CAN** (Counting-Aware Network) — DenseNet + RNN with coverage and counting

## Dataset

The CROHME dataset is organized as follows:
```
CROHME/
├── train/       # Training set (CROHME 2014 training + additional)
├── 2014/        # CROHME 2014 test set (used as additional training)
├── 2016/        # CROHME 2016 test set (used as validation)
└── 2019/        # CROHME 2019 test set (used as test)
```

Each split contains:
- `img/` — BMP images
- `caption.txt` — Tab-separated file: `filename\tlabel` (space-separated LaTeX tokens)

## Gradio Demo

Run the combined demo (supports all three models):

```bash
python gradio_demo.py
```

Or run individual model demos:

```bash
python gradio_demo_comer.py   # CoMER only
```

### Usage

1. Select model: **WAP**, **CoMER**, or **CAN**
2. Choose input type: **Upload image** or **Use sketchpad**
3. Draw or upload a handwritten mathematical expression
4. Click **Recognize**

### Sketchpad

The sketchpad uses a **black background** with **white brush**. This matches the expected input format for all models.

## Project Structure

```
CV_Project/
├── models/
│   ├── comer/               # CoMER model (DenseNet + Transformer + ARM)
│   │   ├── model.py
│   │   ├── encoder.py       # DenseNet encoder
│   │   ├── decoder.py       # Transformer decoder with ARM
│   │   ├── config.py
│   │   ├── pos_enc.py       # Positional encoding
│   │   ├── transformer/     # Transformer modules
│   │   └── vocab.json
│   ├── can/                 # CAN model
│   │   ├── can.py
│   │   ├── can_dataloader.py
│   │   └── can_eval.py
│   └── wap/                 # WAP model
│       ├── wap.py
│       ├── wap_dataloader.py
│       └── wap_eval.py
├── final_trained_models/    # Pretrained model checkpoints
├── dataset/                 # Dataset files
├── notebooks/               # Jupyter notebooks
├── codes/                   # Utility scripts
├── outputs/                 # Evaluation outputs
├── gradio_demo.py           # Combined Gradio demo (all models)
├── gradio_demo_comer.py     # CoMER-only Gradio demo
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
└── README.md
```

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- Python 3.10+
- PyTorch 2.0+
- Gradio 5+
- OpenCV
- PIL
- Matplotlib
- Albumentations (for CAN)

