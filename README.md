# Offline Handwritten Mathematical Expression Recognition (HMER)

This project implements three models for offline handwritten mathematical expression recognition:
- **BTTR** (Backbone Transformer with Transformer decoder)
- **CAN** (Counting-Aware Network)
- **WAP** (Watch, Attend and Parse)

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
- `img/` - BMP images
- `caption.txt` - Tab-separated file: `filename\tlabel` (space-separated LaTeX tokens)

## Training on Vast.ai

### Step 1: Upload to Vast.ai

1. **Zip the project** (excluding large files):
   ```bash
   # On your local machine
   zip -r offline-hmer.zip offline-hmer/ -x "*.git*" "checkpoints/*" "__pycache__/*" "*.pyc" ".ipynb_checkpoints/*"
   ```

2. **Upload to Vast.ai**:
   - Rent an instance with GPU (e.g., RTX 3090, RTX 4090, A5000, etc.)
   - Upload the zip file via the Vast.ai web interface or use `curl`/`wget` from a file hosting service
   - Unzip: `unzip offline-hmer.zip`

### Step 2: Run Training

```bash
cd offline-hmer

# Train both BTTR and CAN
bash setup_and_train.sh

# Or train only one model
bash setup_and_train.sh bttr
bash setup_and_train.sh can
```

The script will:
1. Install all Python dependencies from `requirements.txt`
2. Verify the CROHME dataset structure
3. Create the `checkpoints/` directory
4. Start training

### Step 3: Monitor Training

Training logs will be printed to the console. Checkpoints are saved in `checkpoints/`:
- `checkpoints/bttr_best.pth` - Best BTTR model
- `checkpoints/densenet_can_best.pth` or `checkpoints/p_densenet_can_best.pth` - Best CAN model

## Local Training

```bash
# BTTR
cd models/bttr
python train_bttr.py

# CAN
cd models/can
python can_trainer.py
```

## Evaluation

```bash
# BTTR evaluation
cd models/bttr
python eval_bttr.py

# CAN evaluation
cd models/can
python can_eval.py

# WAP evaluation
cd models/wap
python wap_eval.py
```

## Gradio Demo

```bash
# BTTR demo
python gradio_demo_bttr.py

# CAN demo
python gradio_demo.py

# WAP demo
python gradio_demo_wap.py
```

## Configuration

Edit `config.json` to adjust model hyperparameters, dataset paths, and training settings.

## Project Structure

```
offline-hmer/
├── CROHME/                  # Dataset
│   ├── train/
│   ├── 2014/
│   ├── 2016/
│   └── 2019/
├── models/
│   ├── bttr/               # BTTR model
│   │   ├── train_bttr.py
│   │   ├── bttr.py
│   │   ├── dataloader.py
│   │   ├── vocab.py
│   │   ├── metrics.py
│   │   ├── beam_search.py
│   │   └── dictionary.txt
│   ├── can/                # CAN model
│   │   ├── can_trainer.py
│   │   ├── can.py
│   │   ├── can_dataloader.py
│   │   └── can_eval.py
│   └── wap/                # WAP model
│       └── wap_eval.py
├── checkpoints/            # Saved model checkpoints
├── config.json             # Configuration file
├── requirements.txt        # Python dependencies
├── setup_and_train.sh      # Vast.ai setup & training script
└── README.md
```
