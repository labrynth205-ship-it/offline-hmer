import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import torch
from PIL import Image
import numpy as np
import gradio as gr
from functools import partial
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import sys
sys.path.insert(0, 'models/comer')
from model import build_model
from config import Config


def load_checkpoint(checkpoint_path, device):
    '''
    Load CoMER checkpoint and return model and vocabulary
    '''
    config = Config()
    
    import json
    with open('models/comer/vocab.json', 'r') as f:
        vocab_data = json.load(f)
    
    vocab_size = len(vocab_data['token2idx'])
    
    model = build_model(config, vocab_size).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded CoMER model from {checkpoint_path} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
    
    return model, vocab_data


def preprocess_image_for_comer(image_np):
    """
    Preprocess image to match CoMER training format.
    CoMER expects: black background, white foreground, grayscale, variable size.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    h, w = binary.shape
    border = np.concatenate([
        binary[0, :], binary[-1, :],
        binary[:, 0], binary[:, -1]
    ])
    bg_is_white = np.mean(border) > 128
    
    if bg_is_white:
        binary = 255 - binary
    
    h_hi, w_hi = 128, 512
    scale = min(h_hi / h, w_hi / w, 1.0)
    if scale < 1.0:
        new_h = max(1, int(h * scale))
        new_w = max(1, int(w * scale))
        binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return binary


def recognize_single_image(model, image, vocab_data, device, max_length=200):
    '''
    Recognize handwritten mathematical expression from a single image using CoMER
    '''
    processed_img = preprocess_image_for_comer(image)
    
    h, w = processed_img.shape
    img_tensor = torch.from_numpy(processed_img).float().unsqueeze(0).unsqueeze(0) / 255.0
    mask_tensor = torch.zeros(1, h, w, dtype=torch.bool)
    
    img_tensor = img_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        pred_indices, attentions = model.greedy_decode_with_attention(
            img_tensor, mask_tensor,
            sos_idx=vocab_data['token2idx']['<SOS>'],
            eos_idx=vocab_data['token2idx']['<EOS>'],
            max_len=max_length,
        )
    
    idx2token = {int(k): v for k, v in vocab_data['idx2token'].items()}
    latex_tokens = []
    for idx in pred_indices[0]:
        tok = idx2token.get(idx, '<UNK>')
        if tok == '<EOS>':
            break
        if tok not in ['<PAD>', '<SOS>']:
            latex_tokens.append(tok)
    
    latex = ' '.join(latex_tokens)
    
    return latex, attentions, processed_img


def create_attention_heatmap(original_img, attentions, latex_tokens, max_tokens=20):
    """
    Create a grid of attention heatmaps overlaid on the original image.
    Shows one heatmap per decoded token.
    """
    n_tokens = min(len(latex_tokens), max_tokens, attentions.shape[0])
    
    if n_tokens == 0:
        return None
    
    # Resize attentions to match original image size
    h_orig, w_orig = original_img.shape
    attn_resized = []
    for i in range(n_tokens):
        attn = attentions[i].numpy()
        attn = cv2.resize(attn, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        attn_resized.append(attn)
    
    # Create a grid
    cols = min(5, n_tokens)
    rows = (n_tokens + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(rows * cols):
        r, c = i // cols, i % cols
        if i < n_tokens:
            ax = axes[r, c]
            # Show original image in background
            ax.imshow(original_img, cmap='gray', alpha=0.6)
            # Overlay attention heatmap
            ax.imshow(attn_resized[i], cmap='jet', alpha=0.5, 
                      norm=Normalize(vmin=0, vmax=attn_resized[i].max() if attn_resized[i].max() > 0 else 1))
            ax.set_title(f'Step {i+1}: {latex_tokens[i]}', fontsize=9)
            ax.axis('off')
        else:
            axes[r, c].axis('off')
    
    plt.tight_layout()
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    # Remove alpha channel
    buf = buf[:, :, :3]
    plt.close(fig)
    
    return buf


def recognize_and_display(model, vocab_data, device, image):
    '''
    Process the input image and return the LaTeX string, rendered image, and attention heatmap
    '''
    if isinstance(image, dict):
        image = image.get("composite")

    if image is None:
        return "Please provide an image.", None, None

    latex_string, attentions, processed_img = recognize_single_image(model, image, vocab_data, device)
    rendered_latex = f"$${latex_string}$$"
    
    # Create attention heatmap
    latex_tokens = latex_string.split()
    heatmap = create_attention_heatmap(processed_img, attentions, latex_tokens)

    return latex_string, rendered_latex, heatmap


def process_input(input_type, uploaded_image, sketchpad_data, model, vocab_data, device):
    """
    Wrapper function to select the correct image input based on the dropdown menu
    """
    if input_type == "Upload image":
        image_to_process = uploaded_image
    elif input_type == "Use sketchpad":
        image_to_process = sketchpad_data
    else:
        raise Exception('invalid input type')

    return recognize_and_display(model, vocab_data, device, image_to_process)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint_path = 'final_trained_models/comer_best.pt'
    model, vocab_data = load_checkpoint(checkpoint_path, device)

    with gr.Blocks(title="Offline HMER - CoMER Model") as demo:
        gr.Markdown(
            """
            # 🧮 CoMER Model - Handwritten Mathematical Expression Recognition
            Upload an image or use the sketchpad to recognize a handwritten mathematical expression.
            **CoMER (Counting-Aware Transformer)** - DenseNet Encoder + Transformer Decoder with Attention Refinement Module.
            Pretrained on CROHME 2013/2016/2019 datasets.
            
            The attention heatmap shows which parts of the image the model focuses on when generating each token.
            """
        )

        with gr.Row():
            input_choice = gr.Dropdown(
                choices=["Upload image", "Use sketchpad"],
                label="Select Input Type",
                value="Use sketchpad",
                interactive=True
            )

        upload_img = gr.Image(
            type='numpy',
            label="Input image (Upload)",
            visible=False,
        )

        black_background = np.zeros((256, 1024, 3), dtype=np.uint8)

        sketchpad = gr.Sketchpad(
            type='numpy',
            image_mode='RGB',
            brush=gr.Brush(colors=["#ffffff"], color_mode="fixed", default_size=3),
            value={'layers': [], 'background': black_background, 'composite': None},
            label="Input image (Sketchpad)",
            visible=False,
            height=256,
            width=1024,
        )

        process_button = gr.Button("Recognize")

        with gr.Column():
            latex_output = gr.Textbox(
                label="Recognized LaTeX", interactive=False)
            markdown_output = gr.Markdown(
                label="Rendered LaTeX")
            heatmap_output = gr.Image(
                label="Attention Heatmap (per token)",
                type='numpy',
                visible=True,
            )

        def update_input_visibility(choice):
            if choice == "Upload image":
                return gr.update(visible=True), gr.update(visible=False)
            elif choice == "Use sketchpad":
                return gr.update(visible=False), gr.update(visible=True)
            return gr.update(visible=False), gr.update(visible=False)

        input_choice.change(
            fn=update_input_visibility,
            inputs=[input_choice],
            outputs=[upload_img, sketchpad],
            queue=False
        )

        demo.load(
            fn=update_input_visibility,
            inputs=[input_choice],
            outputs=[upload_img, sketchpad],
            queue=False
        )

        process_button.click(
            fn=partial(process_input, model=model, vocab_data=vocab_data, device=device),
            inputs=[input_choice, upload_img, sketchpad],
            outputs=[latex_output, markdown_output, heatmap_output]
        )

    demo.launch()
