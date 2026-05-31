import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'  # Fix matplotlib threading issues

import torch
from PIL import Image
import numpy as np
import gradio as gr
from functools import partial
import traceback
import cv2
import albumentations as A
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== WAP imports =====
from models.wap.wap import WAP
from models.wap.wap_dataloader import Vocabulary as WAPVocab
torch.serialization.add_safe_globals([WAPVocab])
from models.wap.wap_eval import recognize_single_image as recognize_single_image_wap, load_checkpoint as load_checkpoint_wap

# ===== BTTR imports =====
import sys
sys.path.insert(0, 'models/bttr')
from bttr import BTTR
from vocab import CROHMEVocab
from beam_search import beam_search_batch

# ===== CAN imports =====
sys.path.insert(0, 'models/can')
from can import CAN, create_can_model
from can_dataloader import Vocabulary as CANVocab, INPUT_HEIGHT, INPUT_WIDTH

torch.serialization.add_safe_globals([CANVocab])

# ===== Global variables =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ===== Load WAP model =====
wap_checkpoint_path = 'final_trained_models/wap_best.pth'
wap_model, wap_vocab = load_checkpoint_wap(wap_checkpoint_path, device)
print("WAP model loaded successfully!")

# ===== Load BTTR model =====
bttr_checkpoint_path = 'final_trained_models/bttr_best.pth'
bttr_model = BTTR(
    d_model=256,
    growth_rate=16,
    num_layers=3,
    nhead=8,
    num_decoder_layers=3,
    dim_feedforward=1024,
    dropout=0.1
).to(device)
bttr_model.load_state_dict(torch.load(bttr_checkpoint_path, map_location=device, weights_only=True))
bttr_model.eval()
bttr_vocab = CROHMEVocab()
print("BTTR model loaded successfully!")

# ===== Load CAN model =====
can_checkpoint_path = 'final_trained_models/p_densenet_can_best.pth'
can_checkpoint = torch.load(can_checkpoint_path, map_location=device, weights_only=False)
can_vocab = can_checkpoint.get('vocab')
if can_vocab is None:
    vocab_path = 'models/can/hmer_vocab.pth'
    if os.path.exists(vocab_path):
        vocab_data = torch.load(vocab_path)
        can_vocab = CANVocab()
        can_vocab.word2idx = vocab_data['word2idx']
        can_vocab.idx2word = vocab_data['idx2word']
        can_vocab.idx = vocab_data['idx']
        can_vocab.pad_token = can_vocab.word2idx['<pad>']
        can_vocab.start_token = can_vocab.word2idx['<start>']
        can_vocab.end_token = can_vocab.word2idx['<end>']
        can_vocab.unk_token = can_vocab.word2idx['<unk>']
    else:
        raise ValueError(f"Vocabulary not found in checkpoint and {vocab_path} does not exist")

can_hidden_size = can_checkpoint.get('hidden_size', 256)
can_embedding_dim = can_checkpoint.get('embedding_dim', 256)
can_use_coverage = can_checkpoint.get('use_coverage', True)

can_model = create_can_model(
    num_classes=len(can_vocab),
    hidden_size=can_hidden_size,
    embedding_dim=can_embedding_dim,
    use_coverage=can_use_coverage,
    pretrained_backbone=True,
    backbone_type='densenet'
).to(device)
can_model.load_state_dict(can_checkpoint['model'])
can_model.eval()
print("CAN model loaded successfully!")


# ===== WAP functions =====
def recognize_single_image_wrapper_wap(model, image, vocab, device, max_length=150, visualize_attention=False):
    temp_img_path = 'temp_input_image.png'
    image.save(temp_img_path)
    try:
        output = recognize_single_image_wap(model, temp_img_path, vocab, device, max_length, visualize_attention)
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    return output


def recognize_and_display_wap(model, vocab, device, image):
    if isinstance(image, dict):
        image = image.get("composite")
    if image is None:
        return "Please provide an image.", None, None
    pil_image = Image.fromarray(image)
    latex_string = recognize_single_image_wrapper_wap(
        model, pil_image, vocab, device, visualize_attention=True)
    rendered_latex = f"$${latex_string}$$"
    if os.path.exists('attention_maps_wap.png'):
        attention_maps_image = Image.open('attention_maps_wap.png')
    else:
        attention_maps_image = None
    return latex_string, rendered_latex, attention_maps_image


# ===== BTTR functions =====
def recognize_single_image_bttr(model, image, vocab, device, max_length=200, beam_size=5):
    """Recognize a single image using BTTR model"""
    if isinstance(image, dict):
        image = image.get("composite")
    if image is None:
        return "Please provide an image.", None, None
    
    # Convert to PIL grayscale
    pil_image = Image.fromarray(image).convert('L')
    
    # Invert colors: sketchpad has black bg + white text, BTTR expects white bg + black text
    pil_image = pil_image.point(lambda x: 255 - x)
    
    # Process image
    from torchvision.transforms import ToTensor
    to_tensor = ToTensor()
    image_tensor = to_tensor(pil_image).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    # Create mask (no padding for single image)
    h, w = image_tensor.shape[2:]
    mask = torch.zeros(1, h, w, dtype=torch.bool, device=device)
    
    # Beam search
    model.eval()
    with torch.no_grad():
        preds, attentions, feat_h, feat_w = beam_search_batch(
            model, image_tensor, mask, beam_size=beam_size, 
            max_len=max_length, alpha=1.0, vocab=vocab
        )
    
    # Convert indices to tokens
    pred_indices = preds[0]
    if torch.is_tensor(pred_indices):
        pred_indices = pred_indices.tolist()
    
    latex_tokens = []
    for idx in pred_indices:
        if idx == vocab.EOS_IDX:
            break
        if idx != vocab.PAD_IDX and idx != vocab.SOS_IDX:
            latex_tokens.append(vocab.idx2word[idx])
    
    latex = ' '.join(latex_tokens)
    rendered_latex = f"$${latex}$$"
    
    # Create attention visualization
    attention_img = create_bttr_attention_visualization(latex_tokens, attentions)
    
    return latex, rendered_latex, attention_img


def create_bttr_attention_visualization(latex_tokens, attentions):
    """Create attention visualization for BTTR"""
    if not attentions or len(attentions) == 0:
        return None
    
    try:
        # attentions is a list of lists: [step][layer] -> tensor
        # Take the last step's attention from the last decoder layer
        last_step_attentions = attentions[-1]
        if isinstance(last_step_attentions, list):
            attn = last_step_attentions[-1]
            attn = attn[0].mean(dim=0)  # [tgt_len, src_len]
        else:
            attn = last_step_attentions[0].mean(dim=0)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(attn.cpu().numpy(), aspect='auto', cmap='jet')
        ax.set_title('BTTR Attention Weights')
        ax.set_xlabel('Source Position')
        ax.set_ylabel('Target Position')
        plt.tight_layout()
        plt.savefig('attention_maps_bttr.png', bbox_inches='tight', dpi=150)
        plt.close()
        return Image.open('attention_maps_bttr.png')
    except Exception as e:
        print(f"Attention visualization error: {e}")
        return None


# ===== CAN functions =====
def preprocess_image_for_can(image_np):
    """Preprocess a numpy image for CAN model"""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    white = np.sum(thresh == 255)
    black = np.sum(thresh == 0)
    if white > black:
        thresh = 255 - thresh

    denoised = cv2.medianBlur(thresh, 3)

    coords = cv2.findNonZero(denoised)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(denoised.shape[1] - x, w + 2 * pad)
        h = min(denoised.shape[0] - y, h + 2 * pad)
        cropped = denoised[y:y+h, x:x+w]
    else:
        cropped = denoised

    h, w = cropped.shape
    new_w = int((INPUT_HEIGHT / h) * w)

    if new_w > INPUT_WIDTH:
        resized = cv2.resize(cropped, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(cropped, (new_w, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
        padded = np.zeros((INPUT_HEIGHT, INPUT_WIDTH), dtype=np.uint8)
        x_offset = (INPUT_WIDTH - new_w) // 2
        padded[:, x_offset:x_offset + new_w] = resized
        resized = padded

    return resized


def recognize_single_image_can(model, image_np, vocab, device, max_length=150, visualize_attention=False):
    """Recognize a single image using CAN model"""
    processed_img = preprocess_image_for_can(image_np)

    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),
        A.pytorch.ToTensorV2()
    ])
    processed_img = np.expand_dims(processed_img, axis=-1)
    image_tensor = transform(image=processed_img)['image'].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        predictions, attention_weights = model.recognize(
            image_tensor,
            max_length=max_length,
            start_token=vocab.start_token,
            end_token=vocab.end_token,
            beam_width=5
        )

    latex_tokens = []
    for idx in predictions:
        if idx == vocab.end_token:
            break
        if idx != vocab.start_token:
            latex_tokens.append(vocab.idx2word[idx])

    latex = ' '.join(latex_tokens)

    attention_img = None
    if visualize_attention and attention_weights is not None and len(latex_tokens) > 0:
        attention_img = visualize_attention_maps_can(image_np, attention_weights, latex_tokens)

    return latex, attention_img


def visualize_attention_maps_can(orig_image_np, attention_weights, latex_tokens, max_cols=4):
    """Visualize attention maps for CAN model"""
    try:
        if len(orig_image_np.shape) == 3:
            orig_image = Image.fromarray(orig_image_np)
        else:
            orig_image = Image.fromarray(orig_image_np).convert('RGB')
        
        orig_w, orig_h = orig_image.size
        ratio = INPUT_HEIGHT / INPUT_WIDTH

        num_tokens = len(latex_tokens)
        num_cols = min(max_cols, num_tokens)
        num_rows = max(1, int(np.ceil(num_tokens / num_cols)))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 4))
        axes = np.array(axes).reshape(-1)

        for i, (token, attn) in enumerate(zip(latex_tokens, attention_weights)):
            ax = axes[i]
            attn = attn[0:1].squeeze(0)
            attn_len = attn.shape[0]
            attn_w = int(np.sqrt(attn_len / ratio))
            attn_h = int(np.sqrt(attn_len * ratio))

            attn = attn.view(1, 1, attn_h, attn_w)
            interp_w = int(orig_h / ratio)
            attn = F.interpolate(attn, size=(orig_h, interp_w), mode='bilinear', align_corners=False)
            attn = attn.squeeze().cpu().numpy()

            if interp_w > orig_w:
                start = (interp_w - orig_w) // 2
                attn = attn[:, start:start + orig_w]
            elif interp_w < orig_w:
                attn = cv2.resize(attn, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            ax.imshow(orig_image)
            ax.imshow(attn, cmap='jet', alpha=0.4)
            ax.set_title(f'{token}', fontsize=10)
            ax.axis('off')

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        plt.savefig('attention_maps_can.png', bbox_inches='tight', dpi=150)
        plt.close()
        return Image.open('attention_maps_can.png')
    except Exception as e:
        print(f"CAN attention visualization error: {e}")
        return None


def recognize_and_display_can(model, vocab, device, image):
    """Process image with CAN model"""
    if isinstance(image, dict):
        image = image.get("composite")
    if image is None:
        return "Please provide an image.", None, None

    latex_string, attention_img = recognize_single_image_can(
        model, image, vocab, device, visualize_attention=True
    )
    rendered_latex = f"$${latex_string}$$"
    return latex_string, rendered_latex, attention_img


# ===== Main processing function =====
def process_input(model_choice, input_type, uploaded_image, sketchpad_data):
    """Select model and process image"""
    try:
        # Get image
        if input_type == "Upload image":
            image_to_process = uploaded_image
        elif input_type == "Use sketchpad":
            image_to_process = sketchpad_data
        else:
            return "Invalid input type.", None, None
        
        # Check if image is valid
        if image_to_process is None:
            return "Please draw or upload an image first.", None, None
        if isinstance(image_to_process, dict):
            composite = image_to_process.get("composite")
            if composite is None:
                return "Please draw something on the sketchpad first.", None, None
            image_to_process = composite
        
        # Select model
        if model_choice == "WAP":
            return recognize_and_display_wap(wap_model, wap_vocab, device, image_to_process)
        elif model_choice == "BTTR":
            return recognize_single_image_bttr(bttr_model, image_to_process, bttr_vocab, device)
        elif model_choice == "CAN":
            return recognize_and_display_can(can_model, can_vocab, device, image_to_process)
        else:
            return "Unknown model selected.", None, None
    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, None, None


# ===== Gradio UI =====
if __name__ == '__main__':
    with gr.Blocks(title="Offline Handwritten Mathematical Expression Recognition") as demo:
        gr.Markdown(
            """
            # 🧮 Offline Handwritten Mathematical Expression Recognition
            Upload an image or use the sketchpad to recognize a handwritten mathematical expression.
            Choose between **WAP**, **BTTR** (Transformer-based), or **CAN** (Counting-Aware Network) model.
            """
        )

        with gr.Row():
            model_choice = gr.Dropdown(
                choices=["WAP", "BTTR", "CAN"],
                label="Select Model",
                value="WAP",
                interactive=True
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

        process_button = gr.Button("Recognize", variant="primary")

        with gr.Column():
            latex_output = gr.Textbox(
                label="Recognized LaTeX", interactive=False)
            markdown_output = gr.Markdown(
                label="Rendered LaTeX")
            attention_map_output = gr.Image(type='pil', label="Attention maps")

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
            fn=process_input,
            inputs=[model_choice, input_choice, upload_img, sketchpad],
            outputs=[latex_output, markdown_output, attention_map_output]
        )

    demo.launch()
