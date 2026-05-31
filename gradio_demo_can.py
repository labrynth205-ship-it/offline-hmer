import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'

import torch
from PIL import Image
import numpy as np
import gradio as gr
from functools import partial
import cv2
import albumentations as A
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, 'models/can')
from can import CAN, create_can_model
from can_dataloader import Vocabulary, process_img, before_padding, INPUT_HEIGHT, INPUT_WIDTH

torch.serialization.add_safe_globals([Vocabulary])


def load_checkpoint(checkpoint_path, device):
    """
    Load CAN checkpoint and return model and vocabulary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    vocab = checkpoint.get('vocab')
    if vocab is None:
        # Try to load vocab from a separate file
        vocab_path = 'models/can/hmer_vocab.pth'
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path)
            vocab = Vocabulary()
            vocab.word2idx = vocab_data['word2idx']
            vocab.idx2word = vocab_data['idx2word']
            vocab.idx = vocab_data['idx']
            vocab.pad_token = vocab.word2idx['<pad>']
            vocab.start_token = vocab.word2idx['<start>']
            vocab.end_token = vocab.word2idx['<end>']
            vocab.unk_token = vocab.word2idx['<unk>']
        else:
            raise ValueError(f"Vocabulary not found in checkpoint and {vocab_path} does not exist")

    # Initialize model with parameters from checkpoint
    hidden_size = checkpoint.get('hidden_size', 256)
    embedding_dim = checkpoint.get('embedding_dim', 256)
    use_coverage = checkpoint.get('use_coverage', True)

    model = create_can_model(
        num_classes=len(vocab),
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        use_coverage=use_coverage,
        pretrained_backbone=True,
        backbone_type='densenet'
    ).to(device)

    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Loaded CAN model from {checkpoint_path}")

    return model, vocab


def preprocess_image_for_can(image_np):
    """
    Preprocess a numpy image (from Gradio) for CAN model.
    CAN expects: grayscale, black background, white text, resized to (INPUT_HEIGHT, INPUT_WIDTH)
    """
    # Convert to grayscale
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np

    # Invert if needed: sketchpad has black bg + white text, CAN expects black bg + white text
    # So we keep as is (black background, white text)
    
    # Apply thresholding to clean up
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Ensure background is black (0) and text is white (255)
    white = np.sum(thresh == 255)
    black = np.sum(thresh == 0)
    if white > black:
        thresh = 255 - thresh

    # Clean up noise
    denoised = cv2.medianBlur(thresh, 3)

    # Find content bounding box
    coords = cv2.findNonZero(denoised)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding
        pad = 5
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(denoised.shape[1] - x, w + 2 * pad)
        h = min(denoised.shape[0] - y, h + 2 * pad)
        cropped = denoised[y:y+h, x:x+w]
    else:
        cropped = denoised

    # Resize maintaining aspect ratio
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
    """
    Recognize handwritten mathematical expression from a single image using CAN model
    """
    # Preprocess image
    processed_img = preprocess_image_for_can(image_np)

    # Transform
    transform = A.Compose([
        A.Normalize(mean=[0.0], std=[1.0]),
        A.pytorch.ToTensorV2()
    ])
    processed_img = np.expand_dims(processed_img, axis=-1)  # [H, W, 1]
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

    # Convert indices to LaTeX tokens
    latex_tokens = []
    for idx in predictions:
        if idx == vocab.end_token:
            break
        if idx != vocab.start_token:
            latex_tokens.append(vocab.idx2word[idx])

    latex = ' '.join(latex_tokens)

    # Visualize attention if requested
    attention_img = None
    if visualize_attention and attention_weights is not None:
        attention_img = visualize_attention_maps_can(image_np, attention_weights, latex_tokens)

    return latex, attention_img


def visualize_attention_maps_can(orig_image_np, attention_weights, latex_tokens, max_cols=4):
    """
    Visualize attention maps over the image for CAN model
    """
    if len(latex_tokens) == 0 or len(attention_weights) == 0:
        return None

    try:
        # Convert to PIL for display
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

            # Resize to original image dimensions
            attn = attn.view(1, 1, attn_h, attn_w)
            interp_w = int(orig_h / ratio)

            attn = F.interpolate(attn, size=(orig_h, interp_w), mode='bilinear', align_corners=False)
            attn = attn.squeeze().cpu().numpy()

            # Fix aspect ratio mismatch
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
        print(f"Attention visualization error: {e}")
        return None


def recognize_and_display_can(model, vocab, device, image):
    """
    Process the input image and return the LaTeX string, rendered image, and attention maps
    """
    if isinstance(image, dict):
        image = image.get("composite")

    if image is None:
        return "Please provide an image.", None, None

    latex_string, attention_img = recognize_single_image_can(
        model, image, vocab, device, visualize_attention=True
    )
    rendered_latex = f"$${latex_string}$$"

    return latex_string, rendered_latex, attention_img


def process_input_can(input_type, uploaded_image, sketchpad_data, model, vocab, device):
    """
    Wrapper function to select the correct image input based on the dropdown menu
    """
    if input_type == "Upload image":
        image_to_process = uploaded_image
    elif input_type == "Use sketchpad":
        image_to_process = sketchpad_data
    else:
        raise Exception('invalid input type')

    return recognize_and_display_can(model, vocab, device, image_to_process)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint_path = 'final_trained_models/p_densenet_can_best.pth'
    model, vocab = load_checkpoint(checkpoint_path, device)

    with gr.Blocks(title="Offline HMER - CAN Model") as demo:
        gr.Markdown(
            """
            # 🧮 CAN Model - Handwritten Mathematical Expression Recognition
            Upload an image or use the sketchpad to recognize a handwritten mathematical expression.
            **CAN (Counting-Aware Network)** uses a DenseNet backbone with multi-scale counting module.
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
            fn=partial(process_input_can, model=model, vocab=vocab, device=device),
            inputs=[input_choice, upload_img, sketchpad],
            outputs=[latex_output, markdown_output, attention_map_output]
        )

    demo.launch()
