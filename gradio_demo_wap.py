from models.wap.wap import WAP
import os
import torch
from PIL import Image
import numpy as np
import gradio as gr
from functools import partial


from models.wap.wap_dataloader import Vocabulary
torch.serialization.add_safe_globals([Vocabulary])
from models.wap.wap_eval import recognize_single_image as recognize_single_image_wap, load_checkpoint as load_checkpoint_wap

os.environ['QT_QPA_PLATFORM'] = 'offscreen'



def recognize_single_image_wrapper_wap(model: WAP, image: Image.Image, vocab, device, max_length=150, visualize_attention=False):
    # need to save the image to a temporary file for image processing
    temp_img_path = 'temp_input_image.png'
    image.save(temp_img_path)
    try:
        output = recognize_single_image_wap(model, temp_img_path, vocab, device, max_length, visualize_attention)
    finally:
        if os.path.exists(temp_img_path):
            os.remove(temp_img_path)
    return output


def recognize_and_display_wap(model, vocab, device, image):
    '''
    Process the input image and return the LaTeX string, rendered image, and attention maps
    '''
    if isinstance(image, dict):
        image = image.get("composite")

    if image is None:
        return "Please provide an image.", None, None

    pil_image = Image.fromarray(image)

    latex_string = recognize_single_image_wrapper_wap(
        model, pil_image, vocab, device, visualize_attention=True)
    rendered_latex = f"$${latex_string}$$"
    
    # Check if attention map was created
    if os.path.exists('attention_maps_wap.png'):
        attention_maps_image = Image.open('attention_maps_wap.png')
    else:
        attention_maps_image = None

    return latex_string, rendered_latex, attention_maps_image


def process_input_wap(input_type, uploaded_image, sketchpad_data, model, vocab, device):
    """
    Wrapper function to select the correct image input based on the dropdown menu
    """
    if input_type == "Upload image":
        image_to_process = uploaded_image
    elif input_type == "Use sketchpad":
        image_to_process = sketchpad_data
    else:
        raise Exception('invalid input type')

    return recognize_and_display_wap(model, vocab, device, image_to_process)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    checkpoint_path = 'final_trained_models/wap_best.pth'
    model, vocab = load_checkpoint_wap(checkpoint_path, device)

    with gr.Blocks(title="Offline HMER - WAP Model") as demo:
        gr.Markdown(
            """
            # 🧮 WAP Model - Handwritten Mathematical Expression Recognition
            Upload an image or use the sketchpad to recognize a handwritten mathematical expression.
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

        black_background = np.zeros((256, 1024, 3), dtype=np.uint8)  # RGB black

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
            fn=partial(process_input_wap, model=model, vocab=vocab, device=device),
            inputs=[input_choice, upload_img, sketchpad],
            outputs=[latex_output, markdown_output, attention_map_output]
        )

    demo.launch()
