import gradio as gr
import torch
from fast_model import AlexNet

def give_image(image_path):
    if image_path:
        return image_path
    return None

def select_model(model_path):
    # print(model_path)
    return None

def predict(model_path, image_path):
    if not image_path:
        return "No image selected"
    if not model_path:
        return "No model selected"

    #load model path
    model_state_dict = torch.load(model_path, weights_only=False)
    # print(model_state_dict)
    model = AlexNet()

    # this step does not work
    # Missing key(s) in state_dict: "conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "conv3.weight", "conv3.bias", "conv4.weight", "conv4.bias", "conv5.weight", "conv5.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias".
    # Unexpected key(s) in state_dict: "model", "opt".
    model.load_state_dict(model_state_dict)

    # Needed for inference (https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-state-dict-recommended)
    model.eval()

    # load image
    # resize 224x224
    # predict image
    return f"it doesn't work yet"

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            file_explorer = gr.FileExplorer(root_dir="images", file_count="single")
        with gr.Column():
            image_display = gr.Image(label="Selected Image")
    with gr.Row():
        with gr.Column():
            model_selector = gr.FileExplorer(root_dir="models", file_count="single")
            predict_btn = gr.Button(value="Predict")
        with gr.Column():
            prediction = gr.Textbox(label="Prediction")

    file_explorer.change(fn=give_image, inputs=file_explorer, outputs=image_display)
    model_selector.change(fn=select_model, inputs=model_selector)
    predict_btn.click(fn=predict, inputs=[model_selector, file_explorer], outputs=[prediction])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
