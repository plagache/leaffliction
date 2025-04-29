import gradio as gr
import torch
from pytorch_inference import predict_image, prepare_model

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
    
    model, classes = prepare_model(model_path)
    prediction = predict_image(image_path, model, classes)
    return prediction

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
