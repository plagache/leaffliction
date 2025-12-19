import gradio as gr
from predict import predict_image, predict_dataset, get_accuracy, model_confusion
from pathlib import Path
from Transformation import transform_image
from Augmentation import display_images

def give_image(given_path):
    return given_path

def predict(model_path, image_path):
    if not model_path:
        gr.Warning("No model selected")
        return None
    if not image_path:
        gr.Warning("No image selected")
        return None

    model_path = Path(model_path)
    prediction = predict_image(image_path, model_path)
    transformed_images = transform_image(image_path)
    transformed_images = [item for item in transformed_images if "Masked" in item[0] or "Original" in item[0]]
    figure = display_images(f"Class predicted: {prediction}", transformed_images, show=False)
    return figure

def validate(model_path):
    if not model_path:
        return "No model selected", None

    model_path = Path(model_path)
    labels, predictions, classes = predict_dataset(model_path)
    accuracy = get_accuracy(labels, predictions)
    confusion_figure = model_confusion(labels, predictions, classes)

    return f"{accuracy.item() * 100:.2f}%", confusion_figure

with gr.Blocks() as demo:
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column(scale=1):
                predict_model_selector = gr.FileExplorer(root_dir="models", glob="*.pth", file_count="single", label="Model selection")
                file_explorer = gr.FileExplorer(root_dir="images", glob="**/*.JPG", file_count="single", max_height=360, label="Image selection")
                image_display = gr.Image(label="Selected Image", inputs=file_explorer)
                predict_btn = gr.Button(value="Predict")
            with gr.Column(scale=2):
                prediction = gr.Plot(label="Predicted image")

        file_explorer.change(fn=give_image, inputs=file_explorer, outputs=image_display)
        predict_btn.click(fn=predict, inputs=[predict_model_selector, file_explorer], outputs=[prediction])

    with gr.Tab("Validation"):
        with gr.Row():
            with gr.Column():
                valid_model_selector = gr.FileExplorer(root_dir="models", glob="*.pth", file_count="single", label="Model selection")
                validation_btn = gr.Button(value="Valid")
            with gr.Column():
                validation = gr.Textbox(label="Validation")
        confusion = gr.Plot(label="Confusion Matrix")

        validation_btn.click(fn=validate, inputs=valid_model_selector, outputs=[validation, confusion])


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
