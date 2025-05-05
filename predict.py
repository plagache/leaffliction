import gradio as gr
from pytorch_inference import predict_image, predict_dataset, get_accuracy, model_confusion
from pathlib import Path


def list_models():
    files: list[Path] = list(Path("models").rglob("*.pth"))
    models = []
    for model in files:
        if Path.is_file(model):
            name = model.name.split("-")[0]
            models.append((name, str(model)))
    return models


def give_image(image_path):
    if image_path:
        return image_path
    return None


def predict(model_path, image_path):
    if not image_path:
        return "No image selected"
    if not model_path:
        return "No model selected"

    prediction = predict_image(image_path, model_path)
    return prediction

def asdf(model_path):
    if not model_path:
        return "No model selected", None

    labels, predictions, classes = predict_dataset(model_path)
    la_retourne_a_tourner = get_accuracy(labels, predictions)
    confusion_figure = model_confusion(labels, predictions, classes)

    return la_retourne_a_tourner, confusion_figure

with gr.Blocks() as demo:
    with gr.Tab("Predict"):
        with gr.Row():
            with gr.Column():
                file_explorer = gr.FileExplorer(root_dir="images", file_count="single")
            with gr.Column():
                image_display = gr.Image(label="Selected Image")
        with gr.Row():
            with gr.Column():
                predict_model_selector = gr.Dropdown(choices=list_models(), label="select a model")
                predict_btn = gr.Button(value="Predict")
            with gr.Column():
                prediction = gr.Textbox(label="Prediction")

        file_explorer.change(fn=give_image, inputs=file_explorer, outputs=image_display)
        predict_model_selector.change(inputs=predict_model_selector)
        predict_btn.click(fn=predict, inputs=[predict_model_selector, file_explorer], outputs=[prediction])

    with gr.Tab("Validation"):
        with gr.Row():
            with gr.Column():
                valid_model_selector = gr.Dropdown(choices=list_models(), label="select a model")
                validation_btn = gr.Button(value="Valid")
            with gr.Column():
                validation = gr.Textbox(label="Validation")
        with gr.Row():
            confusion = gr.Plot(label="Confusion Matrix")

        validation_btn.click(fn=asdf, inputs=valid_model_selector, outputs=[validation, confusion])
        valid_model_selector.change(inputs=valid_model_selector)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
