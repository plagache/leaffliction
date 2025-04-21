import gradio as gr

def give_image(file_path):
    if file_path:
        return file_path
    return None

with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                file_explorer = gr.FileExplorer(root_dir="images", file_count="single")
            with gr.Column():
                image_display = gr.Image(label="Selected Image")

        file_explorer.change(
        fn=give_image,
        inputs=file_explorer,
        outputs=image_display
    )



if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, debug=False)
