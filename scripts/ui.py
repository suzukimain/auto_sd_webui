import gradio as gr


from .config import quickly_search_civitai
from modules import script_callbacks


tabs_list = ["checkpoint", "textual inversion", "Lora", "controlnet"]


def ui_checkpoint():
    with gr.Blocks() as demo:
        with gr.Row():
            textbox = gr.Textbox(label="search box", placeholder="please input your search word")
            suggestion = gr.HTML(label="candidate")
    
        textbox.change(fn=quickly_search_civitai, inputs=textbox, outputs=suggestion)
    
    return demo


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as search_ui:
        with gr.Tabs(elem_id="search_tab"):
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False) :
                        if not tab == "checkpoint":
                            search_button = gr.Button(f"Search {tab}")
                        else:#ui_checkpoint()
                            search_button = ui_checkpoint()

    return (search_ui , "Search", "auto_sd_webui"),

script_callbacks.on_ui_tabs(on_ui_tabs)
