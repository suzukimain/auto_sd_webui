import gradio as gr

from modules import script_callbacks


tabs_list = ["checkpoint", "textual inversion", "Lora", "controlnet"]


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as search_ui:
        with gr.Tabs(elem_id="search_tab"):
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False) :
                        search_button = gr.Button(f"Search {tab}")
        
    return (search_ui , "Search", "auto_sd_webui")

script_callbacks.on_ui_tabs(on_ui_tabs)
