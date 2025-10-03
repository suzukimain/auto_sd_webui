import requests
import time
from typing import Union, List

from huggingface_hub import hf_api
import gradio as gr

from modules import script_callbacks

tabs_list = ["checkpoint", "textual inversion", "Lora", "controlnet"]

def quickly_search_huggingface_candidates(search_word: str, **kwargs) -> List[str]:
    pipeline_tag = kwargs.pop("pipeline_tag", None)
    limit = kwargs.pop("limit", 5)
    token = kwargs.pop("token", None)
    gated = kwargs.pop("gated", False)

    hf_models = hf_api.list_models(
        search=search_word,
        direction=-1,
        limit=limit,
        pipeline_tag=pipeline_tag,
        fetch_config=False,
        full=False, 
        gated=gated,
        token=token,
    )
    if hf_models:
        return [model.id.split("/")[-1] for model in hf_models]
    return []

def quickly_search_civitai_candidates(search_word: str, **kwargs) -> List[str]:
    model_type = kwargs.pop("model_type", "Checkpoint")
    sort = kwargs.pop("sort", None)
    base_model = kwargs.pop("base_model", None)
    token = kwargs.pop("token", None)
    params = {
        "query": search_word,
        "types": model_type,
        "limit": 5,
    }
    if base_model is not None:
        if not isinstance(base_model, list):
            base_model = [base_model]
        params["baseModel"] = base_model
    if sort is not None:
        params["sort"] = sort
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.get("https://civitai.com/api/v1/models", params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except:
        return []
    else:
        return [item["name"] for item in data.get("items", [])]

def autocomplete_search(search_word, tab):
    time.sleep(3)
    if not search_word:
        return []
    if tab == "checkpoint":
        return quickly_search_huggingface_candidates(search_word)
    else:
        return quickly_search_civitai_candidates(search_word)

def create_tab(tab):
    with gr.Row():
        with gr.Column():
            search_word = gr.Textbox(label="Search", placeholder="Enter a keyword to search", interactive=True)
            candidates = gr.Dropdown(label="candidate", choices=[], interactive=True)
            search_result = gr.Textbox(label="Result", interactive=False)

            search_word.change(
                fn=autocomplete_search,
                inputs=[search_word, gr.State(tab)],
                outputs=[candidates],
                queue=True,
            )
            candidates.change(
                fn=lambda x: x,
                inputs=[candidates],
                outputs=[search_result],
            )

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as search_tab:
        with gr.Tabs(elem_id="Search_tab"):
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False):
                        create_tab(tab)
    return (search_tab , "Search", "Search_ui"),

script_callbacks.on_ui_tabs(on_ui_tabs)