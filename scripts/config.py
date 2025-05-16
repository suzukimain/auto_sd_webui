import requests
from typing import Union

import gradio as gr
from huggingface_hub import hf_api

from modules import script_callbacks

tabs_list = ["checkpoint", "textual inversion", "Lora", "controlnet"]



def quickly_search_huggingface(search_word: str, **kwargs) -> Union[str, None]:
    r"""
    huggingface search engine with emphasis on speed

    Parameters:
        search_word (`str`):
            The search query string.
        pipeline_tag (`str`, *optional*):
            Tag to filter models by pipeline.
        limit (`int`, *optional*, defaults to `100`):
            The number of candidates to retrieve.
        token (`str`, *optional*):
            API token for Hugging Face authentication.
        gated (`bool`, *optional*, defaults to `False` ):
            A boolean to filter models on the Hub that are gated or not.
        
    Returns:
        `str`: The name of the model.
    """
    # Extract additional parameters from kwargs
    pipeline_tag = kwargs.pop("pipeline_tag", None)
    limit = kwargs.pop("limit", 1)
    token = kwargs.pop("token", None)
    gated = kwargs.pop("gated", False)

    # Get model data from HF API
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
        repo_id = [model.id for model in hf_models]
        return repo_id[0].split("/")[-1]
    return None


def quickly_search_civitai(search_word: str, **kwargs) -> Union[str, None]:
    r"""
    civitai search engine with emphasis on speed

    Parameters:
        search_word (`str`):
            The search query string.
        model_type (`str`, *optional*, defaults to `Checkpoint`):
            The type of model to search for.
        sort (`str`, *optional*):
            The order in which you wish to sort the results(for example, `Highest Rated`, `Most Downloaded`, `Newest`).
        base_model (`str`, *optional*):
            The base model to filter by.
        token (`str`, *optional*):
            API token for Civitai authentication.

    Returns:
        `str`: The name of the model.
    """

    # Extract additional parameters from kwargs
    model_type = kwargs.pop("model_type", "Checkpoint")
    sort = kwargs.pop("sort", None)
    base_model = kwargs.pop("base_model", None)
    token = kwargs.pop("token", None)
    
    # Set up parameters and headers for the CivitAI API request
    params = {
        "query": search_word,
        "types": model_type,
        "limit": 1,
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
        # Make the request to the CivitAI API
        response = requests.get("https://civitai.com/api/v1/models", params=params, headers=headers)
        response.raise_for_status()
        data = response.json()
    except:
        return None
    else:
        return data["items"][0]["name"]

def create_tab(tab):
    with gr.Row():
        with gr.Column():
            search_word = gr.Textbox(label="Search", placeholder="Enter a keyword to search")
            search_button = gr.Button("Search")
            search_result = gr.Textbox(label="Result", interactive=False)
            search_button.click(
                lambda x: quickly_search_huggingface(x) if tab == "checkpoint" else quickly_search_civitai(x),
                inputs=[search_word],
                outputs=[search_result],
            )
        with gr.Column():
            search_word2 = gr.Textbox(label="Search", placeholder="Enter a keyword to search")
            search_button2 = gr.Button("Search")
            search_result2 = gr.Textbox(label="Result", interactive=False)
            search_button2.click(
                lambda x: quickly_search_huggingface(x) if tab == "checkpoint" else quickly_search_civitai(x),
                inputs=[search_word2],
                outputs=[search_result2],
            )

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as search_tab:
        with gr.Tabs(elem_id="images_history_tab") as tabs:
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False) :
                        create_tab(tab)
                         
    return (search_tab , "Image Browser", "images_history"),



script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_ui_tabs(on_ui_tabs)