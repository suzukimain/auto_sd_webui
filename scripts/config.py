import requests
from typing import Union, List

from huggingface_hub import hf_api
import gradio as gr

from modules import script_callbacks
from fastapi import FastAPI
from fastapi.responses import JSONResponse

hub_name_list = ["huggingface", "civitai"]

def quickly_search_huggingface(search_word: str, **kwargs) -> Union[str, None, List[str]]:
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
        repo_ids = [model.id for model in hf_models]
        names = [repo_id.split("/")[-1] for repo_id in repo_ids]
        return names
    return []

def quickly_search_civitai(search_word: str, **kwargs) -> Union[str, None, List[str]]:
    model_type = kwargs.pop("model_type", "Checkpoint")
    sort = kwargs.pop("sort", None)
    base_model = kwargs.pop("base_model", None)
    token = kwargs.pop("token", None)
    limit = kwargs.pop("limit", 5)
    
    params = {
        "query": search_word,
        "types": model_type,
        "limit": limit,
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
        names = [item["name"] for item in data.get("items", [])]
        return names
    except:
        return []


def on_app_started(app: FastAPI):
    @app.get("/civitai_candidates")
    async def civitai_candidates(q: str):
        result = quickly_search_civitai(q, limit=5)
        return JSONResponse(content=result)

    @app.get("/huggingface_candidates")
    async def huggingface_candidates(q: str):
        result = quickly_search_huggingface(q, limit=5)
        return JSONResponse(content=result)

script_callbacks.on_app_started(on_app_started)

def create_tab(hub_name):
    with gr.Row():
        search_word = gr.Textbox(label="Search", placeholder="Enter a keyword to search", element_id="search_box")
        search_button = gr.Button("Search")
        search_result = gr.Textbox(label="Result", interactive=False)
        search_fn = quickly_search_huggingface if hub_name == "huggingface" else quickly_search_civitai
        search_button.click(
            lambda x: search_fn(x, limit=1)[0] if search_fn(x, limit=1) else "",
            inputs=[search_word],
            outputs=[search_result],
        )

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as search_tab:
        with gr.Tabs(elem_id="Search_tab"):
            for hub_name in hub_name_list:
                with gr.Tab(hub_name):
                    with gr.Blocks(analytics_enabled=False):
                        create_tab(hub_name)
    return (search_tab, "Search", "Search_ui"),

script_callbacks.on_ui_tabs(on_ui_tabs)