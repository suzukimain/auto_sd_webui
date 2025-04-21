import requests

from huggingface_hub import hf_api

def quickly_search_huggingface(search_word: str, **kwargs) -> str:
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

    model_ids = [model.id for model in hf_models]
    return model_ids[0] 


def quickly_search_civitai(search_word: str, **kwargs) -> str:
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
    except requests.exceptions.HTTPError as err:
        raise requests.HTTPError(f"Could not get elements from the URL: {err}")
    else:
        try:
            data = response.json()
        except AttributeError:
            raise ValueError("Invalid JSON response")
    
    return data[0]["name"]
