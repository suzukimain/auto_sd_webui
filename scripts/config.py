import os
import requests
from typing import Union
from dataclasses import dataclass, field, asdict
import re
from urllib.parse import urlparse
import gradio as gr

from huggingface_hub import hf_api


from modules import scripts, script_callbacks
current_extension_directory = scripts.basedir()

tabs_list = ["checkpoint"] # "textual inversion", "Lora", "controlnet"


VALID_URL_PREFIXES = ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]

@dataclass
class RepoStatus:
    r"""
    Data class for storing repository status information.

    Attributes:
        repo_id (`str`):
            The name of the repository.
        repo_hash (`str`):
            The hash of the repository.
        version (`str`):
            The version ID of the repository.
    """

    repo_id: str = ""
    repo_hash: str = ""
    version: str = ""


@dataclass
class ModelStatus:
    r"""
    Data class for storing model status information.

    Attributes:
        search_word (`str`):
            The search word used to find the model.
        download_url (`str`):
            The URL to download the model.
        file_name (`str`):
            The name of the model file.
        local (`bool`):
            Whether the model exists locally
        site_url (`str`):
            The URL of the site where the model is hosted.
    """

    search_word: str = ""
    download_url: str = ""
    file_name: str = ""
    local: bool = False
    site_url: str = ""


@dataclass
class ExtraStatus:
    r"""
    Data class for storing extra status information.

    Attributes:
        trained_words (`str`):
            The words used to trigger the model
    """

    trained_words: Union[List[str], None] = None


@dataclass
class SearchResult:
    r"""
    Data class for storing model data.

    Attributes:
        model_path (`str`):
            The path to the model.
        loading_method (`str`):
            The type of loading method used for the model ( None or 'from_single_file' or 'from_pretrained')
        checkpoint_format (`str`):
            The format of the model checkpoint (`single_file` or `diffusers`).
        repo_status (`RepoStatus`):
            The status of the repository.
        model_status (`ModelStatus`):
            The status of the model.
    """

    model_path: str = ""
    loading_method: Union[str, None] = None
    checkpoint_format: Union[str, None] = None
    repo_status: RepoStatus = field(default_factory=RepoStatus)
    model_status: ModelStatus = field(default_factory=ModelStatus)
    extra_status: ExtraStatus = field(default_factory=ExtraStatus)


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
    





# Copied diffusers/loaders/single_file_utils.py
def is_valid_url(url):
    result = urlparse(url)
    if result.scheme and result.netloc:
        return True

    return False

# Based on diffusers/loaders/single_file_utils.py
def _extract_repo_id_and_weights_name(pretrained_model_name_or_path):
    if not is_valid_url(pretrained_model_name_or_path):
        raise ValueError("Invalid `pretrained_model_name_or_path` provided. Please set it to a valid URL.")

    pattern = r"([^/]+)/([^/]+)/(?:blob/main/)?(.+)"
    weights_name = None
    repo_id = (None,)
    for prefix in ["https://huggingface.co/", "huggingface.co/", "hf.co/", "https://hf.co/"]:
        pretrained_model_name_or_path = pretrained_model_name_or_path.replace(prefix, "")
    match = re.match(pattern, pretrained_model_name_or_path)
    if not match:
        logger.warning("Unable to identify the repo_id and weights_name from the provided URL.")
        return repo_id, weights_name

    repo_id = f"{match.group(1)}/{match.group(2)}"
    weights_name = match.group(3)

    return repo_id, weights_name

def get_keyword_types(keyword):
    r"""
    Determine the type and loading method for a given keyword.

    Parameters:
        keyword (`str`):
            The input keyword to classify.

    Returns:
        `dict`: A dictionary containing the model format, loading method,
                and various types and extra types flags.
    """

    # Initialize the status dictionary with default values
    status = {
        "checkpoint_format": None,
        "loading_method": None,
        "type": {
            "other": False,
            "hf_url": False,
            "hf_repo": False,
            "civitai_url": False,
            "local": False,
        },
        "extra_type": {
            "url": False,
            "missing_model_index": None,
        },
    }

    # Check if the keyword is an HTTP or HTTPS URL
    status["extra_type"]["url"] = bool(re.search(r"^(https?)://", keyword))

    # Check if the keyword is a file
    if os.path.isfile(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = "from_single_file"

    # Check if the keyword is a directory
    elif os.path.isdir(keyword):
        status["type"]["local"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"
        if not os.path.exists(os.path.join(keyword, "model_index.json")):
            status["extra_type"]["missing_model_index"] = True

    # Check if the keyword is a Civitai URL
    elif keyword.startswith("https://civitai.com/"):
        status["type"]["civitai_url"] = True
        status["checkpoint_format"] = "single_file"
        status["loading_method"] = None

    # Check if the keyword starts with any valid URL prefixes
    elif any(keyword.startswith(prefix) for prefix in VALID_URL_PREFIXES):
        repo_id, weights_name = _extract_repo_id_and_weights_name(keyword)
        if weights_name:
            status["type"]["hf_url"] = True
            status["checkpoint_format"] = "single_file"
            status["loading_method"] = "from_single_file"
        else:
            status["type"]["hf_repo"] = True
            status["checkpoint_format"] = "diffusers"
            status["loading_method"] = "from_pretrained"

    # Check if the keyword matches a Hugging Face repository format
    elif re.match(r"^[^/]+/[^/]+$", keyword):
        status["type"]["hf_repo"] = True
        status["checkpoint_format"] = "diffusers"
        status["loading_method"] = "from_pretrained"

    # If none of the above apply
    else:
        status["type"]["other"] = True
        status["checkpoint_format"] = None
        status["loading_method"] = None

    return status


def file_downloader(
    url,
    save_path,
    **kwargs,
) -> None:
    """
    Downloads a file from a given URL and saves it to the specified path.

    parameters:
        url (`str`):
            The URL of the file to download.
        save_path (`str`):
            The local path where the file will be saved.
        resume (`bool`, *optional*, defaults to `False`):
            Whether to resume an incomplete download.
        headers (`dict`, *optional*, defaults to `None`):
            Dictionary of HTTP Headers to send with the request.
        proxies (`dict`, *optional*, defaults to `None`):
            Dictionary mapping protocol to the URL of the proxy passed to `requests.request`.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether to force the download even if the file already exists.
        displayed_filename (`str`, *optional*):
            The filename of the file that is being downloaded. Value is used only to display a nice progress bar. If
            not set, the filename is guessed from the URL or the `Content-Disposition` header.

    returns:
        None
    """

    # Get optional parameters from kwargs, with their default values
    resume = kwargs.pop("resume", False)
    headers = kwargs.pop("headers", None)
    proxies = kwargs.pop("proxies", None)
    force_download = kwargs.pop("force_download", False)
    displayed_filename = kwargs.pop("displayed_filename", None)

    # Default mode for file writing and initial file size
    mode = "wb"
    file_size = 0

    # Create directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Check if the file already exists at the save path
    if os.path.exists(save_path):
        if not force_download:
            # If the file exists and force_download is False, skip the download
            logger.info(f"File already exists: {save_path}, skipping download.")
            return None
        elif resume:
            # If resuming, set mode to append binary and get current file size
            mode = "ab"
            file_size = os.path.getsize(save_path)

    # Open the file in the appropriate mode (write or append)
    with open(save_path, mode) as model_file:
        # Call the http_get function to perform the file download
        return http_get(
            url=url,
            temp_file=model_file,
            resume_size=file_size,
            displayed_filename=displayed_filename,
            headers=headers,
            proxies=proxies,
            **kwargs,
        )

class BaseConfig:
    def search_huggingface(search_word: str, **kwargs) -> Union[str, SearchResult, None]:
        r"""
        Downloads a model from Hugging Face.

        Parameters:
            search_word (`str`):
                The search query string.
            revision (`str`, *optional*):
                The specific version of the model to download.
            checkpoint_format (`str`, *optional*, defaults to `"single_file"`):
                The format of the model checkpoint.
            download (`bool`, *optional*, defaults to `False`):
                Whether to download the model.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force the download if the model already exists.
            include_params (`bool`, *optional*, defaults to `False`):
                Whether to include parameters in the returned data.
            pipeline_tag (`str`, *optional*):
                Tag to filter models by pipeline.
            token (`str`, *optional*):
                API token for Hugging Face authentication.
            gated (`bool`, *optional*, defaults to `False` ):
                A boolean to filter models on the Hub that are gated or not.
            skip_error (`bool`, *optional*, defaults to `False`):
                Whether to skip errors and return None.

        Returns:
            `Union[str,  SearchResult, None]`: The model path or  SearchResult or None.
        """
        # Extract additional parameters from kwargs
        revision = kwargs.pop("revision", None)
        checkpoint_format = kwargs.pop("checkpoint_format", "single_file")
        download = kwargs.pop("download", False)
        force_download = kwargs.pop("force_download", False)
        include_params = kwargs.pop("include_params", False)
        pipeline_tag = kwargs.pop("pipeline_tag", None)
        token = kwargs.pop("token", None)
        gated = kwargs.pop("gated", False)
        skip_error = kwargs.pop("skip_error", False)

        file_list = []
        hf_repo_info = {}
        hf_security_info = {}
        model_path = ""
        repo_id, file_name = "", ""
        diffusers_model_exists = False

        # Get the type and loading method for the keyword
        search_word_status = get_keyword_types(search_word)

        if search_word_status["type"]["hf_repo"]:
            hf_repo_info = hf_api.model_info(repo_id=search_word, securityStatus=True)
            if download:
                model_path = DiffusionPipeline.download(
                    search_word,
                    revision=revision,
                    token=token,
                    force_download=force_download,
                    **kwargs,
                )
            else:
                model_path = search_word
        elif search_word_status["type"]["hf_url"]:
            repo_id, weights_name = _extract_repo_id_and_weights_name(search_word)
            if download:
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=weights_name,
                    force_download=force_download,
                    token=token,
                )
            else:
                model_path = search_word
        elif search_word_status["type"]["local"]:
            model_path = search_word
        elif search_word_status["type"]["civitai_url"]:
            if skip_error:
                return None
            else:
                raise ValueError(
                    "The URL for Civitai is invalid with `for_hf`. Please use `for_civitai` instead."
                )
        else:
            # Get model data from HF API
            hf_models = hf_api.list_models(
                search=search_word,
                direction=-1,
                limit=100,
                fetch_config=True,
                pipeline_tag=pipeline_tag,
                full=True,
                gated=gated,
                token=token,
            )
            model_dicts = [asdict(value) for value in list(hf_models)]

            # Loop through models to find a suitable candidate
            for repo_info in model_dicts:
                repo_id = repo_info["id"]
                file_list = []
                hf_repo_info = hf_api.model_info(repo_id=repo_id, securityStatus=True)
                # Lists files with security issues.
                hf_security_info = hf_repo_info.security_repo_status
                exclusion = [issue["path"] for issue in hf_security_info["filesWithIssues"]]

                # Checks for multi-folder diffusers model or valid files (models with security issues are excluded).
                if hf_security_info["scansDone"]:
                    for info in repo_info["siblings"]:
                        file_path = info["rfilename"]
                        if "model_index.json" == file_path and checkpoint_format in [
                            "diffusers",
                            "all",
                        ]:
                            diffusers_model_exists = True
                            break

                        elif (
                            any(file_path.endswith(ext) for ext in EXTENSION)
                            and not any(config in file_path for config in CONFIG_FILE_LIST)
                            and not any(exc in file_path for exc in exclusion)
                            and os.path.basename(os.path.dirname(file_path))
                            not in DIFFUSERS_CONFIG_DIR
                        ):
                            file_list.append(file_path)

                # Exit from the loop if a multi-folder diffusers model or valid file is found
                if diffusers_model_exists or file_list:
                    break
            else:
                # Handle case where no models match the criteria
                if skip_error:
                    return None
                else:
                    raise ValueError(
                        "No models matching your criteria were found on huggingface."
                    )

            if diffusers_model_exists:
                if download:
                    model_path = DiffusionPipeline.download(
                        repo_id,
                        token=token,
                        **kwargs,
                    )
                else:
                    model_path = repo_id

            elif file_list:
                # Sort and find the safest model
                file_name = next(
                    (
                        model
                        for model in sorted(file_list, reverse=True)
                        if re.search(r"(?i)[-_](safe|sfw)", model)
                    ),
                    file_list[0],
                )

                if download:
                    model_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file_name,
                        revision=revision,
                        token=token,
                        force_download=force_download,
                    )

        # `pathlib.PosixPath` may be returned
        if model_path:
            model_path = str(model_path)

        if file_name:
            download_url = f"https://huggingface.co/{repo_id}/blob/main/{file_name}"
        else:
            download_url = f"https://huggingface.co/{repo_id}"

        output_info = get_keyword_types(model_path)

        if include_params:
            return SearchResult(
                model_path=model_path or download_url,
                loading_method=output_info["loading_method"],
                checkpoint_format=output_info["checkpoint_format"],
                repo_status=RepoStatus(
                    repo_id=repo_id, repo_hash=hf_repo_info.sha, version=revision
                ),
                model_status=ModelStatus(
                    search_word=search_word,
                    site_url=download_url,
                    download_url=download_url,
                    file_name=file_name,
                    local=download,
                ),
                extra_status=ExtraStatus(trained_words=None),
            )

        else:
            return model_path

    def search_civitai(search_word: str, **kwargs) -> Union[str, SearchResult, None]:
        r"""
        Downloads a model from Civitai.

        Parameters:
            search_word (`str`):
                The search query string.
            model_type (`str`, *optional*, defaults to `Checkpoint`):
                The type of model to search for.
            sort (`str`, *optional*):
                The order in which you wish to sort the results(for example, `Highest Rated`, `Most Downloaded`, `Newest`).
            base_model (`str`, *optional*):
                The base model to filter by.
            download (`bool`, *optional*, defaults to `False`):
                Whether to download the model.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force the download if the model already exists.
            token (`str`, *optional*):
                API token for Civitai authentication.
            include_params (`bool`, *optional*, defaults to `False`):
                Whether to include parameters in the returned data.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            resume (`bool`, *optional*, defaults to `False`):
                Whether to resume an incomplete download.
            skip_error (`bool`, *optional*, defaults to `False`):
                Whether to skip errors and return None.

        Returns:
            `Union[str,  SearchResult, None]`: The model path or ` SearchResult` or None.
        """

        # Extract additional parameters from kwargs
        model_type = kwargs.pop("model_type", "Checkpoint")
        sort = kwargs.pop("sort", None)
        download = kwargs.pop("download", False)
        base_model = kwargs.pop("base_model", None)
        force_download = kwargs.pop("force_download", False)
        token = kwargs.pop("token", None)
        include_params = kwargs.pop("include_params", False)
        resume = kwargs.pop("resume", False)
        cache_dir = kwargs.pop("cache_dir", None)
        skip_error = kwargs.pop("skip_error", False)

        # Initialize additional variables with default values
        model_path = ""
        repo_name = ""
        repo_id = ""
        version_id = ""
        trainedWords = ""
        models_list = []
        selected_repo = {}
        selected_model = {}
        selected_version = {}
        civitai_cache_dir = cache_dir or os.path.join(CACHE_HOME, "Civitai")

        # Set up parameters and headers for the CivitAI API request
        params = {
            "query": search_word,
            "types": model_type,
            "limit": 20,
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
            response = requests.get(
                "https://civitai.com/api/v1/models", params=params, headers=headers
            )
            response.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise requests.HTTPError(f"Could not get elements from the URL: {err}")
        else:
            try:
                data = response.json()
            except AttributeError:
                if skip_error:
                    return None
                else:
                    raise ValueError("Invalid JSON response")

        # Sort repositories by download count in descending order
        sorted_repos = sorted(
            data["items"], key=lambda x: x["stats"]["downloadCount"], reverse=True
        )

        for selected_repo in sorted_repos:
            repo_name = selected_repo["name"]
            repo_id = selected_repo["id"]

            # Sort versions within the selected repo by download count
            sorted_versions = sorted(
                selected_repo["modelVersions"],
                key=lambda x: x["stats"]["downloadCount"],
                reverse=True,
            )
            for selected_version in sorted_versions:
                version_id = selected_version["id"]
                trainedWords = selected_version["trainedWords"]
                models_list = []
                # When searching for textual inversion, results other than the values entered for the base model may come up, so check again.
                if base_model is None or selected_version["baseModel"] in base_model:
                    for model_data in selected_version["files"]:
                        # Check if the file passes security scans and has a valid extension
                        file_name = model_data["name"]
                        if (
                            model_data["pickleScanResult"] == "Success"
                            and model_data["virusScanResult"] == "Success"
                            and any(file_name.endswith(ext) for ext in EXTENSION)
                            and os.path.basename(os.path.dirname(file_name))
                            not in DIFFUSERS_CONFIG_DIR
                        ):
                            file_status = {
                                "filename": file_name,
                                "download_url": model_data["downloadUrl"],
                            }
                            models_list.append(file_status)

                if models_list:
                    # Sort the models list by filename and find the safest model
                    sorted_models = sorted(
                        models_list, key=lambda x: x["filename"], reverse=True
                    )
                    selected_model = next(
                        (
                            model_data
                            for model_data in sorted_models
                            if bool(
                                re.search(r"(?i)[-_](safe|sfw)", model_data["filename"])
                            )
                        ),
                        sorted_models[0],
                    )

                    break
            else:
                continue
            break

        # Exception handling when search candidates are not found
        if not selected_model:
            if skip_error:
                return None
            else:
                raise ValueError(
                    "No model found. Please try changing the word you are searching for."
                )

        # Define model file status
        file_name = selected_model["filename"]
        download_url = selected_model["download_url"]

        # Handle file download and setting model information
        if download:
            # The path where the model is to be saved.
            model_path = os.path.join(
                str(civitai_cache_dir), str(repo_id), str(version_id), str(file_name)
            )
            # Download Model File
            file_downloader(
                url=download_url,
                save_path=model_path,
                resume=resume,
                force_download=force_download,
                displayed_filename=file_name,
                headers=headers,
                **kwargs,
            )

        else:
            model_path = download_url

        output_info = get_keyword_types(model_path)

        if not include_params:
            return model_path
        else:
            return SearchResult(
                model_path=model_path,
                loading_method=output_info["loading_method"],
                checkpoint_format=output_info["checkpoint_format"],
                repo_status=RepoStatus(
                    repo_id=repo_name, repo_hash=repo_id, version=version_id
                ),
                model_status=ModelStatus(
                    search_word=search_word,
                    site_url=f"https://civitai.com/models/{repo_id}?modelVersionId={version_id}",
                    download_url=download_url,
                    file_name=file_name,
                    local=output_info["type"]["local"],
                ),
                extra_status=ExtraStatus(trained_words=trainedWords or None),
            )







def download_model(model_name: str, **kwargs):
    save_path = os.path.join(scripts.basedir(), "webui/models/Stable-diffusion")
    auto_diffusers = importlib.import_module("auto_diffusers")
    search_civitai = getattr(auto_diffusers, "search_civitai")

    model_path = search_civitai(
        model_name,
        model_type="Checkpoint",
        cache_dir=save_path,
        include_params=True,
        **kwargs,
    )
    return f"download model: {model_path}"

def create_ui():
    with gr.Blocks() as demo:
        with gr.Row():
            textbox = gr.Textbox(label="Search box", placeholder="please input your search word")
            suggestion = gr.HTML(label="assist")
        textbox.change(fn=quickly_search_civitai, inputs=[textbox], outputs=[suggestion])

        search_button = gr.Button("Search")
        search_output = gr.Textbox(label="Result", interactive=False)
        search_button.click(fn=download_model, inputs=[textbox], outputs=[search_output])

    return demo

def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as search_ui:
        with gr.Tabs(elem_id="search_tab"):
            for tab in tabs_list:
                with gr.Tab(tab):
                    with gr.Blocks(analytics_enabled=False):
                        search_ui_content = create_ui()

    return (search_ui, "Search", "auto_sd_webui")

script_callbacks.on_ui_tabs(on_ui_tabs)