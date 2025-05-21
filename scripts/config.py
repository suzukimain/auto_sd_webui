import requests
from typing import Union, List

from huggingface_hub import hf_api
import gradio as gr

from modules import script_callbacks,shared
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from pathlib import Path

from modules import scripts, shared

try:
    from modules.paths import extensions_dir, script_path

    # Webui root path
    FILE_DIR = Path(script_path).absolute()

    # The extension base path
    EXT_PATH = Path(extensions_dir).absolute()
except ImportError:
    # Webui root path
    FILE_DIR = Path().absolute()
    # The extension base path
    EXT_PATH = FILE_DIR.joinpath("extensions").absolute()

# Tags base path
TAGS_PATH = Path(scripts.basedir()).joinpath("tags").absolute()

# The path to the folder containing the wildcards and embeddings
try: # SD.Next
    WILDCARD_PATH = Path(shared.opts.wildcards_dir).absolute()
except Exception: # A1111
    WILDCARD_PATH = FILE_DIR.joinpath("scripts/wildcards").absolute()
EMB_PATH = Path(shared.cmd_opts.embeddings_dir).absolute()

# Forge Classic detection
try:
    from modules_forge.forge_version import version as forge_version
    IS_FORGE_CLASSIC = forge_version == "classic"
except ImportError:
    IS_FORGE_CLASSIC = False

# Forge Classic skips it
if not IS_FORGE_CLASSIC:
    try:
        HYP_PATH = Path(shared.cmd_opts.hypernetwork_dir).absolute()
    except AttributeError:
        HYP_PATH = None
else:
    HYP_PATH = None

try:
    LORA_PATH = Path(shared.cmd_opts.lora_dir).absolute()
except AttributeError:
    LORA_PATH = None

try:
    try:
        LYCO_PATH = Path(shared.cmd_opts.lyco_dir_backcompat).absolute()
    except:
        LYCO_PATH = Path(shared.cmd_opts.lyco_dir).absolute() # attempt original non-backcompat path
except AttributeError:
    LYCO_PATH = None


def find_ext_wildcard_paths():
    """Returns the path to the extension wildcards folder"""
    found = list(EXT_PATH.glob("*/wildcards/"))
    # Try to find the wildcard path from the shared opts
    try:
        from modules.shared import opts
    except ImportError:  # likely not in an a1111 context
        opts = None

    # Append custom wildcard paths
    custom_paths = [
        getattr(shared.cmd_opts, "wildcards_dir", None),    # Cmd arg from the wildcard extension
        getattr(opts, "wildcard_dir", None),                # Custom path from sd-dynamic-prompts
    ]
    for path in [Path(p).absolute() for p in custom_paths if p is not None]:
        if path.exists():
            found.append(path)

    return found


# The path to the extension wildcards folder
WILDCARD_EXT_PATHS = find_ext_wildcard_paths()

# The path to the temporary files
# In the webui root, on windows it exists by default, on linux it doesn't
STATIC_TEMP_PATH = FILE_DIR.joinpath("tmp").absolute()
TEMP_PATH = TAGS_PATH.joinpath("temp").absolute()  # Extension specific temp files

# Make sure these folders exist
if not TEMP_PATH.exists():
    TEMP_PATH.mkdir()
if not STATIC_TEMP_PATH.exists():
    STATIC_TEMP_PATH.mkdir()


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



def write_tag_base_path():
    """Writes the tag base path to a fixed location temporary file"""
    with open(STATIC_TEMP_PATH.joinpath('tagAutocompletePath.txt'), 'w', encoding="utf-8") as f:
        f.write(TAGS_PATH.as_posix())


def write_to_temp_file(name, data):
    """Writes the given data to a temporary file"""
    with open(TEMP_PATH.joinpath(name), 'w', encoding="utf-8") as f:
        f.write(('\n'.join(data)))


csv_files = []
csv_files_withnone = []
def update_tag_files(*args, **kwargs):
    """Returns a list of all potential tag files"""
    global csv_files, csv_files_withnone
    files = [str(t.relative_to(TAGS_PATH)) for t in TAGS_PATH.glob("*.csv") if t.is_file()]
    csv_files = files
    csv_files_withnone = ["None"] + files

json_files = []
json_files_withnone = []
def update_json_files(*args, **kwargs):
    """Returns a list of all potential json files"""
    global json_files, json_files_withnone
    files = [str(j.relative_to(TAGS_PATH)) for j in TAGS_PATH.glob("*.json") if j.is_file()]
    json_files = files
    json_files_withnone = ["None"] + files


# Write the tag base path to a fixed location temporary file
# to enable the javascript side to find our files regardless of extension folder name
if not STATIC_TEMP_PATH.exists():
    STATIC_TEMP_PATH.mkdir(exist_ok=True)

write_tag_base_path()
update_tag_files()
update_json_files()

sort_criteria = {
    "Name": lambda path, name, subpath: name.lower() if subpath else path.stem.lower(),
    "Date Modified (newest first)": lambda path, name, subpath: path.stat().st_mtime if path.exists() else name.lower(),
    "Date Modified (oldest first)": lambda path, name, subpath: path.stat().st_mtime if path.exists() else name.lower()
}

def on_ui_settings():
    TAC_SECTION = ("tac", "Tag Autocomplete")

    # Backwards compatibility for pre 1.3.0 webui versions
    if not (hasattr(shared.OptionInfo, "info") and callable(getattr(shared.OptionInfo, "info"))):
        def info(self, info):
            self.label += f" ({info})"
            return self
        shared.OptionInfo.info = info
    if not (hasattr(shared.OptionInfo, "needs_restart") and callable(getattr(shared.OptionInfo, "needs_restart"))):
        def needs_restart(self):
            self.label += " (Requires restart)"
            return self
        shared.OptionInfo.needs_restart = needs_restart

    # Dictionary of function options and their explanations
    frequency_sort_functions = {
        "Logarithmic (weak)": "Will respect the base order and slightly prefer often used tags",
        "Logarithmic (strong)": "Same as Logarithmic (weak), but with a stronger bias",
        "Usage first": "Will list used tags by frequency before all others",
    }

    tac_options = {
        # Main tag file
        "tac_tagFile": shared.OptionInfo("danbooru.csv", "Tag filename", gr.Dropdown, lambda: {"choices": csv_files_withnone}, refresh=update_tag_files),
        # Active in settings
        "tac_active": shared.OptionInfo(True, "Enable Tag Autocompletion"),
        "tac_activeIn.txt2img": shared.OptionInfo(True, "Active in txt2img").needs_restart(),
        "tac_activeIn.img2img": shared.OptionInfo(True, "Active in img2img").needs_restart(),
        "tac_activeIn.negativePrompts": shared.OptionInfo(True, "Active in negative prompts").needs_restart(),
        "tac_activeIn.thirdParty": shared.OptionInfo(True, "Active in third party textboxes").info("See <a href=\"https://github.com/DominikDoom/a1111-sd-webui-tagcomplete#-features\" target=\"_blank\">README</a> for supported extensions").needs_restart(),
        "tac_activeIn.modelList": shared.OptionInfo("", "Black/Whitelist models").info("Model names [with file extension] or their hashes, separated by commas"),
        "tac_activeIn.modelListMode": shared.OptionInfo("Blacklist", "Mode to use for model list", gr.Dropdown, lambda: {"choices": ["Blacklist","Whitelist"]}),
        # Results related settings
        "tac_slidingPopup": shared.OptionInfo(True, "Move completion popup together with text cursor"),
        "tac_maxResults": shared.OptionInfo(5, "Maximum results"),
        "tac_showAllResults": shared.OptionInfo(False, "Show all results"),
        "tac_resultStepLength": shared.OptionInfo(100, "How many results to load at once"),
        "tac_delayTime": shared.OptionInfo(100, "Time in ms to wait before triggering completion again").needs_restart(),
        "tac_useWildcards": shared.OptionInfo(True, "Search for wildcards"),
        "tac_sortWildcardResults": shared.OptionInfo(True, "Sort wildcard file contents alphabetically").info("If your wildcard files have a specific custom order, disable this to keep it"),
        "tac_wildcardExclusionList": shared.OptionInfo("", "Wildcard folder exclusion list").info("Add folder names that shouldn't be searched for wildcards, separated by comma.").needs_restart(),
        "tac_skipWildcardRefresh": shared.OptionInfo(False, "Don't re-scan for wildcard files when pressing the extra networks refresh button").info("Useful to prevent hanging if you use a very large wildcard collection."),
        "tac_useEmbeddings": shared.OptionInfo(True, "Search for embeddings"),
        "tac_forceRefreshEmbeddings": shared.OptionInfo(False, "Force refresh embeddings when pressing the extra networks refresh button").info("Turn this on if you have issues with new embeddings not registering correctly in TAC. Warning: Seems to cause reloading issues in gradio for some users."),
        "tac_includeEmbeddingsInNormalResults": shared.OptionInfo(False, "Include embeddings in normal tag results").info("The 'JumpTo...' keybinds (End & Home key by default) will select the first non-embedding result of their direction on the first press for quick navigation in longer lists."),
        "tac_useHypernetworks": shared.OptionInfo(True, "Search for hypernetworks"),
        "tac_useLoras": shared.OptionInfo(True, "Search for Loras"),
        "tac_useLycos": shared.OptionInfo(True, "Search for LyCORIS/LoHa"),
        "tac_useLoraPrefixForLycos": shared.OptionInfo(True, "Use the '<lora:' prefix instead of '<lyco:' for models in the LyCORIS folder").info("The lyco prefix is included for backwards compatibility and not used anymore by default. Disable this if you are on an old webui version without built-in lyco support."),
        "tac_showWikiLinks": shared.OptionInfo(False, "Show '?' next to tags, linking to its Danbooru or e621 wiki page").info("Warning: This is an external site and very likely contains NSFW examples!"),
        "tac_showExtraNetworkPreviews": shared.OptionInfo(True, "Show preview thumbnails for extra networks if available"),
        "tac_modelSortOrder": shared.OptionInfo("Name", "Model sort order", gr.Dropdown, lambda: {"choices": list(sort_criteria.keys())}).info("Order for extra network models and wildcards in dropdown"),
        "tac_useStyleVars": shared.OptionInfo(False, "Search for webui style names").info("Suggests style names from the webui dropdown with '$'. Currently requires a secondary extension like <a href=\"https://github.com/SirVeggie/extension-style-vars\" target=\"_blank\">style-vars</a> to actually apply the styles before generating."),
        # Frequency sorting settings
        "tac_frequencySort": shared.OptionInfo(True, "Locally record tag usage and sort frequent tags higher").info("Will also work for extra networks, keeping the specified base order"),
        "tac_frequencyFunction": shared.OptionInfo("Logarithmic (weak)", "Function to use for frequency sorting", gr.Dropdown, lambda: {"choices": list(frequency_sort_functions.keys())}).info("; ".join([f'<b>{key}</b>: {val}' for key, val in frequency_sort_functions.items()])),
        "tac_frequencyMinCount": shared.OptionInfo(3, "Minimum number of uses for a tag to be considered frequent").info("Tags with less uses than this will not be sorted higher, even if the sorting function would normally result in a higher position."),
        "tac_frequencyMaxAge": shared.OptionInfo(30, "Maximum days since last use for a tag to be considered frequent").info("Similar to the above, tags that haven't been used in this many days will not be sorted higher. Set to 0 to disable."),
        "tac_frequencyRecommendCap": shared.OptionInfo(10, "Maximum number of recommended tags").info("Limits the maximum number of recommended tags to not drown out normal results. Set to 0 to disable."),
        "tac_frequencyIncludeAlias": shared.OptionInfo(False, "Frequency sorting matches aliases for frequent tags").info("Tag frequency will be increased for the main tag even if an alias is used for completion. This option can be used to override the default behavior of alias results being ignored for frequency sorting."),
        # Insertion related settings
        "tac_replaceUnderscores": shared.OptionInfo(True, "Replace underscores with spaces on insertion"),
        "tac_undersocreReplacementExclusionList": shared.OptionInfo("0_0,(o)_(o),+_+,+_-,._.,<o>_<o>,<|>_<|>,=_=,>_<,3_3,6_9,>_o,@_@,^_^,o_o,u_u,x_x,|_|,||_||", "Underscore replacement exclusion list").info("Add tags that shouldn't have underscores replaced with spaces, separated by comma."),
        "tac_escapeParentheses": shared.OptionInfo(True, "Escape parentheses on insertion"),
        "tac_appendComma": shared.OptionInfo(True, "Append comma on tag autocompletion"),
        "tac_appendSpace": shared.OptionInfo(True, "Append space on tag autocompletion").info("will append after comma if the above is enabled"),
        "tac_alwaysSpaceAtEnd": shared.OptionInfo(True, "Always append space if inserting at the end of the textbox").info("takes precedence over the regular space setting for that position"),
        "tac_modelKeywordCompletion": shared.OptionInfo("Never", "Try to add known trigger words for LORA/LyCO models", gr.Dropdown, lambda: {"choices": ["Never","Only user list","Always"]}).info("Will use & prefer the native activation keywords settable in the extra networks UI. Other functionality requires the <a href=\"https://github.com/mix1009/model-keyword\" target=\"_blank\">model-keyword</a> extension to be installed, but will work with it disabled.").needs_restart(),
        "tac_modelKeywordLocation": shared.OptionInfo("Start of prompt", "Where to insert the trigger keyword", gr.Dropdown, lambda: {"choices": ["Start of prompt","End of prompt","Before LORA/LyCO"]}).info("Only relevant if the above option is enabled"),
        "tac_wildcardCompletionMode": shared.OptionInfo("To next folder level", "How to complete nested wildcard paths", gr.Dropdown, lambda: {"choices": ["To next folder level","To first difference","Always fully"]}).info("e.g. \"hair/colours/light/...\""),
        # Alias settings
        "tac_alias.searchByAlias": shared.OptionInfo(True, "Search by alias"),
        "tac_alias.onlyShowAlias": shared.OptionInfo(False, "Only show alias"),
        # Translation settings
        "tac_translation.translationFile": shared.OptionInfo("None", "Translation filename", gr.Dropdown, lambda: {"choices": csv_files_withnone}, refresh=update_tag_files),
        "tac_translation.oldFormat": shared.OptionInfo(False, "Translation file uses old 3-column translation format instead of the new 2-column one"),
        "tac_translation.searchByTranslation": shared.OptionInfo(True, "Search by translation"),
        "tac_translation.liveTranslation": shared.OptionInfo(False, "Show live tag translation below prompt ").info("WIP, expect some bugs"),
        # Extra file settings
        "tac_extra.extraFile": shared.OptionInfo("extra-quality-tags.csv", "Extra filename", gr.Dropdown, lambda: {"choices": csv_files_withnone}, refresh=update_tag_files).info("for small sets of custom tags"),
        "tac_extra.addMode": shared.OptionInfo("Insert before", "Mode to add the extra tags to the main tag list", gr.Dropdown, lambda: {"choices": ["Insert before","Insert after"]}),
        # Chant settings
        "tac_chantFile": shared.OptionInfo("demo-chants.json", "Chant filename", gr.Dropdown, lambda: {"choices": json_files_withnone}, refresh=update_json_files).info("Chants are longer prompt presets"),
    }

    # Add normal settings
    for key, opt in tac_options.items():
        opt.section = TAC_SECTION
        shared.opts.add_option(key, opt)

    # Settings that need special treatment
    # Custom mappings
    keymapDefault = """\
{
    "MoveUp": "ArrowUp",
    "MoveDown": "ArrowDown",
    "JumpUp": "PageUp",
    "JumpDown": "PageDown",
    "JumpToStart": "Home",
    "JumpToEnd": "End",
    "ChooseSelected": "Enter",
    "ChooseFirstOrSelected": "Tab",
    "Close": "Escape"
}\
"""
    colorDefault = """\
{
    "danbooru": {
        "-1": ["red", "maroon"],
        "0": ["lightblue", "dodgerblue"],
        "1": ["indianred", "firebrick"],
        "3": ["violet", "darkorchid"],
        "4": ["lightgreen", "darkgreen"],
        "5": ["orange", "darkorange"]
    },
    "e621": {
        "-1": ["red", "maroon"],
        "0": ["lightblue", "dodgerblue"],
        "1": ["gold", "goldenrod"],
        "3": ["violet", "darkorchid"],
        "4": ["lightgreen", "darkgreen"],
        "5": ["tomato", "darksalmon"],
        "6": ["red", "maroon"],
        "7": ["whitesmoke", "black"],
        "8": ["seagreen", "darkseagreen"]
    },
    "derpibooru": {
        "-1": ["red", "maroon"],
        "0": ["#60d160", "#3d9d3d"],
        "1": ["#fff956", "#918e2e"],
        "3": ["#fd9961", "#a14c2e"],
        "4": ["#cf5bbe", "#6c1e6c"],
        "5": ["#3c8ad9", "#1e5e93"],
        "6": ["#a6a6a6", "#555555"],
        "7": ["#47abc1", "#1f6c7c"],
        "8": ["#7871d0", "#392f7d"],
        "9": ["#df3647", "#8e1c2b"],
        "10": ["#c98f2b", "#7b470e"],
        "11": ["#e87ebe", "#a83583"]
    },
    "danbooru_e621_merged": {
        "-1": ["red", "maroon"],
        "0": ["lightblue", "dodgerblue"],
        "1": ["indianred", "firebrick"],
        "3": ["violet", "darkorchid"],
        "4": ["lightgreen", "darkgreen"],
        "5": ["orange", "darkorange"],
        "6": ["red", "maroon"],
        "7": ["lightblue", "dodgerblue"],
        "8": ["gold", "goldenrod"],
        "9": ["gold", "goldenrod"],
        "10": ["violet", "darkorchid"],
        "11": ["lightgreen", "darkgreen"],
        "12": ["tomato", "darksalmon"],
        "14": ["whitesmoke", "black"],
        "15": ["seagreen", "darkseagreen"]
    }
}\
"""
    keymapLabel = "Configure Hotkeys. For possible values, see https://www.w3.org/TR/uievents-key, or leave empty / set to 'None' to disable. Must be valid JSON."
    colorLabel = "Configure colors. See the Settings section in the README for more info. Must be valid JSON."

    try:
        shared.opts.add_option("tac_keymap", shared.OptionInfo(keymapDefault, keymapLabel, gr.Code, lambda: {"language": "json", "interactive": True}, section=TAC_SECTION))
        shared.opts.add_option("tac_colormap", shared.OptionInfo(colorDefault, colorLabel, gr.Code, lambda: {"language": "json", "interactive": True}, section=TAC_SECTION))
    except AttributeError:
        shared.opts.add_option("tac_keymap", shared.OptionInfo(keymapDefault, keymapLabel, gr.Textbox, section=TAC_SECTION))
        shared.opts.add_option("tac_colormap", shared.OptionInfo(colorDefault, colorLabel, gr.Textbox, section=TAC_SECTION))

    #shared.opts.add_option("tac_refreshTempFiles", shared.OptionInfo("Refresh TAC temp files", "Refresh internal temp files", gr.HTML, {}, refresh=refresh_temp_files, section=TAC_SECTION))

script_callbacks.on_ui_settings(on_ui_settings)









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