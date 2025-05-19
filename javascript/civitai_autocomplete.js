(function() {
    // CSSを動的に追加
    const style = document.createElement('style');
    style.innerHTML = `
    .civitai-autocomplete {
        border: 1px solid #aaa;
        background: #fff;
        border-radius: 4px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.15);
        font-size: 1em;
        max-height: 200px;
        overflow-y: auto;
        position: absolute;
        z-index: 9999;
        min-width: 200px;
    }
    .civitai-ac-item {
        padding: 4px 8px;
        cursor: pointer;
    }
    .civitai-ac-item.selected,
    .civitai-ac-item:hover {
        background: #def;
    }
    `;
    document.head.appendChild(style);

    function getTextbox() {
        // 2タブなら2番目のtextareaまで出るので、適宜選択
        // 例: civitaiタブのtextarea
        let tabs = document.querySelectorAll('.tabitem');
        for (let tab of tabs) {
            if (tab.style.display !== "none" && tab.innerText.includes("civitai")) {
                return tab.querySelector('textarea');
            }
        }
        // fallback
        return document.querySelector('textarea');
    }

    function createAutocompleteDiv() {
        let acDiv = document.createElement("div");
        acDiv.className = "civitai-autocomplete";
        acDiv.style.display = "none";
        document.body.appendChild(acDiv);
        return acDiv;
    }

    window.addEventListener("DOMContentLoaded", () => {
        const textbox = getTextbox();
        if (!textbox) return;

        let acDiv = createAutocompleteDiv();
        let candidates = [];
        let selectedIdx = -1;

        async function fetchCandidates(val) {
            // タブ名でAPI切替（ここではcivitaiのみ例示）
            const res = await fetch(`/civitai_candidates?q=${encodeURIComponent(val)}`);
            return await res.json();
        }

        function showCandidates(list) {
            if (!list || !list.length) {
                acDiv.style.display = "none";
                return;
            }
            const rect = textbox.getBoundingClientRect();
            acDiv.style.left = `${rect.left + window.scrollX}px`;
            acDiv.style.top = `${rect.bottom + window.scrollY}px`;
            acDiv.style.minWidth = `${rect.width}px`;
            acDiv.innerHTML = list.map((c, i) =>
                `<div class="civitai-ac-item${i===selectedIdx?' selected':''}" data-idx="${i}">${c}</div>`
            ).join("");
            acDiv.style.display = "block";
        }

        textbox.addEventListener("input", async () => {
            const val = textbox.value;
            if (!val) {
                acDiv.style.display = "none";
                return;
            }
            candidates = await fetchCandidates(val);
            selectedIdx = 0;
            showCandidates(candidates);
        });

        textbox.addEventListener("keydown", e => {
            if (acDiv.style.display === "none") return;
            if (e.key === "ArrowDown") {
                selectedIdx = (selectedIdx + 1) % candidates.length;
                showCandidates(candidates);
                e.preventDefault();
            } else if (e.key === "ArrowUp") {
                selectedIdx = (selectedIdx - 1 + candidates.length) % candidates.length;
                showCandidates(candidates);
                e.preventDefault();
            } else if (e.key === "Enter" && selectedIdx >= 0) {
                textbox.value = candidates[selectedIdx];
                acDiv.style.display = "none";
                e.preventDefault();
            } else if (e.key === "Escape") {
                acDiv.style.display = "none";
            }
        });

        acDiv.addEventListener("mousedown", e => {
            const idx = e.target.getAttribute("data-idx");
            if (idx !== null) {
                textbox.value = candidates[idx];
                acDiv.style.display = "none";
            }
        });
    });
})();