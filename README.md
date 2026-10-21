# AyurBot: Explainable Ayurveda Knowledge Assistant

AyurBot is an educational chatbot that answers questions about Ayurvedic herbs, doshas, classical condition terms, traditional properties and concepts. Every answer is built only from a local, editable JSON knowledge base, and the interface shows exactly how it was built: which words were recognised (rule-based NER), how each was detected, which knowledge-base entries were used, and their retrieval scores.

> **Educational prototype.** AyurBot describes traditional Ayurvedic knowledge, which is not the same as clinical evidence. It has not been clinically validated. It does not diagnose, treat or recommend doses. Consult a qualified healthcare professional about any health concern.

## What it does

AyurBot combines four pieces, each in its own module under `src/`. The **knowledge base** (`knowledge_base.py`) loads and validates `data/herbs.json` once at startup and reports every problem in the file at once. The **NER engine** (`ner_engine.py`) is a transparent rule-based recogniser: a dictionary built from the knowledge base plus regular expressions for dosage mentions, covering seven categories (`HERB`, `BOTANICAL_NAME`, `DOSHA`, `CONDITION_TERM`, `TRADITIONAL_PROPERTY`, `DOSAGE_MENTION`, `AYURVEDIC_CONCEPT`). The **retrieval engine** (`retrieval_engine.py`) runs semantic search with Sentence Transformers (`all-MiniLM-L6-v2`), with a pure-Python BM25 lexical fallback that works without PyTorch. The **response engine** (`response_engine.py`) builds answers from fixed templates and knowledge-base text only, and adds safety notices when a question is personal, mentions a dose, or describes an emergency.

No generative language model is used, no API key is needed, and nothing is sent to the cloud. The only network access is a one-time download of the embedding model (about 90 MB) on first launch.

## Quick start on Windows (VS Code)

You need **Python 3.11** (from python.org, with the "py launcher" option ticked) and **VS Code** with the Microsoft Python extension. No administrator rights are needed.

1. Extract the ZIP, for example to `C:\Users\<you>\Projects\AyurBot`.
2. In VS Code choose **File → Open Folder…** and select the `AyurBot` folder (the one containing `app.py`).
3. Open the integrated terminal with **Terminal → New Terminal** (or Ctrl+`). It opens in the project folder.
4. Run these commands one at a time:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

5. When the terminal shows `AyurBot is ready:  http://127.0.0.1:7860`, open that address in your browser. Press **Ctrl+C** in the terminal to stop the app.

These commands call the virtual environment's Python directly, so you never need to "activate" the environment and PowerShell's script-execution policy does not get in the way. If VS Code asks whether to use the new `.venv` interpreter, choose **Yes** (the included `.vscode/settings.json` already points at it).

**Shortcut:** double-click `setup_windows.bat` once, then `run_windows.bat` to start the app. In VS Code you can also press **F5** and pick "Run AyurBot" or "Run AyurBot (lexical fallback only)".

### First launch and the model cache

On the first launch, AyurBot downloads `all-MiniLM-L6-v2` from Hugging Face (about 90 MB) into your user cache (`C:\Users\<you>\.cache\huggingface`). Later launches load it from that cache without downloading it again: AyurBot always tries the local cache first and only goes online if the model is missing. Startup with a cached model normally takes 5 to 20 seconds on a laptop CPU, most of it importing PyTorch.

If the download fails (no internet, firewall, proxy), AyurBot does not crash. It logs the reason, switches to **lexical fallback mode**, and the **System status** tab in the UI says so. It does not retry the failed model on each question.

## Using AyurBot

Type a question and press **Enter** (Shift+Enter adds a new line), or click an example. Good questions name herbs (Tulsi, Haridra, Ashwagandha, Brahmi, Guduchi, Neem, or synonyms such as Giloy, Holy Basil, Turmeric, or botanical names such as *Withania somnifera*), doshas, classical terms (Kasa, Jwara, Kushtha), English glosses (cough, fever, skin disease), properties (Medhya, Rasayana, Ushna) or concepts (Agni, Prakriti, Ama). Devanagari (तुलसी, वात) and diacritics (Tulasī, Vāta) work too.

The right-hand panel explains each answer. **Recognised entities** highlights the words NER found, colour-coded with a legend. The **Entities** tab is the NER audit trail: original text, category, canonical entry, character offsets, detection method (for example "Exact dictionary match (synonym)") and a plain-language explanation. The **Evidence** tab lists the entries used, why each was selected, and its retrieval score. Scores are relevance indicators (how closely an entry matches your wording or meaning), not measures of medical accuracy. The **System status** tab shows the retrieval mode, model state, knowledge-base size and timing.

## Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

All tests run offline. Semantic-search tests use a small fake encoder, so no model download is needed. One extra test loads the real model; enable it with:

```powershell
$env:AYURBOT_RUN_MODEL_TESTS = "1"; .\.venv\Scripts\python.exe -m pytest -m model; Remove-Item Env:AYURBOT_RUN_MODEL_TESTS
```

In VS Code you can also use the **Testing** sidebar (the flask icon); pytest is already configured.

## Configuration

All settings have safe defaults. To change one, copy `.env.example` to `.env` and edit it, or set an environment variable. Invalid values never crash the app: they are reported as warnings and the default is used.

| Variable | Default | Purpose |
|---|---|---|
| `AYURBOT_PORT` | `7860` | Web server port. If busy, the next free port is used. |
| `AYURBOT_HOST` | `127.0.0.1` | Use `0.0.0.0` only if you want other devices on your network to connect. |
| `AYURBOT_KB_PATH` | `data/herbs.json` | Knowledge base file (relative to the project folder). |
| `AYURBOT_DISABLE_TRANSFORMER` | `false` | `true` runs lexical search only (no PyTorch import, fastest startup). |
| `AYURBOT_OFFLINE` | `false` | `true` never goes online; the model must already be cached. |
| `AYURBOT_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Any Sentence Transformers model. |
| `AYURBOT_MODEL_CACHE_DIR` | *(user cache)* | Store the model elsewhere, e.g. `models` inside the project. |
| `AYURBOT_DEVICE` | `cpu` | `auto` uses an NVIDIA GPU if PyTorch can see one. |
| `AYURBOT_TOP_K` | `3` | Maximum entries retrieved per question (1 to 10). |
| `AYURBOT_MIN_SEMANTIC_SCORE` | `0.30` | Cosine-similarity cut-off for semantic results. |
| `AYURBOT_MIN_LEXICAL_SCORE` | `0.25` | Query-coverage cut-off for lexical results. |
| `AYURBOT_MAX_QUERY_CHARS` | `1000` | Longest question accepted. |
| `AYURBOT_MAX_HISTORY_MESSAGES` | `40` | Chat messages kept per session (older ones are dropped). |

To set a variable just for one run in PowerShell: `$env:AYURBOT_PORT = "7900"; .\.venv\Scripts\python.exe app.py`.

## Adding to the knowledge base

Edit `data/herbs.json` in VS Code (it validates JSON syntax as you type) and restart the app. No Python changes are needed. The new entry is automatically recognised by NER, indexed for search and used in answers. Copy an existing herb as a template:

```json
{
  "id": "amalaki",
  "type": "herb",
  "canonical_name": "Amalaki",
  "sanskrit_names": ["Amalaki", "Dhatri"],
  "devanagari": ["आमलकी"],
  "english_name": "Indian Gooseberry",
  "synonyms": ["Amla", "Indian Gooseberry"],
  "botanical_name": "Phyllanthus emblica",
  "botanical_synonyms": ["Emblica officinalis"],
  "summary": "One sentence describing the herb in traditional terms.",
  "traditional_uses": [
    {"term_id": "jwara", "description": "Described in the context of Jwara."}
  ],
  "traditional_properties": {"rasa": ["Amla"], "virya": "Sheeta", "vipaka": "Madhura", "karma": ["Rasayana"]},
  "dosha_associations": {"pacifies": ["pitta"], "may_aggravate": [], "note": ""},
  "safety_notes": ["Only add safety notes you can support from a reliable source."],
  "evidence_note": "Traditional Ayurvedic description. This is not evidence of clinical efficacy.",
  "data_status": "demonstration",
  "source": {"type": "demonstration_summary", "description": "Where this came from.", "reference": null}
}
```

The rules are: `id` must be unique and use lower-case letters, digits or underscores; `type` is one of `herb`, `dosha`, `condition_term`, `traditional_property` or `concept`; `canonical_name` and `summary` are required. `term_id` values must match the `id` of a `condition_term` entry, and dosha lists use the dosha ids `vata`, `pitta` and `kapha`. If something is wrong, the terminal and the System status tab list every problem with its location. Please never invent references: leave `reference` as `null` unless you have checked the source, and keep `data_status` as `demonstration` until the entry has been reviewed.

## Troubleshooting

**`py` is not recognised, or `py -3.11` says Python 3.11 is not installed.** Install Python 3.11 from python.org and tick "py launcher". Run `py -0p` to list installed versions. If you only have the Microsoft Store Python, use its full path instead of `py -3.11`, or install the python.org build.

**VS Code is using the wrong Python.** Press Ctrl+Shift+P, choose **Python: Select Interpreter**, and pick `.\.venv\Scripts\python.exe`. The commands above call that interpreter directly, so they work regardless.

**PowerShell blocks `Activate.ps1`.** You don't need to activate the environment; use `.\.venv\Scripts\python.exe …` as shown. If you prefer activating, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once (no admin needed), or use Command Prompt and `.venv\Scripts\activate.bat`.

**`pip install` fails or is very slow.** Upgrade pip first (step 2). Check that you are on 64-bit Python 3.11 (`.\.venv\Scripts\python.exe -c "import struct; print(struct.calcsize('P')*8)"` should print 64), because PyTorch has no 32-bit Windows wheels. Behind a proxy, add `--proxy http://user:pass@host:port`. If PyTorch still won't install, use the lite install below.

**Lite install without PyTorch.** `.\.venv\Scripts\python.exe -m pip install -r requirements-lite.txt`, then run with `$env:AYURBOT_DISABLE_TRANSFORMER = "1"; .\.venv\Scripts\python.exe app.py`. Everything works except semantic search; answers use BM25 lexical search and NER.

**The model download failed, or you are offline.** The app keeps working in lexical fallback mode, and System status shows the reason. To get semantic search, connect to the internet and restart once so the model is cached; after that you can set `AYURBOT_OFFLINE=true`. On networks that block Hugging Face, copy a downloaded model folder to the machine and set `AYURBOT_MODEL_NAME` to that folder's path.

**`Port 7860 is busy`.** AyurBot automatically tries the next 10 ports and prints the address it used. To choose one yourself, set `AYURBOT_PORT=7900`. To see what is using a port: `netstat -ano | findstr :7860`.

**Knowledge base error at startup.** The app still opens, and System status lists each problem (for example "Invalid JSON at line 120, column 5" or "duplicate id"). Fix the file and restart. A trailing comma before `]` or `}` is the most common JSON mistake.

**Strange characters in the terminal.** Older Windows consoles are not UTF-8. AyurBot never crashes on this, but you can run `chcp 65001` first for nicer output.

## Tested versions

Development and testing used **Python 3.11.15 and 3.12.3**, Gradio 6.30.0, pandas 3.0, NumPy 2.4 and pytest 9. The automated tests, the Gradio interface and the full question-and-answer flow through a running server were verified on Linux, in lexical fallback mode. The real Sentence Transformers model could not be downloaded or run in the build environment (no access to Hugging Face, and the Linux PyTorch build needs several GB of CUDA libraries), so the semantic path was tested with a fake encoder only. Please run the model test above once on your Windows machine. See `docs/architecture.md` for design details, known limitations and the development roadmap.

## Project structure

```text
AyurBot/
├── app.py                    Gradio web interface and startup
├── config.py                 Settings with validated .env / environment overrides
├── requirements.txt          Full install (semantic search)
├── requirements-lite.txt     Install without PyTorch (lexical fallback only)
├── pytest.ini
├── setup_windows.bat         One-click setup (creates .venv, installs packages)
├── run_windows.bat           One-click start
├── .env.example
├── .gitignore
├── .vscode/                  Interpreter, test and run/debug settings
├── data/
│   └── herbs.json            Editable knowledge base (56 demonstration entries)
├── src/
│   ├── __init__.py
│   ├── text_utils.py         Offset-preserving Unicode folding, tokenisation
│   ├── knowledge_base.py     Loading, validation, lexicon, retrieval documents
│   ├── ner_engine.py         Rule-based NER (dictionary + dosage regex)
│   ├── retrieval_engine.py   Semantic retrieval + BM25 fallback
│   ├── response_engine.py    Grounded template answers, safety notices, tables
│   └── service.py            Builds everything once; never crashes on startup
├── tests/                    Offline test suite (pytest)
└── docs/
    └── architecture.md       Design, data flow, limitations, roadmap
```
