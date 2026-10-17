# CQ-editor: build123d + AI Vision

A Python desktop CAD editor for writing, rendering, debugging, and exporting
[build123d](https://github.com/gumyr/build123d) models, with optional AI chat and
visual feedback from the 3D viewport.

This project is a fork of [CadQuery/CQ-editor](https://github.com/CadQuery/CQ-editor).
It uses build123d for modeling and retains the PyQt5 interface and Open CASCADE
(OCCT/OCP) viewer. The bundled examples use build123d's Algebra mode.

## Features

- Python editor with syntax highlighting, completion, and automatic reloading.
- Interactive 3D viewer, object tree, and inspection of faces, edges, and solids.
- Graphical debugger, variable inspection, tracebacks, and an embedded console.
- STL, STEP, and BREP export.
- Eight bundled examples, from a simple box to sweeps and assemblies.
- Optional AI chat that can inspect editor code, generate and validate models,
  render changes, and export the result.
- Reference image attachments and a Vision toggle for screenshot feedback after
  rendering. These require a provider/model that supports image inputs.
- Persistent chat sessions with saved design code, editable model selection,
  reasoning effort controls, and focused AI subagent tasks.
- A model upload dialog with a PCBWay preset and a custom URL option.

## Install from source

Use a Python environment with compatible build123d, OCP, and PyQt5 packages.
The local development environment uses Python 3.12; dependency availability can
vary by Python version and platform. A desktop display and working OpenGL support
are required for the viewer. The code includes Windows, Linux, and macOS support,
but this fork has not been verified on every platform.

Download or clone this repository, open a terminal in its root, and create a
virtual environment.

**Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell does not allow activation, use `.\.venv\Scripts\python.exe` in
place of `python` in the commands below.

**Linux / macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To include AI chat dependencies:

```bash
python -m pip install -e ".[ai]"
```

The normal editor and viewer work without API credentials. AI chat additionally
requires the optional dependencies, provider credentials, and network access.

## Run and create a model

From the repository root:

```bash
python run.py
```

Open a script on startup:

```bash
python run.py examples/01_hello_box.py
```

After installation, `cq-editor` is also available as a command.

Paste the following into the editor and press **F5** to render:

```python
from build123d import *

result = Box(30, 20, 10)
result = fillet(result.edges().filter_by(Axis.Z), radius=2)
show_object(result)
```

`show_object` is supplied by the editor at execution time. A script that uses it
will need its own display helper if run as a standalone Python program. Use the
**Examples** menu for more models and select objects in the object tree to inspect
or export them.

## Configure AI chat

Copy the credential template to a local `.env` file:

```powershell
# Windows PowerShell
Copy-Item .env.example .env
```

```bash
# Linux / macOS
cp .env.example .env
```

Fill in credentials for one provider. All API key fields in the template are
intentionally blank. `.env` and its variants are excluded from Git; keep real
credentials out of source files, screenshots, and chat transcripts.

| Variable | Purpose |
| --- | --- |
| `FIREWORKS_API_KEY` | Fireworks credentials. |
| `FIREWORKS_MODEL_ID` | Fireworks model ID; the template matches the source default. |
| `FIREWORKS_BASE_URL` | Optional Fireworks endpoint override. |
| `ANTHROPIC_API_KEY` | Anthropic credentials or credentials for an Anthropic-compatible gateway. |
| `ANTHROPIC_BASE_URL` | Optional Anthropic-compatible endpoint override. |
| `MODEL_ID` | Model default, used ahead of `FIREWORKS_MODEL_ID`. Set this for direct Anthropic use. |
| `CQ_SUBAGENT_MODEL_ID` | Optional model default for focused subagent tasks. |

For Fireworks, set `FIREWORKS_API_KEY` and leave the Anthropic fields empty. The
client's default endpoint is `https://api.fireworks.ai/inference`, without `/v1`.
For direct Anthropic use, set `ANTHROPIC_API_KEY` and `MODEL_ID`, leaving
`ANTHROPIC_BASE_URL` empty. An Anthropic key or base URL takes priority over
Fireworks configuration. A model selected in the chat UI overrides model defaults.

The client also accepts `OPENAI_API_KEY` together with `OPENAI_API_BASE` as a
fallback for a gateway that implements the **Anthropic Messages API**. These
variable names do not enable a native OpenAI API client.

The app loads `.env` from the repository/package root, then the working directory,
using the first file it finds. Existing process environment variables take
precedence. Restart the application after changing credentials.

In the **AI Chat** dock, choose a model and describe the part you want. Attach
reference images or enable **Vision** when your selected model supports images.
The model picker is editable; its bundled suggestions are not a guarantee of
provider availability or capability.

### Execution and data

AI features send prompts, relevant code, tool results, attached reference images,
and enabled viewport feedback to your configured provider. API usage may incur
provider charges. Generated code is executed locally with your user's access;
the validation subprocess is a crash check, not a security sandbox. Review code
and use scripts and providers you trust.

Chat history and per-session design code are stored outside the repository at
`~/.cq-editor/chat_sessions/` (on Windows,
`%USERPROFILE%\.cq-editor\chat_sessions\`). Exports created by AI tools default
to `exports/` under the working directory. Model uploads send the selected model
or script to the URL chosen in the upload dialog.

## Project layout

```text
cq_editor/
  ai/              AI client, CAD tools, subagent helpers, session storage
  widgets/         Editor, chat, console, debugger, object tree, 3D viewer
  main_window.py   Application layout, menus, and component wiring
  cq_utils.py      build123d/OCP conversion and export helpers
examples/          Eight build123d example scripts
tests/            Geometry and Qt application tests
icons/             Application icons and view assets
pyinstaller/       Packaging assets and runtime hooks
conda/             Conda packaging and installer configuration
run.py             Source launch entry point
pyproject.toml     Dependencies, optional extras, and console commands
pyinstaller.spec   PyInstaller build definition
```

## Development

```bash
python -m pip install -e ".[ai,test,dev]"
python -m pytest --no-xvfb
```

Qt/viewer tests require a display and working OpenGL. Linux CI uses Xvfb and
software rendering; its system package setup is in
[the Linux test workflow](.github/workflows/test_linux.yml).

Check formatting with the command used by CI:

```bash
black --diff --check . --exclude "(icons_res.py|examples/)"
```

PyInstaller builds start from `pyinstaller.spec`. The Conda recipes, constructor
workflow, and Azure pipeline were inherited from upstream and contain upstream
package/publishing settings; review them before using them for your own releases.

## Help and credits

Report issues with this fork in **this repository's** issue tracker, including
your OS, Python version, installation method, traceback, and a minimal example.
Remove credentials and private design data before sharing logs or transcripts.

The original editor was developed by the CadQuery community. Upstream resources:

- [CQ-editor source](https://github.com/CadQuery/CQ-editor)
- [CQ-editor wiki](https://github.com/CadQuery/CQ-editor/wiki) for inherited UI concepts
- [build123d documentation](https://build123d.readthedocs.io/) for modeling APIs

Licensed under the [Apache License 2.0](LICENSE). Existing upstream attribution
is retained.
