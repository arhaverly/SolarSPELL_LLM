# SolarSPELL LLM

Offline large language models for SolarSPELL devices (Raspberry Pi and other
low-power Linux hardware). Everything runs locally with
[llama.cpp](https://github.com/ggml-org/llama.cpp); an Internet connection is
only needed during installation.

The repository holds two independent setups:

| Folder | What it is | Model | Uses the SolarSPELL library? |
|---|---|---|---|
| [`NonFineTuned/`](NonFineTuned/) | A plain LLM server with an OpenAI-compatible API and llama.cpp's built-in chat UI | Stock Qwen3 0.6B | No |
| [`FineTuned/`](FineTuned/) | **Ask SolarSPELL**: a chat app that searches the SolarSPELL library and answers from it (retrieval-augmented generation) | Llama 3.2 1B fine-tuned on SolarSPELL Q&A (falls back to stock Llama 3.2 1B Instruct) | Yes, 231k passages from ~1,500 library PDFs |

Each folder has its own README with installation steps. They share one
`llama.cpp` build (`~/SolarSPELL_LLM/llama.cpp`), but each runs its own
server, so on a 4 GB Pi run only one at a time.

## Quick start

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/arhaverly/SolarSPELL_LLM ~/SolarSPELL_LLM

# Either:
cd ~/SolarSPELL_LLM/NonFineTuned && ./install-llm.sh
# Or:
cd ~/SolarSPELL_LLM/FineTuned && ./install.sh    # read FineTuned/README.md first
```

## Requirements

See [`NonFineTuned/requirements.md`](NonFineTuned/requirements.md). The main
goal is helping teachers create lesson plans from the curated SolarSPELL
library.

## Credits

`FineTuned/` is a rewrite of ASU Next Lab's Edge AI / sol-ai project
(`asu-nextlab/nx-sol-ai`) by Dan Munnerley, Hithesh Rai Purushothama,
Hrushikesh Pandit and Ananya Kodali, in partnership with ASU SolarSPELL. Their
library index, dataset-generation approach and fine-tuning recipe are reused
here.

## License

Add the appropriate project license here.

If third-party models or software are distributed with the project, review
and preserve their respective licenses and attribution requirements. Llama 3.2
models are covered by the Llama 3.2 Community License.
