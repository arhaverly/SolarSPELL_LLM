# SolarSPELL LLM

The SolarSPELL LLM project provides a lightweight local large language model server designed for integration with SolarSPELL devices.

This repository contains scripts and documentation for:

* Installing and building `llama.cpp`
* Downloading and configuring the local LLM
* Running the LLM server
* Hosting an OpenAI-compatible API
* SolarSPELL LLM design requirements
* Example prompts and use cases
* Testing and deployment

The goal is to provide a fully offline LLM that can run directly on SolarSPELL hardware without requiring an Internet connection after installation.

## Installation

Install Git:

```bash
sudo apt-get update
sudo apt-get install -y git
```

Clone this repository:

```bash
git clone https://github.com/arhaverly/SolarSPELL_LLM
cd SolarSPELL_LLM
```

Make the installation script executable:

```bash
chmod +x install-llm.sh
```

Run the installer:

```bash
./install-llm.sh
```

The installation script installs the required dependencies, builds `llama.cpp`, downloads the configured GGUF model, and prepares the system for running the LLM server.

Save the API key that is output from this installation.

## Run the Server

Make the server script executable if necessary:

```bash
chmod +x server.sh
```

Start the LLM server:

```bash
./server.sh
```

By default, the server is expected to listen on:

```text
http://127.0.0.1:8080
```

From the same device, it can be accessed using:

```text
http://localhost:8080
```

## API

`llama.cpp` provides an OpenAI-compatible HTTP API.

For example:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "Explain photosynthesis."
      }
    ]
  }'
```

If API-key authentication is enabled, include:

```bash
-H "Authorization: Bearer <api-key>"
```

## Project Structure

A typical repository layout is:

```text
solarspell-llm/
├── README.md
├── install-llm.sh
├── server.sh
├── requirements/
│   └── design-requirements.md
├── prompts/
│   └── example-prompts.md
└── docs/
```

The model files and compiled `llama.cpp` repository should generally not be committed to Git.

For example, they may be installed under:

```text
~/SolarSPELL_LLM/
├── llama.cpp/
└── models/
```

## Design Goals

The SolarSPELL LLM deployment is designed around the following requirements:

* Fully offline operation after installation
* Low memory and CPU requirements
* Compatibility with ARM and x86 Linux systems where possible
* No dependency on cloud-hosted inference
* Simple installation and startup
* Local HTTP API for integration with SolarSPELL applications
* Small quantized GGUF models suitable for constrained hardware
* Reliable operation in environments with limited connectivity

## Current Model

The default configuration currently uses:

```text
Qwen3-0.6B-Q4_0.gguf
```

The model can be replaced with another GGUF-compatible model by modifying the installation and server configuration.

## Server Configuration

Typical server parameters include:

```text
Context size: 1024
CPU threads: 4
Host: 0.0.0.0
Port: 8080
Reasoning budget: 0
```

These values can be adjusted depending on the available SolarSPELL hardware.

## Development

To rebuild `llama.cpp` manually:

```bash
cd ~/SolarSPELL_LLM/llama.cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=ON \
  -DLLAMA_USE_PREBUILT_UI=ON \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build --target llama-server -j4
```

## Example Prompts

Example prompts for evaluating the model can be stored in:

```text
prompts/example-prompts.md
```

Examples may include:

* Educational question answering
* Summarization
* Science explanations
* Reading comprehension
* Local knowledge-base interaction
* Tutor-style conversations
* Offline educational assistance

## License

Add the appropriate project license here.

If third-party models or software are distributed with the project, review and preserve their respective licenses and attribution requirements.
