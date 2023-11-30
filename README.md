# GPTQ API Server

LLM API server for AutoGPTQ model. This server is designed to be compatible with the OpenAI API, allowing you to seamlessly use OpenAI clients with it.

## Installation

Before you can run the server, you need to install the necessary package. You can do this easily with `pip`:

```bash
pip install gptq-api
```

## Usage

To run the GPTQAPI Server, use the following command:

```bash
python -m gptqapi.server [model-name] [port]
```

The `model-name` argument is mandatory while the `port` argument is optional, if not provided, it will use 8000 as the default.

You can also configure the server using a `.env` file for convenience. Here's an example:

```dotenv
# .env file
MODEL_NAME=robinsyihab/Sidrap-7B-v2-GPTQ-4bit
PORT=8000
WORKERS=1
SYSTEM_PROMPT=
```

This `.env` file sets default values for the model name, the port the server will listen on, the number of worker processes, and the system prompt which can be used to customize behavior.

## API Schema

This server follows the OpenAI API schema, allowing for seamless integration with OpenAPI client libraries. You can utilize all typical endpoints as if you were using the actual OpenAI API, making it easier to integrate into your existing infrastructure if you're familiar with the OpenAI platform.

## Environment Variables

Here is a list of environment variables you can use to configure the server:

- `MODEL_NAME`: (required) Identifies which AutoGPTQ model to use.
- `PORT`: (optional) Specifies the port number on which to run the API server.
- `WORKERS`: (optional) Defines the number of worker processes for handling requests.
- `SYSTEM_PROMPT`: (optional) Sets the system prompt for the model if needed.

[] Robin Syihab ([@anvie](https://x.com/anvie))
