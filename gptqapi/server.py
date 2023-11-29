
import os
import sys
import uvicorn

from gptqapi import create_app
from gptqapi.chat import ChatModel

import dotenv

dotenv.load_dotenv()

if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    port = os.getenv("PORT", '8000')
    workers = os.getenv("WORKERS", '1')

    if not model_name:
        raise ValueError("MODEL_NAME environment variable is not set")

    args = sys.argv[1:]

    if '--help' in args or '-h' in args:
        print("Usage: python gptq_api_server.py [model_name] [port]")
        sys.exit(0)

    if len(args) > 0:
        model_name = args[0]
    if len(args) > 1:
        try:
            port = int(args[1])
        except ValueError:
            print(f"Invalid port: {args[1]}\n", file=sys.stderr)
            sys.exit(1)

    print("GPTQ API Server")
    print(f"Starting server with model `{model_name}` on port {port} with {workers} workers")

    chat_model = ChatModel(model_name, {
        "system_prompt": os.getenv("SYSTEM_PROMPT")
    })
    app = create_app(chat_model)
    uvicorn.run(app, host="0.0.0.0", port=int(port), workers=int(workers))
