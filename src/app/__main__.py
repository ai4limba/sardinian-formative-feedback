"""
Starts the writing environment on this computer.

Usage:
    python -m src.app
"""

import argparse

import uvicorn

from src.config import APP_HOST, APP_PORT


def main():
    parser = argparse.ArgumentParser(description="Run the writing environment.")
    parser.add_argument("--port", type=int, default=APP_PORT)
    args = parser.parse_args()

    print(f"Open http://{APP_HOST}:{args.port} in your browser")
    uvicorn.run("src.app.server:app", host=APP_HOST, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
