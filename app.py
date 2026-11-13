"""Local HTTP chat UI and inference API for Open Ligeng."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

BASE = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "zyss1/qwen25-7b-openligeng"
SYSTEM = "用简短、自然的中文口语回复。直接回答，不加发言者标签。"
ROOT = Path(__file__).resolve().parent


def validate(data):
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object.")
    turns = data.get("messages")
    if not isinstance(turns, list) or not 1 <= len(turns) <= 7 or not len(turns) % 2:
        raise ValueError("Expected alternating user/assistant messages ending with user (1–7 messages).")
    clean = []
    for i, t in enumerate(turns):
        role = "assistant" if i % 2 else "user"
        if not isinstance(t, dict) or t.get("role") != role or not isinstance(t.get("content"), str):
            raise ValueError("Invalid message.")
        content = t["content"].strip()
        if not content:
            raise ValueError("Empty message.")
        clean.append({"role": role, "content": content})
    if sum(len(t["content"]) for t in clean) > 600:
        raise ValueError("The conversation context is limited to 600 characters.")
    return clean


class Model:
    def __init__(self, args):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required.")
        self.torch = torch
        self.lock = threading.Lock()
        self.tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=args.offline)
        base = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.bfloat16, local_files_only=args.offline,
            low_cpu_mem_usage=True, device_map={"": "cuda"},
        )
        self.model = PeftModel.from_pretrained(base, args.adapter, local_files_only=args.offline).eval()

    def reply(self, turns):
        tok = self.tokenizer
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, *turns],
            tokenize=False, add_generation_prompt=True,
        )
        data = tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        with self.lock, self.torch.inference_mode():
            out = self.model.generate(
                **data, max_new_tokens=32, do_sample=True,
                temperature=0.62, top_p=0.9, top_k=30, repetition_penalty=1.16,
            )
        return tok.decode(out[0, data.input_ids.shape[-1]:], skip_special_tokens=True).strip()


def make_handler(model):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, status=200, kind="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def local_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}"

        def do_GET(self):
            if not self.local_host():
                return self.send({"error": "Use the 127.0.0.1 URL."}, 403)
            if self.path == "/":
                return self.send((ROOT / "index.html").read_bytes(), kind="text/html; charset=utf-8")
            if self.path == "/api/health":
                return self.send({"ready": True})
            self.send({"error": "Not found."}, 404)

        def do_POST(self):
            if not self.local_host() or self.headers.get("Origin") not in (None, "http://" + self.headers["Host"]):
                return self.send({"error": "This API is local only."}, 403)
            if self.path != "/api/chat":
                return self.send({"error": "Not found."}, 404)
            if self.headers.get_content_type() != "application/json":
                return self.send({"error": "Send application/json."}, 415)
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("Invalid request size.")
                turns = validate(json.loads(self.rfile.read(size)))
            except ValueError as exc:
                return self.send({"error": str(exc)}, 400)
            try:
                text = model.reply(turns)
            except Exception:
                return self.send({"error": "Generation failed. Check available GPU memory."}, 500)
            if not text:
                return self.send({"error": "The model returned an empty reply. Try again."}, 502)
            self.send({"reply": text})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=BASE, help="Hub ID or local base model directory")
    parser.add_argument("--adapter", default=ADAPTER, help="Hub ID or local adapter directory")
    parser.add_argument("--offline", action="store_true", help="Only load local or already cached model files")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print("Loading BF16 base model and LoRA adapter…", flush=True)
    model = Model(args)
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model)) as server:
        print(f"Open Ligeng: http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
