import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_file(path):
    full_path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(full_path):
        return "File not found"
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def search_code(query):
    results = []
    for root, _, files in os.walk(PROJECT_ROOT):
        if "__pycache__" in root or ".venv" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        if query.lower() in f.read().lower():
                            results.append(path.replace(PROJECT_ROOT, ""))
                except:
                    pass
    return results

def get_project_structure():
    structure = []
    for root, _, files in os.walk(PROJECT_ROOT):
        level = root.replace(PROJECT_ROOT, "").count(os.sep)
        indent = " " * 2 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            if f.endswith(".py"):
                structure.append(f"{indent}  {f}")
    return structure

# ---- JSON-RPC handler ----

def handle_request(method, params):
    if method == "get_file":
        return get_file(params.get("path", ""))

    elif method == "search_code":
        return search_code(params.get("query", ""))

    elif method == "get_project_structure":
        return get_project_structure()

    elif method == "initialize":
        # REQUIRED for MCP handshake
        return {"capabilities": {}}

    return None


def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            request = json.loads(line)

            method = request.get("method")
            params = request.get("params", {})
            req_id = request.get("id")

            result = handle_request(method, params)

            response = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result
            }

            print(json.dumps(response), flush=True)

        except Exception as e:
            error = {
                "jsonrpc": "2.0",
                "id": None,
                "error": str(e)
            }
            print(json.dumps(error), flush=True)


if __name__ == "__main__":
    main()