"""Quick smoke-test for the backend (runs in a single Python process).

Starts the Flask server in a background thread, sends a prediction request,
and prints the results. Useful for CI / verification without needing to
manage long-lived background processes.
"""
import sys, threading, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import create_app
app = create_app()
server = threading.Thread(target=lambda: app.run(host="127.0.0.1", port=5001,
                                                use_reloader=False), daemon=True)
server.start(); time.sleep(3)

import urllib.request
# health
req = urllib.request.Request("http://127.0.0.1:5001/api/health")
resp = urllib.request.urlopen(req)
print("HEALTH:", json.loads(resp.read()))

# predict
data = json.dumps({"context": "Photosynthesis is the process used by plants to convert light energy into chemical energy. It takes place inside organelles called chloroplasts.", "question": "Where does photosynthesis take place?"}).encode()
req = urllib.request.Request("http://127.0.0.1:5001/api/predict", data=data, headers={"Content-Type":"application/json"})
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print("PREDICT:", result)

assert result.get("answer"), "No answer returned!"
assert result.get("confidence", 0) > 0, "Zero confidence!"
print("OK - prediction succeeded:", result["answer"], f"(confidence={result['confidence']})")
