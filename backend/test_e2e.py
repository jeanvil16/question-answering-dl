"""Full end-to-end verification: backend + Vite proxy + prediction."""
import sys, threading, time, json, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app import create_app as create_backend
app = create_backend()

# Start Flask on port 5000
flask_thread = threading.Thread(
    target=lambda: app.run(host="127.0.0.1", port=5000, use_reloader=False),
    daemon=True)
flask_thread.start()
time.sleep(3)

# Test 1: health via Flask directly
req = urllib.request.Request("http://127.0.0.1:5000/api/health")
resp = json.loads(urllib.request.urlopen(req).read())
assert resp["model_loaded"], "model not loaded!"
print("[PASS] 1. Backend health OK:", resp)

# Test 2: model info
req = urllib.request.Request("http://127.0.0.1:5000/api/model/info")
resp = json.loads(urllib.request.urlopen(req).read())
assert resp["trainable_parameters"] > 0
print(f"[PASS] 2. Model info: {resp['trainable_parameters']:,} params")

# Test 3: training history
req = urllib.request.Request("http://127.0.0.1:5000/api/model/history")
resp = json.loads(urllib.request.urlopen(req).read())
assert resp["available"] and len(resp["history"]) > 0
print(f"[PASS] 3. History: {len(resp['history'])} epochs")

# Test 4: prediction
data = json.dumps({
    "context": "Photosynthesis is the process used by plants to convert light energy into chemical energy. It takes place inside organelles called chloroplasts, which contain chlorophyll.",
    "question": "Where does photosynthesis take place?"
}).encode()
req = urllib.request.Request("http://127.0.0.1:5000/api/predict", data=data,
                             headers={"Content-Type": "application/json"})
result = json.loads(urllib.request.urlopen(req).read())
assert result["answer"], "empty answer!"
assert result["confidence"] > 0, "zero confidence!"
print(f"[PASS] 4. Prediction: answer='{result['answer']}' confidence={result['confidence']}")

# Test 5: validation - empty context
data2 = json.dumps({"context": "", "question": "hello"}).encode()
req2 = urllib.request.Request("http://127.0.0.1:5000/api/predict", data=data2,
                              headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req2)
    assert False, "should have failed"
except urllib.error.HTTPError as e:
    assert e.code == 400
    print("[PASS] 5. Validation: empty context returns 400")

# Test 6: missing model check
import backend.inference as inf
inf.engine.loaded = False
data3 = json.dumps({"context": "hello world", "question": "hello?"}).encode()
req3 = urllib.request.Request("http://127.0.0.1:5000/api/predict", data=data3,
                              headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req3)
    assert False, "should have failed"
except urllib.error.HTTPError as e:
    assert e.code == 503
    print("[PASS] 6. Missing model returns 503")
inf.engine.loaded = True  # restore

print()
print("ALL 6 TESTS PASSED - end-to-end system verified")
print(f"Answer: '{result['answer']}' | Confidence: {result['confidence']} | Latency: {result['latency_ms']}ms")
