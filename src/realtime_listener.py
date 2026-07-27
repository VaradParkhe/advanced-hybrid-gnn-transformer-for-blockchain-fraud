import asyncio
import json
import time
import websockets
from inference_realtime import predict_edge

ALCHEMY_WS = "wss://eth-mainnet.g.alchemy.com/v2/_6aYHt7sHCky26Kumyv6p"

# =========================
# FILE OUTPUT (IMPORTANT)
# =========================
OUTPUT_FILE = "live_stream.jsonl"

def write_stream(data):
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")


# ---------------------------------------------------------
# SAFE RPC CALL
# ---------------------------------------------------------
async def rpc_call(method, params):
    async with websockets.connect(ALCHEMY_WS) as ws:
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 99,
            "method": method,
            "params": params
        }))

        while True:
            resp = json.loads(await ws.recv())
            if resp.get("id") == 99:
                return resp.get("result")


# ---------------------------------------------------------
# LISTEN FOR PENDING TX
# ---------------------------------------------------------
async def listen_pending():
    async with websockets.connect(ALCHEMY_WS) as sub_ws:

        await sub_ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_subscribe",
            "params": ["newPendingTransactions"]
        }))

        print("📡 Listening for pending transactions...")

        while True:
            try:
                msg = json.loads(await sub_ws.recv())

                if "params" not in msg:
                    continue

                tx_hash = msg["params"]["result"]

                # Fetch full tx detail are givwn in hash we need to fetch the f
                tx = await rpc_call("eth_getTransactionByHash", [tx_hash])
                if not tx:
                    continue

                # Extract fields
                src = tx["from"].lower()
                dst = tx["to"].lower() if tx["to"] else "0x0"
                value = int(tx["value"], 16) / 1e18
                ts = time.time()

                # Predict
                pred = predict_edge(src, dst, value, ts)

                # =========================
                # CREATE RESULT OBJECT
                # =========================
                result = {
                    "time": time.strftime("%H:%M:%S"),
                    "src": src,
                    "dst": dst,
                    "value": value,
                    "src_pred": pred["src_prediction"]["prediction"],
                    "dst_pred": pred["dst_prediction"]["prediction"],
                    "src_conf": max(pred["src_prediction"]["probabilities"].values()),
                    "dst_conf": max(pred["dst_prediction"]["probabilities"].values())
                }

                # =========================
                # WRITE TO FILE ✅
                # =========================
                write_stream(result)

                # Print for debugging
                print(result)

                await asyncio.sleep(0.15)

            except Exception as e:
                print("Error:", e)
                await asyncio.sleep(1)


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------
def main():
    try:
        asyncio.run(listen_pending())
    except KeyboardInterrupt:
        print("🛑 Stopped.")


if __name__ == "__main__":
    main()