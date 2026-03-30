import requests, time

def test_it():
    for _ in range(20):
        try:
            requests.get("http://localhost:8000/health")
            break
        except:
            time.sleep(1)
    
    # 1. Reset
    print("[*] Calling reset...")
    try:
        r = requests.post("http://localhost:8000/reset")
        print("Reset:", r.status_code, r.text)
    except Exception as e:
        print("Reset error:", e)

    # 2. Reindex
    print("[*] Calling reindex...")
    try:
        r2 = requests.post("http://localhost:8000/reindex")
        print("Reindex:", r2.status_code, r2.text)
    except Exception as e:
        print("Reindex error:", e)

if __name__ == "__main__":
    test_it()
