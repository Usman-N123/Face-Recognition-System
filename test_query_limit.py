import requests, glob, os

images = glob.glob('images/*.jpeg')
if not images:
    print('No images')
    exit()

filename = os.path.basename(images[0])
print(f"Querying with {filename}...")
with open(images[0], 'rb') as f:
    resp = requests.post(
        "http://localhost:8000/query",
        files={"file": (filename, f, "image/jpeg")}
    )

data = resp.json()
print("Matches found:", data.get("matches_found"))
print("Total images returned:", data.get("total_images"))
if data.get("images"):
    print("Images:", len(data["images"]))
