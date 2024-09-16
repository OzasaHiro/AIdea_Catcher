from io import BytesIO
import IPython
import json
import os
from PIL import Image
import requests
import time


STABILITY_KEY = os.getenv('STABILITY_KEY')

def send_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "image/*",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    image = params.pop("image", None)
    mask = params.pop("mask", None)
    if image is not None and image != '':
        files["image"] = open(image, 'rb')
    if mask is not None and mask != '':
        files["mask"] = open(mask, 'rb')
    if len(files)==0:
        files["none"] = ''

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    return response

def send_async_generation_request(
    host,
    params,
):
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {STABILITY_KEY}"
    }

    # Encode parameters
    files = {}
    if "image" in params:
        image = params.pop("image")
        files = {"image": open(image, 'rb')}

    # Send request
    print(f"Sending REST request to {host}...")
    response = requests.post(
        host,
        headers=headers,
        files=files,
        data=params
    )
    if not response.ok:
        raise Exception(f"HTTP {response.status_code}: {response.text}")

    # Process async response
    response_dict = json.loads(response.text)
    generation_id = response_dict.get("id", None)
    assert generation_id is not None, "Expected id in response"

    # Loop until result or timeout
    timeout = int(os.getenv("WORKER_TIMEOUT", 500))
    start = time.time()
    status_code = 202
    while status_code == 202:
        response = requests.get(
            f"{host}/result/{generation_id}",
            headers={
                **headers,
                "Accept": "image/*"
            },
        )

        if not response.ok:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        status_code = response.status_code
        time.sleep(10)
        if time.time() - start > timeout:
            raise Exception(f"Timeout after {timeout} seconds")

    return response

image = "sample.jpg"
prompt = '''
A distinguished Asian man in his early 50s, with short black hair and glasses, standing confidently on a TED Talk stage. 
He's wearing a stylish navy blue suit with a subtle AI-pattern tie. 
Behind him is a massive curved screen displaying intricate AI neural networks and holographic representations of industrial machinery being optimized in real-time. 
The man is gesturing towards the screen with a sleek, transparent smart device in hand. 
The audience is a mix of global leaders, tech innovators, and students from diverse backgrounds, all captivated by the presentation. 
The stage is bathed in soft blue lighting, emphasizing the futuristic atmosphere. 
In the background, through floor-to-ceiling windows, you can see a skyline of sustainable skyscrapers with visible green technology. 
The scene exudes a sense of groundbreaking innovation, global influence, and the seamless integration of AI in solving world challenges. 
Photorealistic quality, with attention to lighting and detail that captures the excitement and importance of the moment.
'''

negative_prompt = "" 
control_strength = 0.7
seed = 42 
output_format = "png"

host = f"https://api.stability.ai/v2beta/stable-image/control/structure"

params = {
    "control_strength" : control_strength,
    "image" : image,
    "seed" : seed,
    "output_format": output_format,
    "prompt" : prompt,
    "negative_prompt" : negative_prompt,
}

response = send_generation_request(
    host,
    params
)

'''
# Decode response
output_image = response.content
finish_reason = response.headers.get("finish-reason")
seed = response.headers.get("seed")

# Check for NSFW classification
if finish_reason == 'CONTENT_FILTERED':
    raise Warning("Generation failed NSFW classifier")

# Save and display result
filename, _ = os.path.splitext(os.path.basename(image))
edited = f"edited_{filename}_{seed}.{output_format}"
with open(edited, "wb") as f:
    f.write(output_image)
print(f"Saved image {edited}")

#output.no_vertical_scroll()
#print("Original image:")
#IPython.display.display(Image.open(image))
print("Result image:")
IPython.display.display(Image.open(edited))
'''
#__all__ = ['send_generation_request']

