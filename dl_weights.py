from huggingface_hub import hf_hub_download
import os

repo_id = "RinKana/RGL-AE-AL-186K"
filename = "anime_rgl_ae.safetensors"

print(f"Downloading {filename} from {repo_id}...")
path = hf_hub_download(repo_id=repo_id, filename=filename)
os.symlink(path, "ganyu_ff_conv.safetensors")
print("Done!")
