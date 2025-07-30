from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="/home/khj6051/alignment-v3/audiobox/checkpoints/2025-04-15_06-28-08/0005000-0.6704.ckpt",
    path_in_repo="dynamic_v3_0414.ckpt",
    repo_id="Daniel777/textalignment",
    repo_type="model"
)