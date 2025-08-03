from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="/workspace/vts/audiobox/checkpoints/2025-08-01_09-38-54/0008000-1.0566.ckpt",
    path_in_repo="vts_ltx_second_8000.ckpt",
    repo_id="Daniel777/personals",
    repo_type="model"
)