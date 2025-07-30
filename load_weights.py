from huggingface_hub import hf_hub_download, login

api_key = ""
login(api_key)

def load_weights():
    temp_path=hf_hub_download(
        repo_id="optimizerai/audiobox",
        filename="new-stage-2.ckpt",
        local_dir="",
        local_dir_use_symlinks=False
        )

if __name__=="__main__":
    load_weights()