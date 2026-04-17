from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

snapshot_download(
    repo_id="Brianferrell787/financial-news-multisource",
    repo_type="dataset",
    local_dir="./financial-news",
)
