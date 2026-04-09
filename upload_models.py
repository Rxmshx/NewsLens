from huggingface_hub import HfApi

api  = HfApi()
repo = "Rxmshx/NewsLens"

# Create repo
api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
print("✅ Repo created")

# Upload DistilBERT model
print("📤 Uploading DistilBERT model...")
api.upload_folder(
    folder_path  = "results/bert_model",
    repo_id      = repo,
    repo_type    = "model",
    path_in_repo = "bert_model"
)

# Upload label map
print("📤 Uploading label map...")
api.upload_file(
    path_or_fileobj = "results/label_map.json",
    path_in_repo    = "label_map.json",
    repo_id         = repo,
    repo_type       = "model"
)

print(f"✅ All models uploaded → https://huggingface.co/Rxmshx/NewsLens")