from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="~/datasets/medical_train/medsss_data",
    repo_id="pixas/MedSSS-data",
    repo_type="dataset",
    allow_patterns=['sft_1.json']
)
