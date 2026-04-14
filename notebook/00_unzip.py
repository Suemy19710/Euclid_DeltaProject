import zipfile

zip_path = "../einstein_rings/generated_rings_vae.zip"
destination = "../einstein_rings/generated_rings_vae"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(destination)
    print(f"Files extracted to {destination}")