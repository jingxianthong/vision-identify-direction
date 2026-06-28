import os
import shutil
import glob
import zipfile
import tarfile

# --- Configuration --- #
# Set the name of your uploaded zip/7z file here.
# For example, if you uploaded 'my_dataset.zip', set this to 'my_dataset.zip'
ZIP_FILE_NAME = 'dataset.zip' # <--- IMPORTANT: Update this if your file has a different name!

# --- Data Preparation --- #

# 1. Wipe old folders to start fresh
print("Cleaning up previous data...\n")
shutil.rmtree('/content/dataset', ignore_errors=True)
shutil.rmtree('/content/temp_extract', ignore_errors=True)

# 2. Create strict YOLO folders
IMG_DIR = '/content/dataset/images'
LBL_DIR = '/content/dataset/labels'
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)
print(f"Created image directory: {IMG_DIR} (Exists: {os.path.exists(IMG_DIR)})\n")
print(f"Created label directory: {LBL_DIR} (Exists: {os.path.exists(LBL_DIR)})\n")

# 3. Find and extract the uploaded file
file_path = os.path.join('/content', ZIP_FILE_NAME)

if not os.path.exists(file_path):
    # Fallback to glob if specific file not found, but warn user
    uploaded_files = glob.glob('/content/*.zip') + glob.glob('/content/*.7z')
    if uploaded_files:
        file_path = uploaded_files[0]
        print(f"⚠️ WARNING: '{ZIP_FILE_NAME}' not found. Using first detected compressed file: {file_path}\n")
    else:
        print("❌ ERROR: No compressed file found! Please upload your zip or 7z file to the left sidebar.")
        print("       Make sure to update 'ZIP_FILE_NAME' in this cell if it's not 'dataset.zip'.")
        # Exit or raise an error if no file is found
        raise FileNotFoundError("No dataset file found.")

print(f"📦 Found file: {file_path}\n")

# Extract it
if file_path.endswith('.zip'):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall('/content/temp_extract')
elif file_path.endswith(('.7z', '.tar.gz', '.tgz', '.tar')):
    # Ensure 7z is installed for .7z files
    if file_path.endswith('.7z'):
        if shutil.which('7z') is None:
            print("Installing 7zip...\n")
            os.system('apt-get update && apt-get install -y p7zip-full')
        os.system(f'7z x "{file_path}" -o/content/temp_extract -y')
    elif tarfile.is_tarfile(file_path):
        with tarfile.open(file_path, 'r:*') as tar_ref:
            tar_ref.extractall('/content/temp_extract')
else:
    print(f"❌ ERROR: Unsupported file type: {file_path}. Please use .zip, .7z, or .tar.gz files.")
    raise ValueError("Unsupported file type.")

print(f"Extracted contents to /content/temp_extract. Contents: {os.listdir('/content/temp_extract') if os.path.exists('/content/temp_extract') else 'Not found'}\n")

# 4. Sort the files into strict YOLO folders
img_count, lbl_count = 0, 0
if os.path.exists('/content/temp_extract'):
    for root, dirs, files in os.walk('/content/temp_extract'):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(os.path.join(root, f), os.path.join(IMG_DIR, f))
                img_count += 1
            elif f.lower().endswith('.txt') and f not in ['classes.txt', 'data.yaml']:
                shutil.move(os.path.join(root, f), os.path.join(LBL_DIR, f))
                lbl_count += 1
else:
    print("❌ ERROR: Extraction failed, /content/temp_extract does not exist.")

# 5. The Moment of Truth
print("-" * 30)
print(f"🖼️ Images moved to {IMG_DIR}: {img_count}")
print(f"📝 Labels moved to {LBL_DIR}: {lbl_count}")
print("-" * 30)

# Final verification
if os.path.exists(IMG_DIR) and len(os.listdir(IMG_DIR)) > 0 and \
   os.path.exists(LBL_DIR) and len(os.listdir(LBL_DIR)) > 0:
    print("✅ SUCCESS! Your data is ready. You can run the training block now!")
else:
    print("❌ FAILED: Still missing images or labels in target directories.")
    if not os.path.exists(IMG_DIR) or len(os.listdir(IMG_DIR)) == 0:
        print(f"       - Image directory '{IMG_DIR}' is missing or empty.")
    if not os.path.exists(LBL_DIR) or len(os.listdir(LBL_DIR)) == 0:
        print(f"       - Label directory '{LBL_DIR}' is missing or empty.")
    print("       Please check your uploaded zip file and the 'ZIP_FILE_NAME' variable.")
