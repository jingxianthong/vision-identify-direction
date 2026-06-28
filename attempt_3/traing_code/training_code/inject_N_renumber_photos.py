import os
import shutil

# -------------------------------------------------------
# renumber_photos.py
#
# Merges new photos from  newly_added_dataset/
# into  dataset/border/  with correct continuing numbers.
#
# Example:
#   dataset/border already has border _0 ... border _61
#   newly_added_dataset has border _0 ... border _20
#
#   Result: new photos become border _62 ... border _82
#           and are moved into dataset/border/
#
# Run:  python renumber_photos.py
# -------------------------------------------------------

script_dir     = os.path.dirname(os.path.abspath(__file__))
BORDER_FOLDER  = os.path.join(script_dir, "dataset", "border")
NEW_FOLDER     = os.path.join(script_dir, "dataset","newly_added_dataset")
FILE_PREFIX    = "border _"

EXTS = ('.jpg', '.jpeg', '.png')

# -------------------------------------------------------

def get_number(filename, prefix):
    """Return the integer suffix after prefix, or None."""
    stem = os.path.splitext(filename)[0]
    if stem.startswith(prefix):
        suffix = stem[len(prefix):]
        if suffix.isdigit():
            return int(suffix)
    return None

def scan_highest(folder, prefix):
    """Return the highest existing number in a folder, or -1 if none."""
    numbers = []
    for f in os.listdir(folder):
        if not f.lower().endswith(EXTS):
            continue
        n = get_number(f, prefix)
        if n is not None:
            numbers.append(n)
    return max(numbers) if numbers else -1

def main():
    # --- Check folders exist ---
    if not os.path.exists(BORDER_FOLDER):
        print(f"ERROR: Main folder not found:\n  {BORDER_FOLDER}")
        return
    if not os.path.exists(NEW_FOLDER):
        print(f"ERROR: New dataset folder not found:\n  {NEW_FOLDER}")
        print("Please create the folder 'newly_added_dataset' next to this script")
        print("and put your new photos inside it.")
        return

    # --- Find what number to continue from ---
    highest = scan_highest(BORDER_FOLDER, FILE_PREFIX)
    next_num = highest + 1
    print(f"dataset/border   highest number : {highest if highest >= 0 else 'empty'}")
    print(f"New photos will start at        : {FILE_PREFIX}{next_num}.jpg")
    print()

    # --- Collect new photos, sorted by their original number ---
    new_files = [f for f in os.listdir(NEW_FOLDER) if f.lower().endswith(EXTS)]
    if not new_files:
        print("No photos found in newly_added_dataset/  — nothing to do.")
        return

    def sort_key(f):
        n = get_number(f, FILE_PREFIX)
        return (0, n) if n is not None else (1, f)

    new_files_sorted = sorted(new_files, key=sort_key)
    print(f"Found {len(new_files_sorted)} new photo(s) to inject:\n")

    # --- Move & rename into dataset/border ---
    for i, fname in enumerate(new_files_sorted):
        ext      = os.path.splitext(fname)[1].lower()
        new_name = f"{FILE_PREFIX}{next_num + i}{ext}"
        src      = os.path.join(NEW_FOLDER, fname)
        dst      = os.path.join(BORDER_FOLDER, new_name)
        shutil.move(src, dst)
        print(f"  newly_added_dataset/{fname:<20}  ->  dataset/border/{new_name}")

    print(f"\nDone!  {len(new_files_sorted)} photos injected.")
    print(f"dataset/border now has numbers 0  to  {next_num + len(new_files_sorted) - 1}.")
    print("\nnewly_added_dataset/ is now empty — ready for your next batch!")

if __name__ == "__main__":
    main()
