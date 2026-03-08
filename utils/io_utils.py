# utils/io_utils.py

import os
import json
import shutil


def create_directory(dir_path):
    """
    Creates a directory if it does not exist.
    """
    try:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Directory created: {dir_path}")
    except Exception as e:
        print(f"Error creating directory {dir_path}: {e}")


def delete_directory(dir_path):
    """
    Deletes a directory and all its contents.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Directory deleted: {dir_path}")
    else:
        print(f"Directory does not exist: {dir_path}")


def list_files_in_directory(dir_path, extensions=None):
    """
    Lists all files in a directory with optional filtering by file extensions.
    """
    if not os.path.exists(dir_path):
        print(f"Directory does not exist: {dir_path}")
        return []

    files = os.listdir(dir_path)
    if extensions:
        files = [f for f in files if f.lower().endswith(tuple(extensions))]
    return files


def save_json(data, file_path):
    """
    Saves a dictionary as a JSON file.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"JSON file saved: {file_path}")
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")


def load_json(file_path):
    """
    Loads a JSON file and returns its content as a dictionary.
    """
    if not os.path.exists(file_path):
        print(f"JSON file does not exist: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {file_path}: {e}")
        return None


def copy_file(src_path, dest_path):
    """
    Copies a file from source to destination.
    """
    try:
        shutil.copy2(src_path, dest_path)
        print(f"File copied from {src_path} to {dest_path}")
    except Exception as e:
        print(f"Error copying file: {e}")


def move_file(src_path, dest_path):
    """
    Moves a file from source to destination.
    """
    try:
        shutil.move(src_path, dest_path)
        print(f"File moved from {src_path} to {dest_path}")
    except Exception as e:
        print(f"Error moving file: {e}")


def delete_file(file_path):
    """
    Deletes a file if it exists.
    """
    try:
        os.remove(file_path)
        print(f"File deleted: {file_path}")
    except FileNotFoundError:
        print(f"File does not exist: {file_path}")
    except Exception as e:
        print(f"Error deleting file: {e}")


def check_file_exists(file_path):
    """
    Checks if a file exists.
    """
    return os.path.isfile(file_path)


def check_directory_exists(dir_path):
    """
    Checks if a directory exists.
    """
    return os.path.isdir(dir_path)


def get_file_size(file_path):
    """
    Returns the size of a file in bytes.
    """
    try:
        return os.path.getsize(file_path)
    except OSError as e:
        print(f"Error getting size for file {file_path}: {e}")
        return -1


def read_text_file(file_path):
    """
    Reads the content of a text file.
    """
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Text file does not exist: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading text file {file_path}: {e}")
        return None


def save_text_file(content, file_path):
    """
    Saves content to a text file.
    """
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Text file saved: {file_path}")
    except Exception as e:
        print(f"Error saving text file {file_path}: {e}")


def get_basename(file_path):
    """
    Returns the base name of the file (without directory).
    """
    return os.path.basename(file_path)


def get_filename_without_extension(file_path):
    """
    Returns the filename without its extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def get_file_extension(file_path):
    """
    Returns the file extension.
    """
    return os.path.splitext(file_path)[1].lower()
