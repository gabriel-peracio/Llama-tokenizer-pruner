import argparse
import safetensors.torch as st
import torch
import re
import zlib


parser = argparse.ArgumentParser(
    description="Load a .safetensors file and print the shapes of tensors inside."
)
parser.add_argument("file_path", type=str, help="Path to the .safetensors file")
args = parser.parse_args()

# Populating vars
file_path = args.file_path


def numerical_sort_key(name):
    """
    This function takes a tensor name and returns a tuple that can be used as a sorting key.
    The tuple contains mixed types of integers and strings, allowing correct numerical sorting of parts.
    """
    components = re.split(r"(\d+)", name)  # Split the name by digit groups
    return [int(text) if text.isdigit() else text for text in components]


def get_short_hash(tensor):
    """
    Compute a short SHA-1 hash of the tensor data.
    """
    # Convert tensor to bytes and hash it
    return (
        zlib.crc32(tensor.to(torch.float32).numpy().tobytes()) & 0xFFFFFFFF
    )  # Ensure it's non-negative


# Open the file in binary mode, read the content, and then load it
with open(file_path, "rb") as f:
    file_content = f.read()  # Read the entire content of the file into memory as bytes

# Load the .safetensors data from the bytes
data = st.load(file_content)

sorted_items = sorted(data.items(), key=lambda item: numerical_sort_key(item[0]))

for name, tensor in sorted_items:
    hash_value = get_short_hash(tensor)
    print(
        f"{name}: {tensor.shape} ({hash_value:08x})"
    )  # Print the name and shape of each tensor
