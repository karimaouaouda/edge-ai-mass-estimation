from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi();
api.authenticate()


# set encoding to utf-8
import sys
import os
# 1. Force the active terminal stream to reconfigure its encoding to UTF-8.
# 'backslashreplace' ensures if an emoji/symbol can't print, it prints as text instead of crashing.
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    sys.stderr.reconfigure(encoding='utf-8', errors='backslashreplace')

# 2. Force Python's environment configurations before importing Kaggle
os.environ["PYTHONIOENCODING"] = "utf-8"


print(api.read_config_environment())


code_slug = "aouaoudakarim/yolo-seg"

regex_onnx = '.onnx$'

files_to_install = [
    "runs/.*\.pt$"
]

""" for file in files_to_install:
    # download the kernel output file
    try:
        api.kernels_output_cli(code_slug,
                                file_pattern=file,
                                path="./kaggle_outputs",
                                kernel_opt={"page": "eyJHY3NQYWdlVG9rZW4iOiIzMTE0ODg5Mjcvb3V0cHV0L2RhdGEvcHJvY2Vzc2VkL2ltYWdlcy90cmFpbi9iYXRjaF8xMS8wMDAwODguanBnIn0="},
                                quiet=False)
        print(f"Downloaded: {file}")
    except Exception as e:
        print(f"Error downloading {file}: {e}") """

""" 
next_page_token = None
try:
    while True:
        result = api.kernels_list_files(code_slug, page_token=next_page_token, page_size=200)
        print(f"Listing files for kernel {code_slug} (Page Token: {next_page_token})")
        for file in result.files:
            print(f"- Found file: {file.name}")
        next_page_token = result.next_page_token
        if not next_page_token:
            break
except Exception as e:
    print(f"Error listing files for kernel {code_slug}: {e}")
    
 """

try:
    api.kernels_output(code_slug,
                        file_pattern=regex_onnx,
                        path="./kaggle_outputs",
                        page_token="eyJHY3NQYWdlVG9rZW4iOiIzMTE0ODg5Mjcvb3V0cHV0L3J1bnMveW9sby1zZWcvd2FzdGUtc2VnLXlvbG8yNm4vYXJncy55YW1sIn0="
@mermaid-chart
    print(f"Downloaded files matching pattern: {regex_onnx}")
except Exception as e:
    print(f"Error downloading files matching pattern {regex_onnx}: {e}")