from typing import Any, Dict, List, Tuple
import jinja2
import json
import argparse
import os
import shutil

TEMPLATE_PATH = "model_config_template.pbtxt"


def get_args():
    parser = argparse.ArgumentParser(
        description="Process the input and output file paths."
    )
    parser.add_argument(
        "--input-dir", type=str, default="models", help="Path to the input dir."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_repository",
        help="Path to the output dir.",
    )

    return parser.parse_args()


def main():
    args = get_args()

    file_pairs = get_model_file_pairs(args.input_dir)
    model_repository_path = args.output_dir

    for metadata_file, model_file in file_pairs:
        with open(metadata_file, "r") as file:
            metadata = json.load(file)

        model_root_dir = f"{model_repository_path}/{metadata['name']}"
        model_file_extension = _model_format_to_modelfile_extension(metadata["format"])
        config_path = f"{model_root_dir}/config.pbtxt"
        model_triton_path = f"{model_root_dir}/1/model.{model_file_extension}"
        os.makedirs(os.path.dirname(model_triton_path), exist_ok=True)

        create_triton_pbtxt_file(metadata, config_path)
        shutil.copy(model_file, model_triton_path)


def get_model_file_pairs(directory_path: str) -> List[Tuple[str, str]]:

    files_in_directory = os.listdir(directory_path)

    json_files = [f for f in files_in_directory if f.endswith(".json")]
    model_files = [f for f in files_in_directory if not f.endswith(".json")]

    json_base_names = [os.path.splitext(f)[0] for f in json_files]
    model_base_names = [os.path.splitext(f)[0] for f in model_files]

    valid_pairs = []
    invalid_pairs = []

    for base_name in json_base_names:
        if base_name in model_base_names:
            json_file = f"{directory_path}/{base_name}.json"
            model_file = f"{directory_path}/{base_name}.{model_files[model_base_names.index(base_name)].split('.')[-1]}"
            valid_pairs.append((json_file, model_file))
        else:
            invalid_pairs.append(base_name)

    # Check for invalid pairs and exit if any
    if invalid_pairs:
        print(f"Error: Invalid pairs found for base names: {', '.join(invalid_pairs)}")
        exit(1)

    return valid_pairs


def create_triton_pbtxt_file(metadata: Dict, output_file: str) -> None:
    template_data = create_template_data(metadata)
    content = render_template(template_data)

    with open(output_file, "w") as file:
        file.write(content)


def render_template(template_data: Dict) -> str:
    with open(TEMPLATE_PATH, "r") as file:
        template = file.read()
    template = jinja2.Template(template)
    return template.render(**template_data)


def create_template_data(metadata: Dict) -> Dict[str, Any]:
    config_template_data = {}
    config_template_data["platform"] = _model_format_to_triton_platform(
        metadata["format"]
    )
    config_template_data["model_name"] = metadata["name"]
    config_template_data["max_batch_size"] = 0

    config_template_data["inputs"] = []
    input_tmpl_data = {
        "name": _get_io_tensorname(metadata, "input"),
        "data_type": _iodtype_to_triton_dtype(metadata["input_datatype"]),
        "dims": metadata["input_shape"],
    }

    input_semantics = metadata["input_semantics"]
    if not input_semantics.startswith("N"):
        input_semantics = "N" + input_semantics
        input_tmpl_data["reshape"] = {"shape": [1, *metadata["input_shape"]]}
    input_tmpl_data["format"] = "FORMAT_" + input_semantics

    config_template_data["inputs"].append(input_tmpl_data)

    config_template_data["outputs"] = [
        {
            "name": _get_io_tensorname(metadata, "output"),
            "data_type": _iodtype_to_triton_dtype(metadata["output_datatype"]),
            "dims": metadata["output_shape"],
        }
    ]
    return config_template_data


def _model_format_to_triton_platform(model_format: str) -> str:
    # https://github.com/triton-inference-server/backend/blob/main/README.md#backends
    mapping = {
        "onnx": "onnxruntime_onnx",
        "pytorch": "pytorch_libtorch",
        "tensorrt": "tensorrt_plan",
        "tensorflow": "tensorflow_graphdef",
    }
    return mapping[model_format]


def _iodtype_to_triton_dtype(dtype: str) -> str:
    # https://github.com/triton-inference-server/common/blob/main/protobuf/model_config.proto
    mapping = {
        "float16": "TYPE_FP16",
        "float32": "TYPE_FP32",
        "float64": "TYPE_FP64",
        "int8": "TYPE_INT8",
        "int16": "TYPE_INT16",
        "int32": "TYPE_INT32",
        "int64": "TYPE_INT64",
        "uint8": "TYPE_UINT8",
        "uint16": "TYPE_UINT16",
        "uint32": "TYPE_UINT32",
        "uint64": "TYPE_UINT64",
        "bool": "TYPE_BOOL",
        "string": "TYPE_STRING",
    }
    return mapping[dtype]


def _get_io_tensorname(metadata: Dict, io_type: str) -> str:
    assert io_type in ("input", "output")

    if f"{io_type}_name" in metadata:
        tensor_name = metadata[f"{io_type}_name"]
    else:
        tensor_name = f"{io_type}0"
        if metadata["format"] == "pytorch":
            tensor_name = f"{io_type}__0"

    return tensor_name


def _model_format_to_modelfile_extension(model_format: str) -> str:
    # https://github.com/triton-inference-server/server/blob/main/docs/model_repository.md#model-files
    mapping = {
        "onnx": "onnx",
        "pytorch": "pt",
        "tensorrt": "plan",
        "tensorflow": "graphdef",
    }
    return mapping[model_format]


if __name__ == "__main__":
    main()
