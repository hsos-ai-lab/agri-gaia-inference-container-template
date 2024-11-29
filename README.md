# Agri Gaia Inference Container Template

This repository contains templates for the to be used as projects from which inference containers for the Agri-Gaia-Platform are build.

## Building a custom inference container template

You can create a custom inference container template and add it to the platform. To build a custom inference container template there are only two things to bear in mind:

1. There is a single Dockerfile expected named "Dockerfile" in the root directory of the inference container template so that the platform can build a docker image from your project. So the file tree should look like:

```bash
/Dockerfile
/rest-of-the-files
/possible-subdir/*
```


2. During the build process the model that you selected in the UI of the Agri-Gaia-Platform is copied into the build context of the image build. Which means that the model is accessible at build-time by the Dockerfile you provide. In addition to the model the platform provides also a metadata file that contains some information about the model which can be useful for running the model. To copy the model and the metadata file into your container image when the platform builds the image with the specific model your Dockerfile must contain the following COPY command:

```bash
COPY models <path-where-the-models-should-be-read-from-by-your-application>
```

This copies the build-context scoped "models" directory and its contents into your container.


### Model metadata file

In addition to the model file e.g. model.onnx there is also a metadata file provided that contains some information about the model which can be useful for running the model. The file is a json file and named "model.json". The file is structured like this:


```json
{
    "name": string, // e.g. "testmodel"
    "format": ModelFormat, // e.g. "onnx"
    "input_name": string, // e.g. "input0"
    "input_datatype": Datatype, // e.g. "float32"
    "input_shape": list[number], // e.g. [1, 3, 224, 224]
    "input_semantics": ShapeSemantics, // e.g "NCHW"
    "output_name": model.output_name, // e.g. "output0"
    "output_datatype": Datatype, // e.g. "float32"
    "output_shape": list[number], // e.g. [1, 2]
    "output_labels": list[string], // e.g. [good, bad]
}
```

Possible values for ModelFormat: "onnx", "pytorch", "tensorflow", "tensorrt"
Possible values for ShapeSemantics: "HWC", "NHWC", "CHW", "NCHW"
Possible values for Datatype: "float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool", "string"