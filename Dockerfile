FROM python:3.12-slim as build_stage
WORKDIR /work

RUN pip install jinja2~=3.0.3
COPY model_config_template.pbtxt .
COPY create_triton_model_repository.py .

COPY models /work/models
RUN python3 create_triton_model_repository.py --use-autoconfig

FROM nvcr.io/nvidia/tritonserver:23.12-py3

COPY --from=build_stage /work/model_repository /model_repository

CMD ["tritonserver", "--model-repository=/model_repository"]