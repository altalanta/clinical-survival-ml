FROM condaforge/mambaforge:23.11.0-0

WORKDIR /workspace

COPY env/environment.yml env/environment.yml
RUN mamba env create -f env/environment.yml && mamba clean --all --yes

COPY . .
RUN /opt/conda/envs/clinical-survival-ml/bin/pip install -e .[dev]

ENV PATH="/opt/conda/envs/clinical-survival-ml/bin:$PATH"
ENTRYPOINT ["clinical-ml"]
CMD ["--help"]
