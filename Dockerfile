FROM condaforge/mambaforge:23.11.0-0

WORKDIR /workspace

# Copy environment file and create environment
COPY env/environment.yml env/environment.yml
RUN mamba env create -f env/environment.yml && mamba clean --all --yes

# Install additional production dependencies
RUN /opt/conda/envs/clinical-survival-ml/bin/pip install fastapi uvicorn[standard]

# Copy source code
COPY . .

# Install the package in development mode
RUN /opt/conda/envs/clinical-survival-ml/bin/pip install -e .[dev]

# Set the PATH
ENV PATH="/opt/conda/envs/clinical-survival-ml/bin:$PATH"

# Expose the default API port
EXPOSE 8000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the default command (can be overridden for different modes)
ENTRYPOINT ["clinical-ml"]

# Default to showing help if no arguments provided
CMD ["--help"]
