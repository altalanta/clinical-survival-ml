# Contributing

Thanks for helping improve `clinical-survival-ml`! The project ships with a deterministic pipeline and a collection of companion scripts. Before opening a pull request, please make sure the commands below succeed on your machine:

```bash
make install
pytest -q
bash examples/toy/run_toy.sh
docker build -t clinical-survival-ml .
docker run --rm -p 8000:8000 -v $(pwd)/results:/workspace/results clinical-survival-ml serve --models-dir results/artifacts/models
```

The toy workflow script executes the same steps as the CI end-to-end stage (configuration validation, training, evaluation/report rendering, and API smoke checks). The Docker commands ensure the published image starts the CLI as an unprivileged user.

If you add new CLI features, please update the feature matrix in `README.md` and include an automated test referencing the corresponding command.
