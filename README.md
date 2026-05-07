# NeuroCast

A benchmarking framework for neural decoding from podcast listening data.

## Documentation

**📚 Full documentation available at: https://driveway3662.github.io/neurocast-submission-neurips-2026/

## Quick Start

```bash
# Setup environment and download data
./setup.sh

# Train all tasks using the CNN baseline.
make train-all MODEL=baselines/neural_conv_decoder
```

## Features

- **Flexible model architecture**: Register custom models with simple decorators
- **Multiple tasks**: Word embeddings, classification, or custom prediction targets
- **Configurable training**: YAML-based configs with cross-validation and early stopping
- **Multiple metrics**: ROC-AUC, perplexity, top-k accuracy, and custom metrics
- **Time lag analysis**: Automatically find optimal temporal offsets

## Learn More

- **[Quickstart Guide](https://driveway3662.github.io/neurocast-submission-neurips-2026/quickstart/)** - Get up and running
- **[Onboarding a Model](https://driveway3662.github.io/neurocast-submission-neurips-2026/onboarding-model/)** - Add your own models
- **[Adding a Task](https://driveway3662.github.io/neurocast-submission-neurips-2026/adding-task/)** - Create custom tasks
- **[Configuration](https://driveway3662.github.io/neurocast-submission-neurips-2026/configuration/)** - Understanding configs
- **[Registry API](https://driveway3662.github.io/neurocast-submission-neurips-2026/api-reference/)** - Function signatures
