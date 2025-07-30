---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{}
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

This modelcard aims to be a base template for new models. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md?plain=1).

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->



- **Developed by:** [More Information Needed]
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** [More Information Needed]
- **Language(s) (NLP):** [More Information Needed]
- **License:** [More Information Needed]
- **Finetuned from model [optional]:** [More Information Needed]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** [More Information Needed]
- **Paper [optional]:** [More Information Needed]
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

[More Information Needed]

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

[More Information Needed]

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

### Installation

```bash
pip install "huggingface-hub[cli]"
huggingface-cli login # Paste access token w/ read access to this repository.
                      # Tokens look like this: hf_*****
export TEMP_DIR=$(mktemp -d)
huggingface-cli download optimizerai/vocos --exclude "*.safetensors" --local-dir $TEMP_DIR
pip install "file://$TEMP_DIR"
```

Or for an automated approach:

```bash
pip install "huggingface-hub[cli]"
export HF_TOKEN=hf_******
export TEMP_DIR=$(mktemp -d)
huggingface-cli download optimizerai/vocos --exclude "*.safetensors" --local-dir $TEMP_DIR
pip install "file://$TEMP_DIR"
```

If you want to hardcode your token for some reason:

```bash
pip install "huggingface-hub[cli]"
export TEMP_DIR=$(mktemp -d)
huggingface-cli download optimizerai/vocos --exclude "*.safetensors" --local-dir $TEMP_DIR --token hf_*****
pip install "file://$TEMP_DIR"
```

### Example usage

```python
import torch
from vocos import get_voco

mel_voco = get_voco("mel")
encodec_voco = get_voco("encodec")
dac_voco = get_voco("dac")
dac_vae_voco = get_voco("dacvae")
oobleck_voco = get_voco("oobleck")

audio = torch.randn(1, 44100, 2) # [batch, audio_length, audio_channels]
latents = oobleck_voco.encode(audio) # [batch, encoded_length, latent_dim]
recon = oobleck_voco.decode(latents) # [batch, recon_length, audio_channels]
```

Sampling rate: `oobleck_voco.sampling_rate`
Audio channels: `oobleck_voco.channel`

Length conversion:

```python
import torch
from vocos import get_voco

oobleck_voco = get_voco("oobleck")

audio_length = 44100
encode_length = oobleck_voco.encode_length(audio_length)
recon_length = oobleck_voco.decode_length(encode_length)

audio = torch.randn(1, audio_length, oobleck_voco.channel)
latent = oobleck_voco.encode(audio)
recon = oobleck_voco.decode(latent)

assert encode_length == latent.shape[1]
assert recon_length == recon.shape[1]
```
## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

[More Information Needed]

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

[More Information Needed]


#### Training Hyperparameters

- **Training regime:** [More Information Needed] <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary



## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]