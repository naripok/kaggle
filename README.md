# [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/c/feedback-prize-2021)

## Requirements

- Docker
- Make

## Running

```bash
make dev
```

Grab the produced link (`http://127.0.0.1:8888/lab?token=<token>`) and paste it into your web browser

## Credentials

_TL;DR_: Get API credentials from kaggle and put it into `./.kaggle/kaggle.json`

See [documentation](https://github.com/Kaggle/kaggle-api#api-credentials) for the long explanation

## Kaggle secrets

Add any credentials you like to use in the notebook environments to `./.kaggle/secrets.json`.
Example for weights and biases experiment tracking:

```python
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("wandb_api") // get it by the same key you've used in secrets.json

import wandb
wandb.login(key=api_key)
wandb.init(project="feedback_prize", entity="naripok")
```
