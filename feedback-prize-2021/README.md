# [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/c/feedback-prize-2021)

## Requirements

- Docker
- Make

## Credentials

*TL;DR*: Get API credentials from kaggle and put it into `./.kaggle/kaggle.json`

See [documentation](https://github.com/Kaggle/kaggle-api#api-credentials) for the long explanation

## Running

```bash
make build
make fetch-data
make dev
```

Grab the produced link (`http://127.0.0.1:8888/lab?token=<token>`) and paste it into your web browser
