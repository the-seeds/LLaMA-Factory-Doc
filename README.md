# LlamaFactory Document

Document for https://github.com/hiyouga/LlamaFactory

Visit https://llamafactory.readthedocs.io/ for the document.

## Contribution

Doc contribution welcome. Before creating a PR, please check and test your docs locally as follows:

1. Step into the path `docs`:

```shell
cd docs
```

2. Install the required dependencies:

```shell
pip install -r requirements.txt
```

3. Build

```shell
make html
```

4. Server on localhost for doc preview:

```shell
python -m http.server -d build/html 8008
```

### Translation

For translation, you can refer to the format of [example](https://github.com/hiyouga/LLaMA-Factory-Doc/blob/main/docs/locales/en/LC_MESSAGES/index.po) to translate the documents into English.
After completing the translation, you can run the command:

```bash
# LLaMA-Factory-Doc/docs
sphinx-build -b html -D language=en ./source/ build/html/en
```

to build HTML from the .po files.

You can preview the corresponding HTML files in `LLaMA-Factory-Doc/docs/build/html/en` before updating.



## Acknowledgement

This repo benefits from [Qwen2](https://github.com/QwenLM/Qwen2). Thanks for their wonderful works.
