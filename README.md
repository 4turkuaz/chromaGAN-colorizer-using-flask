# chromaGAN-colorizer-using-flask

A basic Flask application to colorize pictures using ChromaGAN.

![Casablanca](examples/casablanca.png?raw=true)

Casablanca, 1942

## Requirements

Prerequisite: Python 3.5.2 & Linux

First, clone the repository.

```bash
git clone https://github.com/4turkuaz/chromaGAN-colorizer-using-flask
cd chromaGAN-colorizer-using-flask/
```

Then, download the pretrained model from link below then replace it with `dummyFile` under `MODEL` directory.

Besides, in the first use, it is required to download the `VGG16` to colorize the pictures. It will be downloaded automatically once you run it.

Requirements could be found in requirements.txt, to install them:

```bash
pip install -r requirements.txt
```

In case you don't want to install them all, there are key libraries to be installed:

keras==2.2.4, tensorflow==1.11.0, Flask==1.1.2

#### PS: You need to download the pretrained model in order to use application, thus it could be found [here](https://drive.google.com/drive/folders/12s4rbLmnjW4e8MmESbfRStGbrjOrahlW). You need to download "my_model_colorization.h5".

## Usage

Within the chromaGAN-colorizer-using-flask directory, run:

```bash
EXPORT FLASK_APP=run.py
flask run
```

Then open link `127.0.0.1:5000` in the browser.

## More examples

![Marilyn Monroe](examples/marilyn_monroe.png?raw=true)

Marilyn Monroe

![Migrant Mother](examples/migrant_mother.png)

Migrant Mother, 1936

![Wilt Chamberlain](examples/wilt_chamberlain.png)

Wilt Chamberlain, 1962

#### This project is based on:
- [ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution](https://arxiv.org/pdf/1907.09837.pdf)
- [Official implementation of ChromaGAN](https://github.com/pvitoria/ChromaGAN)


## License
[MIT](https://choosealicense.com/licenses/mit/)
