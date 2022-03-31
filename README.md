# Telegram bot for handwritten text recognition

## Install

### Poerty
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
poetry self update --preview
```

### htr-tg-bot
```  bash
git clone --recursive https://github.com/naereni/htr-tg-bot.git
sh setup.sh
```

If you will you encounter a problem with ctcdecode - you should run several times ```python setup.py install``` in directory **htr-tg-bot/third_party/ctcdecode**
