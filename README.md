# Telegram bot for handwritten text recognition

## Install

To install you must have:
- ```GNU/Linux```
- ```python3.9```
- ```g++```

### Poerty
```
curl -sSL https://install.python-poetry.org | python3 -
poetry self update --preview
reboot # add $poetry to $PATH
```

### htr-tg-bot
```  bash
git clone --recursive https://github.com/naereni/htr-tg-bot.git
cd htr-tg-bot && sh setup.sh
```

If you will you encounter a problem with ctcdecode - you should run several times ```python setup.py install``` in directory **htr-tg-bot/third_party/ctcdecode**
