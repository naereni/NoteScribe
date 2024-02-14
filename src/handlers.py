import asyncio
from random import randrange

import aiofiles.os as aios
from aiogram import Dispatcher
from aiogram.types import Message

from run_model import htr_predict, init_predictor


async def admin_start(message: Message) -> None:
    text = [
        "Привет спишь?",
        "Это короче распознавалка рукописного текста прямо по фоткe тетрадки.",
        "Чтобы протестировать - просто отправь фотку сюда",
        "Нечего отправлять? - посмотри примеры по команде /examples",
        "Хочешь посмотреть исходный код - /source",
    ]
    await message.answer("\n".join(text))


async def get_examples(message: Message) -> None:
    await message.reply("Отсылаю")
    await message.answer_photo(open("cache/examples/1.jpg", "rb"))
    await message.answer_photo(open("cache/examples/2.jpg", "rb"))
    await message.answer_photo(open("cache/examples/3.jpg", "rb"))


async def get_source(message: Message) -> None:
    await message.answer("https://github.com/naereni/NoteScribe/")


class Recognition:
    def __init__(self) -> None:
        self.predictor = init_predictor()
        self.queue: int = 0

    async def __call__(self, message: Message) -> None:
        self.queue += 1
        file_idx = randrange(10**5)
        input_filename = f"cache/input/{file_idx}.jpg"
        pred_filename = f"cache/pred/{file_idx}.jpg"
        await message.reply(f"Фото в очереди под номером: {self.queue}")
        await message.photo[-1].download(input_filename)
        await htr_predict(input_filename, pred_filename, self.predictor)
        while True:
            res = await aios.path.exists(pred_filename)
            if res is True:
                await message.reply_photo(open(pred_filename, "rb"))
                break
            else:
                await asyncio.sleep(5)
        self.queue -= 1


def register_handlers(dp: Dispatcher) -> None:
    recognition = Recognition()
    dp.register_message_handler(admin_start, commands=["start"])
    dp.register_message_handler(get_source, commands=["source"])
    dp.register_message_handler(get_examples, commands=["examples"])
    dp.register_message_handler(recognition, content_types=["photo"])
