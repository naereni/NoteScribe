import asyncio
import logging
import os

from aiogram import Bot, Dispatcher

from handlers import register_handlers

logger = logging.getLogger(__name__)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(filename)s:%(lineno)d - [%(asctime)s] - %(message)s",
    )
    logger.info("Starting bot")

    try:
        TOKEN = os.environ["TG_API_TOKEN"]
    except UnboundLocalError:
        print("API_TOKEN was not found!")

    bot = Bot(TOKEN)
    dp = Dispatcher(bot)

    register_handlers(dp)

    try:
        await dp.start_polling()
    finally:
        await dp.storage.close()
        await dp.storage.wait_closed()
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.error("Bot stopped!")
