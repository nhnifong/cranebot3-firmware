import asyncio

def foo():
	x = 1/0

async def main():
    cr_foo = asyncio.create_task(asyncio.to_thread(foo))
    x = await cr_foo

asyncio.run(main())