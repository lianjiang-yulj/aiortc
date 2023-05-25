import asyncio
import time
from functools import partial
import asyncio

async def get_html(url):
    print("start get url")
    await asyncio.sleep(2)
    return "bobby"

def callback(url, future):
    print(url)
    print("send email to bobby")

if __name__ == "__main__":
    start_time = time.time()
    #loop = asyncio.get_event_loop()
    #task = loop.create_task(get_html("http://www.imooc.com"))
    #loop.run_until_complete(task)
    #print(task.result())
    asyncio.run(get_html(None))
