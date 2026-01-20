import asyncio
import aiohttp
import time
import statistics
import csv
from typing import List, Dict
from datetime import datetime

# ---------------- 基本配置 ----------------
BASE_URL = "http://localhost:8880/generate_tts"

REQUEST_TEMPLATE = {
    "text": "哇，你这个新买的T-shirt好cool啊！是哪个brand的？周末我们去新开的mall里那家Starbucks喝杯coffee吧？我听说他们的new season限定款Latte很OK。"
}

# REQUEST_TEMPLATE = {
#     "text": "澳门在哪里啊",
#     "cfg_value": "2.0",
#     "inference_timesteps": "10",
#     "normalize": "true",
#     "denoise": "true",
#     "prompt_text": "澳门有乜嘢好食嘅？"
# }

# # 音频路径（确保文件存在）
# PROMPT_WAV_PATH = "/home/verachen/Music/voice/2food.wav"

# headers 一般不必指定 multipart，会自动设置
DEFAULT_HEADERS = {}


# ---------------- 请求逻辑 ----------------
async def tts_request(session: aiohttp.ClientSession, request_id: int) -> float:
    """执行一次请求，返回耗时（秒）"""
    start = time.perf_counter()
    try:
        form = aiohttp.FormData()
        for k, v in REQUEST_TEMPLATE.items():
            form.add_field(k, v)     
        # form.add_field(
        #     "prompt_wav",
        #     open(PROMPT_WAV_PATH, "rb"),
        #     filename="2food.wav",
        #     content_type="audio/wav"
        # )           
           
        async with session.post(BASE_URL, data=form, headers=DEFAULT_HEADERS) as resp:
            await resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}")
    except Exception as e:
        print(f"[请求 {request_id}] 出错: {e}")
        return -1
    return time.perf_counter() - start


# ---------------- 并发测试核心 ----------------
async def benchmark(concurrency: int, total_requests: int) -> Dict:
    """在指定并发下发起多次请求，统计性能指标"""
    timings = []
    errors = 0

    conn = aiohttp.TCPConnector(limit=0, force_close=False)
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        sem = asyncio.Semaphore(concurrency)

        async def worker(i):
            nonlocal errors
            async with sem:
                t = await tts_request(session, i)
                if t > 0:
                    timings.append(t)
                else:
                    errors += 1

        tasks = [asyncio.create_task(worker(i)) for i in range(total_requests)]
        await asyncio.gather(*tasks)

    succ = len(timings)
    total_time = sum(timings)
    result = {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successes": succ,
        "errors": errors,
    }

    if succ > 0:
        result.update({
            "min": min(timings),
            "max": max(timings),
            "avg": statistics.mean(timings),
            "median": statistics.median(timings),
            "qps": succ / total_time if total_time > 0 else 0
        })
    return result


# ---------------- CSV 写入 ----------------
def write_to_csv(filename: str, data: List[Dict]):
    """把测试结果写入 CSV 文件"""
    fieldnames = ["concurrency", "total_requests", "successes", "errors",
                  "min", "max", "avg", "median", "qps"]
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"\n✅ 测试结果已保存到: {filename}")


# ---------------- 主流程 ----------------
async def run_tests(concurrency_list: List[int], total_requests: dict):
    print("开始压测接口:", BASE_URL)
    results = []
    for c in concurrency_list:
        print(f"\n=== 并发数: {c} ===")
        res = await benchmark(c, total_requests[c])
        results.append(res)

        print(f"请求总数: {res['total_requests']} | 成功: {res['successes']} | 失败: {res['errors']}")
        if "avg" in res:
            print(f"耗时 (s) → min={res['min']:.3f}, avg={res['avg']:.3f}, median={res['median']:.3f}, max={res['max']:.3f}")
            print(f"近似 QPS: {res['qps']:.2f}")
        else:
            print("所有请求均失败")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    csv_filename = f"tts_benchmark_{timestamp}.csv"
    write_to_csv(csv_filename, results)


if __name__ == "__main__":
    # 你可以根据机器能力调整这两个参数
    concurrency_list = [1, 5, 10, 20, 50, 100, 150, 200]  # 并发测试范围
    total_requests = {
        1: 50,
        5: 100,
        10: 200,
        20: 400,
        50: 1000,
        100: 2000,
        150: 3000,
        200: 4000,
        300: 6000,
        500: 10000
    }                                  # 每个并发等级下的请求数
    asyncio.run(run_tests(concurrency_list, total_requests))


# import asyncio
# import aiohttp
# import time
# import statistics
# from typing import List, Dict

# # 新的接口 URL
# BASE_URL = "http://127.0.0.1:8880/generate_tts"

# # # 固定参数（不包括音频）
# # FORM_PARAMS = {
# #     "text": "澳门在哪里啊",
# #     "cfg_value": "2.0",
# #     "inference_timesteps": "10",
# #     "normalize": "true",
# #     "denoise": "true",
# #     "prompt_text": "澳门有乜嘢好食嘅？"
# # }
# FORM_PARAMS = {
#     "text": "澳门在哪里啊"
# }


# # 音频路径（确保文件存在）
# # PROMPT_WAV_PATH = "/home/verachen/Music/voice/2food.wav"

# # headers 一般不必指定 multipart，会自动设置
# DEFAULT_HEADERS = {}


# async def tts_request(session: aiohttp.ClientSession) -> float:
#     """
#     发起一次 multipart/form-data 格式的 TTS 请求，返回耗时。
#     """
#     start = time.perf_counter()
#     form = aiohttp.FormData()
#     for k, v in FORM_PARAMS.items():
#         form.add_field(k, v)
#     # form.add_field(
#     #     "prompt_wav",
#     #     open(PROMPT_WAV_PATH, "rb"),
#     #     filename="2food.wav",
#     #     content_type="audio/wav"
#     # )

#     async with session.post(BASE_URL, data=form, headers=DEFAULT_HEADERS) as resp:
#         data = await resp.read()
#         if resp.status != 200:
#             raise RuntimeError(f"HTTP {resp.status}, body: {data[:200]!r}")

#     elapsed = time.perf_counter() - start
#     return elapsed


# async def benchmark(concurrency: int, total_requests: int) -> Dict:
#     timings: List[float] = []
#     errors = 0

#     conn = aiohttp.TCPConnector(limit=0)
#     timeout = aiohttp.ClientTimeout(total=None)

#     async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
#         sem = asyncio.Semaphore(concurrency)

#         async def worker():
#             nonlocal errors
#             async with sem:
#                 try:
#                     t = await tts_request(session)
#                     timings.append(t)
#                 except Exception as e:
#                     errors += 1
#                     print("请求出错：", e)

#         tasks = [asyncio.create_task(worker()) for _ in range(total_requests)]
#         await asyncio.gather(*tasks)

#     succ = len(timings)
#     total_time = sum(timings) if timings else 0.0

#     result = {
#         "concurrency": concurrency,
#         "total_requests": total_requests,
#         "successes": succ,
#         "errors": errors,
#     }
#     if succ > 0:
#         result.update({
#             "min": min(timings),
#             "max": max(timings),
#             "avg": statistics.mean(timings),
#             "median": statistics.median(timings),
#             "qps": succ / total_time if total_time > 0 else None
#         })
#     return result


# async def run_tests(concurrency_list: List[int], total_requests: int):
#     print("开始并发测试，目标接口：", BASE_URL)
#     for c in concurrency_list:
#         print(f"\n--- 并发 = {c} ---")
#         res = await benchmark(c, total_requests)
#         print("总请求：", res["total_requests"])
#         print("成功：", res["successes"], "失败：", res["errors"])
#         if "avg" in res:
#             print(f"耗时 min / avg / median / max = "
#                   f"{res['min']:.3f} / {res['avg']:.3f} / {res['median']:.3f} / {res['max']:.3f} 秒")
#             print(f"近似 QPS = {res['qps']:.2f}")
#         else:
#             print("所有请求都失败了")


# if __name__ == "__main__":
#     # 你可以调整这些参数
#     concurrency_list = [200]     # 要试的并发数列表
#     total_requests = 100                      # 每个并发等级下总请求数

#     asyncio.run(run_tests(concurrency_list, total_requests))


# import multiprocessing as mp
# import requests
# import time
# import statistics
# from typing import Dict, List

# BASE_URL = "http://127.0.0.1:8880/generate_tts"
# FORM_PARAMS = {
#     "text": "澳门在哪里啊",
#     "cfg_value": "2.0",
#     "inference_timesteps": "10",
#     "normalize": "true",
#     "denoise": "true",
#     "prompt_text": "澳门有乜嘢好食嘅？"
# }
# PROMPT_WAV_PATH = "/home/verachen/Music/voice/2food.wav"

# def single_request() -> float:
#     """用 requests 同步发一次 multipart/form-data 请求，返回耗时（秒）或抛异常。"""
#     files = {
#         "prompt_wav": ("2food.wav", open(PROMPT_WAV_PATH, "rb"), "audio/wav")
#     }
#     data = FORM_PARAMS.copy()
#     start = time.perf_counter()
#     resp = requests.post(BASE_URL, data=data, files=files, timeout=60)
#     elapsed = time.perf_counter() - start
#     if resp.status_code != 200:
#         raise RuntimeError(f"HTTP {resp.status_code}, body: {resp.text[:200]!r}")
#     return elapsed

# def worker_task(num_requests: int, return_list: mp.Manager().list, err_list: mp.Manager().list):
#     """子进程做 num_requests 次请求，将各次耗时记录到 return_list（共享 list），错误次数记录到 err_list。"""
#     for _ in range(num_requests):
#         try:
#             t = single_request()
#             return_list.append(t)
#         except Exception as e:
#             err_list.append(str(e))

# def run_multiproc(concurrency: int, total_requests: int) -> Dict:
#     """
#     用多个进程模拟并发：
#     - 每个进程发 total_requests／concurrency 次请求（向下取整或略分配）
#     - 或者简单地让每个进程跑 total_requests 次（更激进）
#     """
#     manager = mp.Manager()
#     times = manager.list()
#     errs = manager.list()

#     procs = []
#     # 任务分配：每个子进程跑一部分请求
#     per = total_requests // concurrency
#     if per < 1:
#         per = 1

#     for i in range(concurrency):
#         p = mp.Process(target=worker_task, args=(per, times, errs))
#         p.start()
#         procs.append(p)

#     for p in procs:
#         p.join()

#     timings = list(times)
#     errors = list(errs)
#     succ = len(timings)
#     total_time = sum(timings) if timings else 0.0

#     ret = {
#         "concurrency": concurrency,
#         "total_requests": total_requests,
#         "successes": succ,
#         "errors": len(errors),
#     }
#     if succ > 0:
#         ret.update({
#             "min": min(timings),
#             "max": max(timings),
#             "avg": statistics.mean(timings),
#             "median": statistics.median(timings),
#             "qps": succ / total_time if total_time > 0 else None
#         })
#     return ret

# if __name__ == "__main__":
#     concurrency = 4
#     total_requests = 40
#     print("开始多进程并发测试")
#     result = run_multiproc(concurrency, total_requests)
#     print(result)



