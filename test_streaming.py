import asyncio
import httpx
import time

async def test_streaming():
    url = "http://localhost:8880/generate_tts_streaming"
    data = {
        "text": "你好，这是一段流式输出的测试音频。",
        "cfg_value": "2.0",
        "inference_timesteps": "10",
        "do_normalize": "True",
        "denoise": "True"
    }
    
    start_time = time.time()
    first_byte_received = False

    async with httpx.AsyncClient() as client:
        # 模拟文件上传（如果有 prompt_wav）
        # files = {'prompt_wav': open('test.wav', 'rb')}
        
        async with client.stream("POST", url, data=data) as response:
            print(f"状态码: {response.status_code}")
            
            async for chunk in response.aiter_bytes():
                if not first_byte_received:
                    ttfb = time.time() - start_time
                    print(f"🚀 首包到达 (TTFB): {ttfb:.4f} 秒")
                    first_byte_received = True
                
                print(f"收到数据块: {len(chunk)} 字节")

if __name__ == "__main__":
    asyncio.run(test_streaming())