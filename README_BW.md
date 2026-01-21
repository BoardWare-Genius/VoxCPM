# VoxCPM-TTS

https://github.com/BoardWare-Genius/VoxCPM

## 📦 VoxCPM-TTS Version History

| Version | Date       | Summary                         |
|---------|------------|---------------------------------|
| 0.0.2   | 2026-01-21 | Supports streaming               |
| 0.0.1   | 2026-01-20 | Initial version                 |

### 🔄 Version Details

#### 🆕 0.0.2 – *2026-01-21*

- ✅ **Core Features**
  - Update Model Weights, use VoxCPM1.5 and model parameters
  - Supports streaming 

#### 🆕 0.0.1 – *2026-01-20*

- ✅ **Core Features**
  - Initial VoxCPM-TTS

---


# Start
```bash
docker pull harbor.bwgdi.com/library/voxcpmtts:0.0.2

docker run -d --restart always -p 5001:5000 --gpus '"device=0"' --mount type=bind,source=/Workspace/NAS11/model/Voice/VoxCPM,target=/models harbor.bwgdi.com/library/voxcpmtts:0.0.2
```

# Usage

## Non-streaming
```bash
curl --location 'http://localhost:5001/generate_tts' \
--form 'text="你好，这是一段测试文本"' \
--form 'prompt_text="这是提示文本"' \
--form 'cfg_value="2.0"' \
--form 'inference_timesteps="10"' \
--form 'do_normalize="true"' \
--form 'denoise="true"' \
--form 'retry_badcase="true"' \
--form 'retry_badcase_max_times="3"' \
--form 'retry_badcase_ratio_threshold="6.0"' \
--form 'prompt_wav=@"/assets/2food16k_2.wav"'
```

## Streaming
```bash
curl --location 'http://localhost:5001/generate_tts_streaming' \
--form 'text="你好，这是一段测试文本"' \
--form 'prompt_text="这是提示文本"' \
--form 'cfg_value="2.0"' \
--form 'inference_timesteps="10"' \
--form 'do_normalize="true"' \
--form 'denoise="true"' \
--form 'retry_badcase="true"' \
--form 'retry_badcase_max_times="3"' \
--form 'retry_badcase_ratio_threshold="6.0"' \
--form 'prompt_wav=@"/Workspace/NAS11/model/Voice/assets/2food16k_2.wav"'
```
