# VoxCPM-TTS

https://github.com/BoardWare-Genius/VoxCPM

## 📦 VoxCPM-TTS Version History

| Version | Date       | Summary                         |
|---------|------------|---------------------------------|
| 0.0.1   | 2026-01-20 | Initial version                 |

### 🔄 Version Details

#### 🆕 0.0.1 – *2026-01-20*

- ✅ **Core Features**
  - Initial VoxCPM-TTS

---


# Start
```bash
docker pull harbor.bwgdi.com/library/voxcpmtts:0.0.1

docker run -d --restart always -p 5001:5000 --gpus all --mount type=bind,source=/Workspace/NAS11/model,target=/models harbor.bwgdi.com/library/voxcpmtts:0.0.1
```

# Usage
```bash
curl --location 'http://localhost:5001/generate_tts' \
--form 'text="你好，这是一段测试文本"' \
--form 'prompt_text="这是提示文本"' \
--form 'cfg_value="2.0"' \
--form 'inference_timesteps="10"' \
--form 'do_normalize="true"' \
--form 'denoise="true"' \
--form 'prompt_wav=@"/assets/2play16k_2.wav"'
```
