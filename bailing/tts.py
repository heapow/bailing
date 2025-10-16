import asyncio
import logging
import os
import subprocess
import time
import uuid
import threading
from abc import ABC, ABCMeta, abstractmethod
from datetime import datetime
from gtts import gTTS
import edge_tts
import ChatTTS
import torch
import torchaudio
import soundfile as sf
from kokoro import KModel, KPipeline
import requests

logger = logging.getLogger(__name__)


class AbstractTTS(ABC):
    __metaclass__ = ABCMeta

    @abstractmethod
    def to_tts(self, text):
        pass


class GTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file")
        self.lang = config.get("lang")

    def _generate_filename(self, extension=".aiff"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, text):
        tmpfile = self._generate_filename(".aiff")
        try:
            start_time = time.time()
            tts = gTTS(text=text, lang=self.lang)
            tts.save(tmpfile)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.debug(f"生成TTS文件失败: {e}")
            return None


class MacTTS(AbstractTTS):
    """
    macOS 系统自带的TTS
    voice: say -v ? 可以打印所有语音
    """

    def __init__(self, config):
        super().__init__()
        self.voice = config.get("voice")
        self.output_file = config.get("output_file")

    def _generate_filename(self, extension=".aiff"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"执行时间: {execution_time:.2f} 秒")

    def to_tts(self, phrase):
        logger.debug(f"正在转换的tts：{phrase}")
        tmpfile = self._generate_filename(".aiff")
        try:
            start_time = time.time()
            res = subprocess.run(
                ["say", "-v", self.voice, "-o", tmpfile, phrase],
                shell=False,
                universal_newlines=True,
            )
            self._log_execution_time(start_time)
            if res.returncode == 0:
                return tmpfile
            else:
                logger.info("TTS 生成失败")
                return None
        except Exception as e:
            logger.info(f"执行TTS失败: {e}")
            return None


class EdgeTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", "tmp/")
        self.voice = config.get("voice")

    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    async def text_to_speak(self, text, output_file):
        communicate = edge_tts.Communicate(text, voice=self.voice)  # Use your preferred voice
        await communicate.save(output_file)

    def to_tts(self, text):
        tmpfile = self._generate_filename(".wav")
        start_time = time.time()
        try:
            asyncio.run(self.text_to_speak(text, tmpfile))
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.info(f"Failed to generate TTS file: {e}")
            return None


class CHATTTS(AbstractTTS):
    def __init__(self, config):
        self.output_file = config.get("output_file", ".")
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)  # Set to True for better performance
        self.rand_spk = self.chat.sample_random_speaker()

    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")

    def to_tts(self, text):
        tmpfile = self._generate_filename(".wav")
        start_time = time.time()
        try:
            params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=self.rand_spk,  # add sampled speaker
                temperature=.3,  # using custom temperature
                top_P=0.7,  # top P decode
                top_K=20,  # top K decode
            )
            params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt='[oral_2][laugh_0][break_6]',
            )
            wavs = self.chat.infer(
                [text],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )
            try:
                torchaudio.save(tmpfile, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
            except:
                torchaudio.save(tmpfile, torch.from_numpy(wavs[0]), 24000)
            self._log_execution_time(start_time)
            return tmpfile
        except Exception as e:
            logger.error(f"Failed to generate TTS file: {e}")
            return None



# class KOKOROTTS(AbstractTTS):
#     def __init__(self, config):
#         from kokoro import KPipeline
#         self.output_file = config.get("output_file", ".")
#         self.lang = config.get("lang", "z")
#         self.pipeline = KPipeline(lang_code=self.lang)  # <= make sure lang_code matches voice
#         self.voice = config.get("voice", "zm_yunyang")
#
#     def _generate_filename(self, extension=".wav"):
#         return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")
#
#     def _log_execution_time(self, start_time):
#         end_time = time.time()
#         execution_time = end_time - start_time
#         logger.debug(f"Execution Time: {execution_time:.2f} seconds")
#
#     def to_tts(self, text):
#         tmpfile = self._generate_filename(".wav")
#         start_time = time.time()
#         try:
#             generator = self.pipeline(
#                 text, voice=self.voice,  # <= change voice here
#                 speed=1, split_pattern=r'\n+'
#             )
#             for i, (gs, ps, audio) in enumerate(generator):
#                 logger.debug(f"KOKOROTTS: i: {i}, gs：{gs}, ps：{ps}")  # i => index
#                 sf.write(tmpfile, audio, 24000)  # save each audio file
#             self._log_execution_time(start_time)
#             return tmpfile
#         except Exception as e:
#             logger.error(f"Failed to generate TTS file: {e}")
#             return None


class KOKOROTTS(AbstractTTS):
    def __init__(self, config):
        """
        config keys:
          - repo_id: HuggingFace repo ID for Kokoro model (e.g. 'hexgrad/Kokoro-82M-v1.1-zh')
          - lang:      'z' for Chinese, 'a' for multilingual/IPA fallback
          - voice:     e.g. 'zf_001' or 'zm_010'
          - output_dir: directory to write wav files to
          - sample_rate: audio sample rate (default 24000)
          - zero_padding: number of zeros between segments (default 5000)
        """
        self.repo_id      = config.get("repo_id", "hexgrad/Kokoro-82M-v1.1-zh")
        self.output_dir   = config.get("output_dir", ".")
        self.lang         = config.get("lang", "z")
        self.voice        = config.get("voice", "zf_001")
        self.sample_rate  = config.get("sample_rate", 24000)

        # device selection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load model if Chinese TTS
        self.model = None
        if self.lang == "z":
            self.model = KModel(repo_id=self.repo_id).to(self.device).eval()

        # set up pipelines
        self._setup_pipelines()

    def _setup_pipelines(self):
        # English/IPA fallback pipeline
        self.en_pipeline = KPipeline(
            lang_code="a", repo_id=self.repo_id, model=False
        )

        def en_callable(text):
            if text == "Kokoro":
                return "kˈOkəɹO"
            elif text == "Sol":
                return "sˈOl"
            return next(self.en_pipeline(text)).phonemes

        self.en_callable = en_callable

        # Main TTS pipeline
        self.pipeline = KPipeline(
            lang_code=self.lang,
            repo_id=self.repo_id,
            model=self.model,
            en_callable=self.en_callable
        )

    def _generate_filename(self, extension=".wav"):
        fname = f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}"
        return os.path.join(self.output_dir, fname)

    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"Execution Time: {execution_time:.2f} seconds")
    @staticmethod
    def _speed_callable(len_ps: int) -> float:
        """
        piecewise linear function to slow down generation for long inputs
        """
        speed = 0.8
        if len_ps <= 83:
            speed = 1.0
        elif len_ps < 183:
            speed = 1.0 - (len_ps - 83) / 500.0
        return speed * 1.1

    def to_tts(self, text: str) -> str:
        """
        Generate TTS for `text`. Returns path to the combined WAV file.
        """
        output_file = self._generate_filename(".wav")
        start_time = time.time()

        try:
            generator = self.pipeline(
                text.replace("\n", " "),
                voice=self.voice,
                speed=self._speed_callable,
                #split_pattern=r"\n+"
            )
            result = next(generator)
            wav = result.audio
            sf.write(output_file, wav, self.sample_rate)

            self._log_execution_time(start_time)
            return output_file
        except Exception as e:
            logger.error(f"[KOKOROTTS] Failed to generate TTS: {e}")
            return ""


class DifyTTS(AbstractTTS):
    """
    基于Dify平台的TTS服务
    使用Dify的text-to-audio API进行语音合成
    """
    def __init__(self, config):
        self.api_url = config.get("api_url", "http://115.190.170.250:8088/v1")
        self.api_key = config.get("api_key")
        self.output_file = config.get("output_file", "tmp/")
        self.user = config.get("user", "bailing-user")
        self.streaming = config.get("streaming", False)
        
        # 添加请求锁，避免并发问题
        self._request_lock = threading.Lock()
        self._request_count = 0
        
        # 验证配置
        if not self.api_key:
            raise ValueError("DifyTTS requires api_key in configuration")
        
        logger.info(f"DifyTTS initialized with API: {self.api_url}")
    
    def _generate_filename(self, extension=".wav"):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}{extension}")
    
    def _log_execution_time(self, start_time):
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"DifyTTS Execution Time: {execution_time:.2f} seconds")
    
    def to_tts(self, text):
        """
        使用Dify API生成语音
        
        Args:
            text: 要合成的文本
            
        Returns:
            生成的音频文件路径
        """
        # 使用锁避免并发问题
        with self._request_lock:
            self._request_count += 1
            request_id = self._request_count
            
            output_file = self._generate_filename(".wav")
            start_time = time.time()
            
            try:
                # 构建请求
                url = f"{self.api_url}/text-to-audio"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "text": text,
                    "user": self.user
                }
                
                logger.info(f"[#{request_id}] Sending TTS request: {text[:50]}...")
                
                # 发送请求
                response = requests.post(url, headers=headers, json=payload, timeout=30)
                
                # 检查响应状态
                if response.status_code != 200:
                    logger.error(f"[#{request_id}] TTS failed: {response.status_code} - {response.text[:100]}")
                    return None
                
                # 检查内容
                if len(response.content) == 0:
                    logger.error(f"[#{request_id}] Empty response from Dify TTS")
                    return None
                
                # 保存音频文件
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                
                file_size = os.path.getsize(output_file)
                logger.info(f"[#{request_id}] TTS success: {output_file} ({file_size/1024:.2f} KB)")
                
                self._log_execution_time(start_time)
                return output_file
                
            except requests.exceptions.Timeout:
                logger.error(f"[#{request_id}] TTS timeout for text: {text[:30]}...")
                return None
            except requests.exceptions.HTTPError as e:
                logger.error(f"[#{request_id}] HTTP error: {e.response.status_code} - {e.response.text[:100]}")
                return None
            except Exception as e:
                logger.error(f"[#{request_id}] TTS failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return None


def create_instance(class_name, *args, **kwargs):
    # 获取类对象
    cls = globals().get(class_name)
    if cls:
        # 创建并返回实例
        return cls(*args, **kwargs)
    else:
        raise ValueError(f"Class {class_name} not found")
