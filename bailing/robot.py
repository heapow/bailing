import json
import queue
import threading
import uuid
from abc import ABC
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import argparse
import time

from bailing import (
    recorder,
    player,
    asr,
    llm,
    tts,
    vad,
    memory
)
from bailing.dialogue import Message, Dialogue
from bailing.utils import is_interrupt, read_config, is_segment, extract_json_from_string, is_segment_sentence
from bailing.prompt import sys_prompt

from plugins.registry import Action
from plugins.task_manager import TaskManager

logger = logging.getLogger(__name__)


class Robot(ABC):
    def send_progress(self, step, message, percentage):
        """发送初始化进度到前端"""
        if hasattr(self, 'websocket') and self.websocket and hasattr(self, 'loop') and self.loop:
            try:
                import asyncio
                progress_data = {
                    "type": "init_progress",
                    "step": step,
                    "message": message,
                    "percentage": percentage
                }
                # 在事件循环中发送消息
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send_text(json.dumps(progress_data, ensure_ascii=False)),
                    self.loop
                )
                logger.info(f"发送初始化进度: Step {step} - {message} ({percentage}%)")
            except Exception as e:
                logger.error(f"发送初始化进度失败: {e}")
    
    def __init__(self, config_file, websocket = None, loop = None):
        config = read_config(config_file)
        self.audio_queue = queue.Queue()

        # WebSocket连接（用于发送进度通知）
        self.websocket = websocket
        self.loop = loop
        
        # Step 1: 初始化基础组件 (0-30%)
        self.send_progress(1, "正在初始化录音和播放器...", 5)
        
        self.recorder = recorder.create_instance(
            config["selected_module"]["Recorder"],
            config["Recorder"][config["selected_module"]["Recorder"]]
        )
        
        self.send_progress(1, "正在加载ASR语音识别模型...", 10)
        logger.info("开始加载ASR模型...")

        self.asr = asr.create_instance(
            config["selected_module"]["ASR"],
            config["ASR"][config["selected_module"]["ASR"]]
        )
        
        logger.info("ASR模型加载完成")
        self.send_progress(1, "正在加载对话模型...", 20)

        self.llm = llm.create_instance(
            config["selected_module"]["LLM"],
            config["LLM"][config["selected_module"]["LLM"]]
        )

        self.send_progress(1, "正在加载语音合成模型...", 25)
        
        self.tts = tts.create_instance(
            config["selected_module"]["TTS"],
            config["TTS"][config["selected_module"]["TTS"]]
        )

        self.send_progress(1, "正在加载语音检测模型...", 28)
        
        self.vad = vad.create_instance(
            config["selected_module"]["VAD"],
            config["VAD"][config["selected_module"]["VAD"]]
        )

        self.player = player.create_instance(
            config["selected_module"]["Player"],
            config["Player"][config["selected_module"]["Player"]]
        )
        
        self.send_progress(1, "基础模型加载完成", 30)
        
        # Step 2: 初始化对话记忆 (30-60%)
        self.send_progress(2, "正在加载对话历史...", 35)
        
        # 使用快速加载模式，跳过历史对话处理，加快初始化速度
        self.memory = memory.Memory(config.get("Memory"), lazy_load=True)
        
        self.send_progress(2, "对话记忆快速加载完成", 60)
        
        # Step 3: 准备对话系统 (60-85%)
        self.send_progress(3, "正在构建系统提示词...", 65)
        
        self.prompt = sys_prompt.replace("{memory}", self.memory.get_memory()).strip()
        
        self.send_progress(3, "正在初始化对话管理器...", 70)
        
        self.vad_queue = queue.Queue()
        self.dialogue = Dialogue(config["Memory"]["dialogue_history_path"])
        self.dialogue.put(Message(role="system", content=self.prompt))

        # 保证tts是顺序的
        self.tts_queue = queue.Queue()
        # 初始化线程池
        self.executor = ThreadPoolExecutor(max_workers=10)

        self.vad_start = True

        # 打断相关配置
        self.INTERRUPT = config["interrupt"]
        self.silence_time_ms = int((1000 / 1000) * (16000 / 512))  # ms

        # 线程锁
        self.chat_lock = False

        # 打断状态标志
        self.client_abort = False

        # 事件用于控制程序退出
        self.stop_event = threading.Event()

        self.callback = None

        self.speech = []

        # 初始化单例
        #rag.Rag(config["Rag"])  # 第一次初始化

        self.send_progress(3, "正在加载任务管理器...", 80)
        
        self.task_queue = queue.Queue()
        self.task_manager = TaskManager(config.get("TaskManager"), self.task_queue)
        self.start_task_mode = config.get("StartTaskMode")
        
        self.send_progress(3, "对话系统准备完成", 85)

        # Step 4: 最终准备 (85-100%)
        self.send_progress(4, "正在启动播放器...", 90)
        
        if config["selected_module"]["Player"].lower().find("websocket") > -1:
            self.player.init(websocket, loop)
            self.listen_dialogue(self.player.send_messages)
        
        self.send_progress(4, "系统自检中...", 95)
        
        # 发送初始化完成通知
        self.send_progress(4, "✨ 初始化完成，准备就绪！", 100)
        logger.info("Robot初始化完成！")

    def listen_dialogue(self, callback):
        self.callback = callback
    
    def _stream_vad(self):
        def vad_thread():
            while not self.stop_event.is_set():
                try:
                    data = self.audio_queue.get()
                    vad_statue = self.vad.is_vad(data)
                    self.vad_queue.put({"voice": data, "vad_statue": vad_statue})
                except Exception as e:
                    logger.error(f"VAD 处理出错: {e}")
        consumer_audio = threading.Thread(target=vad_thread, daemon=True)
        consumer_audio.start()

    def _tts_priority(self):
        def priority_thread():
            while not self.stop_event.is_set():
                try:
                    future = self.tts_queue.get(timeout=1)
                    # 检查是否被打断
                    if self.client_abort:
                        logger.info("收到打断信号，跳过当前TTS任务")
                        continue
                    try:
                        tts_file = future.result(timeout=5)
                    except TimeoutError:
                        logger.error("TTS 任务超时")
                        continue
                    except Exception as e:
                        logger.error(f"TTS 任务出错: {e}")
                        continue
                    if tts_file is None:
                        continue
                    # 再次检查是否被打断
                    if self.client_abort:
                        logger.info("收到打断信号，跳过播放")
                        continue
                    self.player.play(tts_file)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"tts_priority priority_thread: {e}")
        tts_priority = threading.Thread(target=priority_thread, daemon=True)
        tts_priority.start()

    def interrupt_playback(self):
        """中断当前的语音播放"""
        logger.info("Interrupting current playback.")
        # 设置打断标志
        self.client_abort = True
        # 清空播放队列
        self.player.stop()
        # 清空TTS队列
        self.clear_tts_queue()
    
    def clear_tts_queue(self):
        """清空TTS队列"""
        with self.tts_queue.mutex:
            self.tts_queue.queue.clear()
        logger.info("TTS队列已清空")

    def shutdown(self):
        """关闭所有资源，确保程序安全退出"""
        logger.info("Shutting down Robot...")
        self.stop_event.set()
        self.executor.shutdown(wait=True)
        self.recorder.stop_recording()
        self.player.shutdown()
        logger.info("Shutdown complete.")

    def start_recording_and_vad(self):
        # 开始监听语音流
        self.recorder.start_recording(self.audio_queue)
        logger.info("Started recording.")
        # vad 实时识别
        self._stream_vad()
        # tts优先级队列
        self._tts_priority()

    def _duplex(self):
        # 处理识别结果
        data = self.vad_queue.get()
        # 识别到vad开始
        if self.vad_start:
            self.speech.append(data)
        vad_status = data.get("vad_statue")
        # 空闲的时候，取出耗时任务进行播放
        if not self.task_queue.empty() and  not self.vad_start and vad_status is None \
                and not self.player.get_playing_status() and self.chat_lock is False:
            result = self.task_queue.get()
            future = self.executor.submit(self.speak_and_play, result.response)
            self.tts_queue.put(future)

        """ 语音唤醒
        if time.time() - self.start_time>=60:
            self.silence_status = True

        if self.silence_status:
            return
        """
        if vad_status is None:
            return
        if "start" in vad_status:
            if self.player.get_playing_status() or self.chat_lock is True:  # 正在播放，打断场景
                if self.INTERRUPT:
                    self.interrupt_playback()
                    self.chat_lock = False
                    self.vad_start = True
                    self.speech.append(data)
                else:
                    return
            else:  # 没有播放，正常
                self.vad_start = True
                self.speech.append(data)
        elif "end" in vad_status and len(self.speech) > 0:
            try:
                logger.debug(f"语音包的长度：{len(self.speech)}")
                self.vad_start = False
                voice_data = [d["voice"] for d in self.speech]
                text, tmpfile = self.asr.recognizer(voice_data)
                self.speech = []
            except Exception as e:
                self.vad_start = False
                self.speech = []
                logger.error(f"ASR识别出错: {e}")
                return
            if not text.strip():
                logger.debug("识别结果为空，跳过处理。")
                return

            logger.debug(f"ASR识别结果: {text}")
            if self.callback:
                self.callback({"role": "user", "content": str(text)})
            self.executor.submit(self.chat, text)
        return True

    def run(self):
        try:
            self.start_recording_and_vad()  # 监听语音流
            while not self.stop_event.is_set():
                self._duplex()  # 双工处理
        except KeyboardInterrupt:
            logger.info("Received KeyboardInterrupt. Exiting...")
        finally:
            self.shutdown()

    def speak_and_play(self, text):
        if text is None or len(text)<=0:
            logger.info(f"无需tts转换，query为空，{text}")
            return None
        # 在TTS转换前检查是否被打断
        if self.client_abort:
            logger.info(f"已被打断，跳过TTS转换：{text[:20]}...")
            return None
        tts_file = self.tts.to_tts(text)
        if tts_file is None:
            logger.error(f"tts转换失败，{text}")
            return None
        # TTS完成后再次检查是否被打断
        if self.client_abort:
            logger.info(f"已被打断，跳过播放：{text[:20]}...")
            return None
        logger.debug(f"TTS 文件生成完毕{self.chat_lock}")
        #if self.chat_lock is False:
        #    return None
        # 开始播放
        # self.player.play(tts_file)
        #return True
        return tts_file

    def chat_tool(self, query):
        # 打印逐步生成的响应内容
        start = 0
        is_first_sentence = True  # 标记是否是第一句话
        try:
            start_time = time.time()  # 记录开始时间
            llm_responses = self.llm.response_call(self.dialogue.get_llm_dialogue(), functions_call=self.task_manager.get_functions())
        except Exception as e:
            #self.chat_lock = False
            logger.error(f"LLM 处理出错 {query}: {e}")
            return []

        tool_call_flag = False
        response_message = []
        # tool call 参数
        function_name = None
        function_id = None
        function_arguments = ""
        content_arguments = ""
        for chunk in llm_responses:
            content, tools_call = chunk
            if content is not None and len(content)>0:
                if len(response_message)<=0 and content=="```":
                    tool_call_flag = True
            if tools_call is not None:
                tool_call_flag = True
                if tools_call[0].id is not None:
                    function_id = tools_call[0].id
                if tools_call[0].function.name is not None:
                    function_name = tools_call[0].function.name
                if tools_call[0].function.arguments is not None:
                    function_arguments += tools_call[0].function.arguments
            if content is not None and len(content) > 0:
                if tool_call_flag:
                    content_arguments+=content
                else:
                    response_message.append(content)
                    response_message_concat = "".join(response_message)
                    end_time = time.time()  # 记录结束时间
                    logger.debug(f"大模型返回时间时间tool: {end_time - start_time} 秒, 生成token={content}")
                    flag_segment, index_segment = is_segment_sentence(response_message_concat, start, is_first_sentence)
                    logger.debug(
                        f"大模型返回时间时间tool: flag_segment={flag_segment} 秒, index_segment={index_segment}, is_first={is_first_sentence}")

                    if flag_segment:
                        segment_text = response_message_concat[start:index_segment + 1]
                        # 为了保证语音的连贯，至少2个字才转tts

                        if len(segment_text) <= max(2, start):
                            continue
                        future = self.executor.submit(self.speak_and_play, segment_text)
                        self.tts_queue.put(future)
                        # futures.append(future)
                        start = index_segment + 1  # len(response_message)
                        # 第一句话已经处理完，后续使用严格断句
                        is_first_sentence = False

        if not tool_call_flag:
            response_message_concat = "".join(response_message)
            if start < len(response_message_concat):
                segment_text = response_message_concat[start:]
                future = self.executor.submit(self.speak_and_play, segment_text)
                self.tts_queue.put(future)
        else:
            # 处理函数调用
            if function_id is None:
                a = extract_json_from_string(content_arguments)
                if a is not None:
                    content_arguments_json = json.loads(a)
                    function_name = content_arguments_json["function_name"]
                    function_arguments = json.dumps(content_arguments_json["args"], ensure_ascii=False)
                    function_id = str(uuid.uuid4().hex)
                else:
                    return []
                function_arguments = json.loads(function_arguments)
            logger.info(f"function_name={function_name}, function_id={function_id}, function_arguments={function_arguments}")
            # 调用工具
            result = self.task_manager.tool_call(function_name, function_arguments)
            if result.action == Action.NOTFOUND: # = (0, "没有找到函数")
                logger.error(f"没有找到函数{function_name}")
                return []
            elif result.action == Action.NONE: # = (1,  "啥也不干")
                return []
            elif result.action == Action.RESPONSE: # = (2, "直接回复")
                future = self.executor.submit(self.speak_and_play, result.response)
                self.tts_queue.put(future)
                return [result.response]
            elif result.action == Action.REQLLM: # = (3, "调用函数后再请求llm生成回复")
                # 添加工具内容
                self.dialogue.put(Message(role='assistant',
                                          tool_calls=[{"id": function_id, "function": {"arguments": json.dumps(function_arguments ,ensure_ascii=False),
                                                                                       "name": function_name},
                                                       "type": 'function', "index": 0}]))

                self.dialogue.put(Message(role="tool", tool_call_id=function_id, content=result.result))
                self.chat_tool(query)
            elif result.action == Action.ADDSYSTEM: # = (4, "添加系统prompt到对话中去")
                self.dialogue.put(Message(**result.result))
                return []
            elif result.action == Action.ADDSYSTEMSPEAK: # = (5, "添加系统prompt到对话中去&主动说话")
                self.dialogue.put(Message(role='assistant',
                                          tool_calls=[{"id": function_id, "function": {
                                              "arguments": json.dumps(function_arguments, ensure_ascii=False),
                                              "name": function_name},
                                                       "type": 'function', "index": 0}]))

                self.dialogue.put(Message(role="tool", tool_call_id=function_id, content=result.response))
                self.dialogue.put(Message(**result.result))
                self.dialogue.put(Message(role="user", content="ok"))
                return self.chat_tool(query)
            else:
                logger.error(f"not found action type: {result.action}")
        return response_message

    def chat(self, query):
        self.dialogue.put(Message(role="user", content=query))
        response_message = []
        # futures = []
        start = 0
        is_first_sentence = True  # 标记是否是第一句话
        self.chat_lock = True
        # 重置打断标志，开始新的对话
        self.client_abort = False
        if self.start_task_mode:
            response_message = self.chat_tool(query)
        else:
            # 提交 LLM 任务
            try:
                start_time = time.time()  # 记录开始时间
                llm_responses = self.llm.response(self.dialogue.get_llm_dialogue())
            except Exception as e:
                self.chat_lock = False
                logger.error(f"LLM 处理出错 {query}: {e}")
                return None
            # 提交 TTS 任务到线程池
            for content in llm_responses:
                response_message.append(content)
                response_message_concat = "".join(response_message)
                end_time = time.time()  # 记录结束时间
                logger.debug(f"大模型返回时间时间tool: {end_time - start_time} 秒, 生成token={content}")
                flag_segment, index_segment = is_segment_sentence(response_message_concat, start, is_first_sentence)
                logger.debug(
                    f"大模型返回时间时间tool: flag_segment={flag_segment} 秒, index_segment={index_segment}, is_first={is_first_sentence}")
                if flag_segment:
                    segment_text = response_message_concat[start:index_segment + 1]
                    # 为了保证语音的连贯，至少2个字才转tts

                    if len(segment_text) <= max(2, start):
                        continue
                    future = self.executor.submit(self.speak_and_play, segment_text)
                    self.tts_queue.put(future)
                    # futures.append(future)
                    start = index_segment + 1  # len(response_message)
                    # 第一句话已经处理完，后续使用严格断句
                    is_first_sentence = False

            # 处理剩余的响应
            response_message_concat = "".join(response_message)
            if start < len(response_message_concat):
                segment_text = response_message_concat[start:]
                future = self.executor.submit(self.speak_and_play, segment_text)
                self.tts_queue.put(future)
            # 等待所有 TTS 任务完成
            """
            for future in futures:
                try:
                    playing = future.result(timeout=5)
                except TimeoutError:
                    logger.error("TTS 任务超时")
                except Exception as e:
                    logger.error(f"TTS 任务出错: {e}")
            """
        self.chat_lock = False
        # 更新对话
        if self.callback:
            self.callback({"role": "assistant", "content": "".join(response_message)})
        self.dialogue.put(Message(role="assistant", content="".join(response_message)))
        self.dialogue.dump_dialogue()
        logger.debug(json.dumps(self.dialogue.get_llm_dialogue(), indent=4, ensure_ascii=False))
        return True

    async def _send_init_progress(self, websocket, step, message, percentage):
        """发送初始化进度通知"""
        try:
            if websocket:
                progress_data = {
                    "type": "init_progress",
                    "step": step,
                    "message": message,
                    "percentage": percentage
                }
                await websocket.send_text(json.dumps(progress_data))
                logger.info(f"发送初始化进度: {message} ({percentage}%)")
        except Exception as e:
            logger.error(f"发送进度通知失败: {e}")

    async def _send_init_complete(self, websocket):
        """发送初始化完成通知"""
        try:
            if websocket:
                complete_data = {
                    "type": "init_complete",
                    "message": "初始化完成，可以开始对话了！"
                }
                await websocket.send_text(json.dumps(complete_data))
                logger.info("发送初始化完成通知")
        except Exception as e:
            logger.error(f"发送完成通知失败: {e}")


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="小福机器人")

    # Add arguments
    parser.add_argument('config_path', type=str, help="配置文件", default=None)

    # Parse arguments
    args = parser.parse_args()
    config_path = args.config_path

    # 创建 Robot 实例并运行
    robot = Robot(config_path)
    robot.run()
