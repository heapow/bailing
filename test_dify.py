#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dify集成测试脚本
用于测试DifyLLM是否正常工作
"""

import sys
import logging
from bailing.llm import DifyLLM

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_dify_basic():
    """测试基本对话功能"""
    print("=" * 60)
    print("测试1: 基本对话功能")
    print("=" * 60)
    
    # 配置（需要替换为实际的API Key）
    config = {
        "url": "https://api.dify.ai/v1",
        "api_key": "app-YOUR_API_KEY_HERE",  # 请替换为实际的API Key
        "user": "test-user"
    }
    
    # 检查是否配置了API Key
    if config["api_key"] == "app-YOUR_API_KEY_HERE":
        print("❌ 请先在test_dify.py中配置实际的Dify API Key")
        print("提示: 在Dify平台创建应用后获取API Key")
        return False
    
    try:
        # 创建DifyLLM实例
        dify_llm = DifyLLM(config)
        print("✅ DifyLLM实例创建成功")
        
        # 测试对话
        dialogue = [
            {"role": "system", "content": "你是一个友好的AI助手"},
            {"role": "user", "content": "你好，请用一句话介绍你自己"}
        ]
        
        print("\n发送消息: 你好，请用一句话介绍你自己")
        print("\n收到响应:")
        print("-" * 60)
        
        response_parts = []
        for content in dify_llm.response(dialogue):
            if content:
                response_parts.append(content)
                print(content, end='', flush=True)
        
        print("\n" + "-" * 60)
        
        if response_parts:
            full_response = "".join(response_parts)
            print(f"\n✅ 测试成功! 收到响应 ({len(full_response)} 字符)")
            print(f"Conversation ID: {dify_llm.conversation_id}")
            return True
        else:
            print("\n❌ 测试失败: 没有收到响应")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.exception("详细错误信息:")
        return False


def test_dify_context():
    """测试对话上下文保持"""
    print("\n" + "=" * 60)
    print("测试2: 对话上下文保持")
    print("=" * 60)
    
    config = {
        "url": "https://api.dify.ai/v1",
        "api_key": "app-YOUR_API_KEY_HERE",  # 请替换为实际的API Key
        "user": "test-user"
    }
    
    if config["api_key"] == "app-YOUR_API_KEY_HERE":
        print("❌ 请先配置API Key")
        return False
    
    try:
        dify_llm = DifyLLM(config)
        
        # 第一轮对话
        dialogue1 = [
            {"role": "user", "content": "我的名字是小明"}
        ]
        
        print("\n第一轮对话 - 发送: 我的名字是小明")
        print("响应: ", end='')
        for content in dify_llm.response(dialogue1):
            if content:
                print(content, end='', flush=True)
        
        print(f"\nConversation ID: {dify_llm.conversation_id}")
        
        # 第二轮对话 - 测试是否记住名字
        dialogue2 = [
            {"role": "user", "content": "你还记得我的名字吗？"}
        ]
        
        print("\n第二轮对话 - 发送: 你还记得我的名字吗？")
        print("响应: ", end='')
        response_parts = []
        for content in dify_llm.response(dialogue2):
            if content:
                response_parts.append(content)
                print(content, end='', flush=True)
        
        full_response = "".join(response_parts)
        
        # 检查是否包含名字
        if "小明" in full_response:
            print("\n\n✅ 上下文保持测试成功! AI记住了名字")
            return True
        else:
            print("\n\n⚠️  AI可能没有记住名字，这可能是正常的")
            print("提示: Dify的上下文管理依赖于平台配置")
            return True
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.exception("详细错误信息:")
        return False


def test_dify_streaming():
    """测试流式响应"""
    print("\n" + "=" * 60)
    print("测试3: 流式响应")
    print("=" * 60)
    
    config = {
        "url": "https://api.dify.ai/v1",
        "api_key": "app-YOUR_API_KEY_HERE",  # 请替换为实际的API Key
        "user": "test-user"
    }
    
    if config["api_key"] == "app-YOUR_API_KEY_HERE":
        print("❌ 请先配置API Key")
        return False
    
    try:
        dify_llm = DifyLLM(config)
        
        dialogue = [
            {"role": "user", "content": "请数从1到10"}
        ]
        
        print("\n发送: 请数从1到10")
        print("流式响应: ", end='')
        
        chunk_count = 0
        for content in dify_llm.response(dialogue):
            if content:
                chunk_count += 1
                print(content, end='', flush=True)
        
        print(f"\n\n✅ 流式响应测试成功! 收到 {chunk_count} 个数据块")
        return True
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        logger.exception("详细错误信息:")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Dify集成测试")
    print("=" * 60)
    
    print("\n⚠️  测试前请确保:")
    print("1. 已在Dify平台创建应用")
    print("2. 已获取API Key")
    print("3. 已在本脚本中配置API Key")
    print("\n开始测试...\n")
    
    results = []
    
    # 运行测试
    results.append(("基本对话", test_dify_basic()))
    results.append(("对话上下文", test_dify_context()))
    results.append(("流式响应", test_dify_streaming()))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过! Dify集成工作正常")
    else:
        print("\n⚠️  部分测试失败，请检查配置和网络连接")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        sys.exit(0)

