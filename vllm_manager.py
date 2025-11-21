import subprocess
import time
import requests
import signal
import sys
import atexit

class VLLMManager:
    def __init__(self):
        self.process = None
    
    def start(self):
        """启动vLLM服务"""
        if self.is_running():
            print("vLLM server is already running")
            return True
        
        print("Starting vLLM server...")
        self.process = subprocess.Popen([
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "meta-llama/Llama-3.1-8B-Instruct",
            "--host", "0.0.0.0", 
            "--port", "8001",
            "--served-model-name", "Llama3.1-8B-Instruct",
            "--max-model-len", "8192",
            "--gpu-memory-utilization", "0.8",
            "--dtype", "float16"
        ])
        
        # 等待服务就绪
        for i in range(60):
            if self.is_running():
                print("vLLM server started successfully")
                return True
            time.sleep(1)
        
        print("Failed to start vLLM server")
        return False
    
    def stop(self):
        """停止vLLM服务"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None
            print("vLLM server stopped")
    
    def is_running(self):
        """检查服务是否在运行"""
        try:
            response = requests.get("http://localhost:8001/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# 全局管理器实例
vllm_manager = VLLMManager()

def cleanup():
    """清理函数"""
    vllm_manager.stop()

# 注册退出时的清理函数
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda s, f: cleanup())
signal.signal(signal.SIGTERM, lambda s, f: cleanup())


if __name__ == "__main__":
    try:
        vllm_manager.start()
        # 保持主进程运行，否则脚本启动后立即退出
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Shutting down...")
        vllm_manager.stop()