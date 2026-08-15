# LangChain 视频教程代码整理

这套目录按视频中真正出现代码的主题拆成了 11 个 Python 示例文件。`Intro`、`LangChain Ecosystem`、`Environment Setup`、`Outro` 属于概念或环境配置，不单独算代码文件。

## 代码文件清单

1. `01_simple_agent_weather.py`：用 `create_agent` 创建天气 Agent，并把普通 Python 函数包装成工具。
2. `02_standalone_model_inference.py`：不用 Agent，直接用 `init_chat_model` 调模型。
3. `03_conversation_history.py`：用 `SystemMessage`、`HumanMessage`、`AIMessage` 传递对话历史。
4. `04_streaming_responses.py`：用 `model.stream()` 流式输出。
5. `05_advanced_agent_context_memory_structured_output.py`：上下文、用户定位、工具调用、结构化输出和线程记忆。
6. `06_multimodal_input.py`：把本地图片或 URL 图片作为多模态输入传给模型。
7. `07_rag_example.py`：用 OpenAI Embeddings + FAISS 做相似性搜索，并把检索器接进 Agent。
8. `08_dynamic_system_prompt.py`：用 middleware 根据用户角色动态切换系统提示词。
9. `09_dynamic_model_choice.py`：用 middleware 根据请求复杂度动态切换模型。
10. `10_custom_agent_middleware.py`：自定义 `AgentMiddleware`，观察 Agent 生命周期钩子。
11. `11_builtin_middleware_examples.py`：展示摘要 middleware，并列出视频提到的常见内置 middleware 场景。

## 怎么运行

建议先创建虚拟环境，然后安装依赖：

```bash
pip install -r langchain_course_examples/requirements.txt
```

复制 `.env.example` 为 `.env`，填入自己的 API key。然后运行任意示例：

```bash
python langchain_course_examples/01_simple_agent_weather.py
```

## 这门教程要掌握什么

核心目标不是背 API，而是建立 LangChain 的几个基本心智模型：

- 模型抽象：用统一接口调用不同供应商的聊天模型。
- 消息抽象：用系统消息、人类消息、AI 消息组织上下文。
- Agent 抽象：让模型在系统提示词、工具、状态和中间件之间协同工作。
- 工具调用：把普通函数变成模型可调用的能力。
- 上下文与记忆：用 runtime context 传业务信息，用 thread_id 保留对话状态。
- 结构化输出：让模型输出可被程序直接读取的数据对象。
- 多模态输入：用文本 + 图片构造输入。
- RAG：把外部知识先嵌入到向量空间，再检索相关片段交给 Agent。
- Middleware：在请求和响应中间插入策略，例如动态提示词、动态模型选择、摘要、限流、人审、回退等。

## 后续最值得复用的知识点

- 工具模式：任何 API、数据库查询、文件操作都可以包装成 tool。
- RAG 模式：个人知识库、客服知识库、课程笔记问答都能复用。
- 动态系统提示词：同一个 Agent 可以根据用户身份、水平、场景改变回答风格。
- 动态模型选择：简单问题用便宜模型，复杂任务切换到强模型。
- 结构化输出：适合做表单抽取、数据归类、自动报告和工作流输入。
- 线程记忆：适合聊天机器人、学习助手、长期任务助手。
- Middleware：适合加企业级控制，比如审计、限流、人工确认、敏感信息处理和失败重试。
