"""
文档加载器模块 (LangChain 1.0+ 版本)
支持 PDF, Word( DOCX/DOC ), PPTX 格式
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional
from pathlib import Path


from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader

from langchain_classic.schema import Document

class BaseDocumentLoader(ABC):
    """文档加载器抽象基类"""

    @abstractmethod
    def load(self) -> List[Document]:
        """加载文档并返回Document对象列表"""
        pass

    @staticmethod
    def get_loader(file_path: str):
        """
        工厂方法：根据文件扩展名返回对应的加载器实例
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在{file_path}")
        
        suffix = path.suffix.lower()

        # 根据文件扩展名选择加载器
        if suffix == '.pdf':
            return PDFDocumentLoader(file_path)
        elif suffix in ['.docx', '.doc']:
            return WordDocumentLoader(file_path)
        elif suffix in  ['.pptx', '.ppt']:
            return PPTXDocumentLoader(file_path)
        else:
            raise ValueError(f"不支持得到文件格式：{suffix}. 支持格式：.pdf、.docx, .doc, .pptx, .ppt")


class PDFDocumentLoader(BaseDocumentLoader):
    """PDF文档加载器"""
    def __init__(self, file_path:str):
        self.file_path = file_path
        self._loader = PyPDFLoader(file_path)  # uv add pypdf
    
    def load(self):
        return self._loader.load()


class WordDocumentLoader(BaseDocumentLoader):
    """Word 文档加载器"""
    def __init__(self, file_path):
        self.file_apth = file_path
        self._loader = Docx2txtLoader(file_path)  # uv add docx2txt

    def load(self):
        return self._loader.load()

class PPTXDocumentLoader(BaseDocumentLoader):
    """PPTX 文档加载器"""

    def __init__(self, file_path:str):
        self.file_path = file_path
        self._loader = UnstructuredPowerPointLoader(file_path)   # uv add unstructured[pdf,pptx]
    
    def load(self):
        return self._loader.load()



from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

def get_text_splitter(
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
) -> RecursiveCharacterTextSplitter:
    """
    获取配置好的文本分割器, 将长文本切分成适合处理的小块
    
    Args:
        chunk_size: 每个文本块的最大字符数
        chunk_overlap: 块之间的重叠字符数
        separators: 分割文本的分隔符列表
        
    Returns:
        配置好的 RecursiveCharacterTextSplitter 实例
    """

    if separators is None:
        # 中文友好的分隔符列表
        separators = ["\n\n", "\n", "。", "？", "！", "；", "，", " ", ""]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        is_separator_regex=False,  # 指定分隔符是否为正则表达式模式
        length_function=len, # 定义如何计算文本长度的函数
    )



if __name__ == "__main__":
    test_file = 'data/documents/test.docx'

    test_file_path  = Path(test_file)
    if test_file_path.exists():
        try:
            loader = BaseDocumentLoader.get_loader(test_file_path)
            print(f"✅ 加载器创建成功: {type(loader).__name__}")

            documents = loader.load()
            print(f"✅ 文档加载成功，共 {len(documents)} 页")

            splitter: RecursiveCharacterTextSplitter = get_text_splitter()
            texts = splitter.split_documents(documents)
            print(f"✅ 文本分割成功，共 {len(texts)} 个文本块")

            for i, text in enumerate(texts[:2]):
                preview = text.page_content[:100] + '...' if len(text.page_content) > 100 else text.page_content
                print(f"\n文本块 {i+1} 预览:\n{preview}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    else:
        print(f"⚠️ 测试文件不存在，请创建: {test_file}")
        print("或者使用你自己的文档路径进行测试")

