"""
集成文档加载、分块、元数据提取等功能
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from langchain_classic.schema import Document
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownTextSplitter


from .loader import BaseDocumentLoader, get_text_splitter


@dataclass
class ProcessingConfig:
    """文档处理配置"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    splitter_type: str = 'recursive' # recursive, character, markdown
    extract_metadata: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 2000



class DocumentProcessor:
    """
    文档处理器：负责文档加载、分割和预处理的全流程
    
    功能：
    1. 多格式文档加载 (PDF, Word, PPTX)
    2. 多种分块策略 (递归字符、字符、Markdown)
    3. 元数据提取和增强
    4. 分块质量验证
    """

    def __init__(self,
               config: Optional[ProcessingConfig]= None, 
               logger: Optional[logging.Logger]= None):
        """
        初始化文档处理器
        
        Args:
            config: 处理配置，如未提供则使用默认配置
            logger: 日志记录器，如未提供则创建新的
        """
        self.config = config or ProcessingConfig()
        self.logger = logger or self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("DocumentProcessor")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _get_splitter(self) -> Any:
        """根据配置获取文本分割器"""
        if self.config.splitter_type == 'character':
            return CharacterTextSplitter(
                chunk_size = self.config.chunk_size,
                chunk_overlap = self.config.chunk_overlap,
                separator='\n'
            )
        elif self.config.splitter_type == 'markdown':
            return MarkdownTextSplitter(
                chunk_size = self.config.chunk_size,
                chunk_overlap = self.config.chunk_overlap,
            )
        else:
            return get_text_splitter(
                chunk_size = self.config.chunk_size,
                chunk_overlap = self.config.chunk_overlap,
            )
        
    def _enhance_metadata(self, document: Document, file_path: str, chunk_index:int) -> Dict[str, Any]:
        """
        增强文档快的元数据
        为每个文本块添加丰富的上下文信息，使后续的检索、分析和展示更加智能。
        """
        metadata = document.metadata.copy() if document.metadata else {}
        # 添加文件信息
        metadata.update({
            "source_file": file_path,
            "file_name": Path(file_path).name,
            "file_type": Path(file_path).suffix.lower(),
            "chunk_index": chunk_index,
            "chunk_size": len(document.page_content),
            "processor": "DocumentProcessor_v1.0",
        })

        # 提取前几个字符作为摘要
        content_preview = document.page_content[:100].replace("\n", " ")
        metadata["content_preview"] = content_preview

        return metadata
    
    def _validate_chunk(self, chunk:Document) -> Tuple[bool, str]:
        """验证文本块质量"""
        content = chunk.page_content.strip()

        # 检查内容
        if not content:
            return False, '空内容'
        
        if len(content) < self.config.min_chunk_size:
            return False, '内容过短'
        
        if len(content) > self.config.max_chunk_size:
            return False, '内容过多'
        
        return True, '有效'
    
    def procecss_cocument(
            self,
            file_path: str,
            return_raw: bool = False
    ) -> List[Document]:
        """处理单个文档的全流程"""

        self.logger.info(f"开始处理文档 {file_path}")

        try:
            loader = BaseDocumentLoader.get_loader(file_path)
            raw_documents = loader.load()
            self.logger.info(f"文档加载成功， 共 {len(raw_documents)} 个原始页面/部分")

            splitter = self._get_splitter()
            chunks = splitter.split_documents(raw_documents)
            self.logger.info(f"文本分割完成， 共{len(chunks)}个文本块")

            if return_raw:
                return chunks
            
            # 处理每个文本块，验证 + 元数据增强
            processed_chunks = []
            valid_count = 0
            invalid_count = 0

            for i, chunk in enumerate(chunks):
                # 验证文本块的质量
                is_valid, reason = self._validate_chunk(chunk)
                if is_valid:
                    # 增强元数据
                    enhanced_metadata = self._enhance_metadata(chunk, file_path, i)
                    processed_chunk = Document(page_content=chunk.page_content)
                    processed_chunk.metadata = enhanced_metadata
                    processed_chunks.append(processed_chunk)
                    valid_count += 1
                else:
                    self.logger.info(f"文本块被过滤，原因是 {reason}")
                    invalid_count += 1
            
            self.logger.info(
                f"文档处理完毕，有效文本块的数量： {valid_count}, 无效文本块： {invalid_count}"
            )
            return processed_chunks
        except Exception as e:
            self.logger.error(f"处理文档的时候出错： {e}")
            raise

    
    def compare_split_strategies(self, file_path:str, strategies: List[str] = None) -> Dict[str, Any]:
        """比较不同分块策略的效果"""

        if strategies is None:
            strategies = ["recursive", "character", "markdown"]
        
        results = {}
        
        for strategy in strategies:
            try:
                original_strategy = self.config.splitter_type
                self.config.splitter_type = strategy

                chunks = self.procecss_cocument(file_path=file_path, return_raw= True)

                total_chunks = len(chunks)
                avg_length = sum(len(c.page_content) for c in chunks) / total_chunks if total_chunks > 0 else 0
                max_length = max((len(c.page_content) for c in chunks), default=0)
                min_length = min((len(c.page_content) for c in chunks), default=0)

                results[strategy] = {
                    "chunk_count": total_chunks,
                    'avg_chunk_size': round(avg_length, 2),
                    'max_chunk_size': max_length,
                    'min_chunk_size': min_length,
                    'sample_chunks': [
                        c.page_content[:100] + "..." for c in chunks[:2]
                    ] if chunks else []
                }

                self.config.splitter_type = original_strategy
            except Exception as e:
                results[strategy] = {'error': str(e)}
        
        return results
    


def create_processor(
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        splitter_type: str = 'recursive'
) -> DocumentProcessor:
    """工厂函数，创建配置好的文档处理器"""
    config = ProcessingConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        splitter_type=splitter_type
    )
    return DocumentProcessor(config=config)

    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_file = "data/documents/test.pdf"
    test_file_path = Path(test_file)

    if test_file_path.exists():
        try:
            print("=" * 60)
            print("📄 测试文档处理器")
            print("=" * 60)

            processor = create_processor()
            print('文档处理器创建成功')

            print(f"\n 1. 处理文档 {test_file}")
            processed_chunks = processor.procecss_cocument(test_file)
            print(f"处理完成 得到 {len(processed_chunks)} 个有效文本块")


            if processed_chunks:
                # 显示第一个文本块的元数据
                first_chunk = processed_chunks[0]
                print(f"\n 2. 第一个文本块的元数据")
                for key, value in first_chunk.metadata.items():
                    print(f"{key}: {value}")
                
                # 显示内容预览
                preview = first_chunk.page_content[:150] + "..." if len(first_chunk.page_content) > 150 else first_chunk.page_content
                print(f"\n 3. 内容预览：\n {preview}")
            
            # 测试比较不同的分块策略
            print(f"\n 比较不同的分块策略")
            strategies_results = processor.compare_split_strategies(test_file)
            for strategy, result in strategies_results.items():
                print(f'\n {strategy.upper()} 策略')
                if 'error' in result:
                    print(f"错误: {result['error']}")
                else:
                    print(f"     文本块数量: {result['chunk_count']}")
                    print(f"     平均大小: {result['avg_chunk_size']} 字符")
                    print(f"     最大大小: {result['max_chunk_size']} 字符")
                    print(f"     最小大小: {result['min_chunk_size']} 字符")
            
            print("\n" + "=" * 60)
            print("✅ 所有测试完成")
            print("=" * 60)
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ 测试文件不存在: {test_file}")
        print("请确保已将测试文档放入 data/documents/ 目录")






