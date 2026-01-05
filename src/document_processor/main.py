"""
提供完整的文档处理流水线，支持批量处理
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.document_processor.processor import DocumentProcessor, create_processor
from src.document_processor.loader import BaseDocumentLoader

from langchain_classic.schema import Document

class BatchDocumentProcessor:
    """
    批量文档处理器
    支持并发处理多个文档，生成处理报告
    """

    def __init__(self, 
                 processor_config: Optional[Dict[str, Any]] = None,
                 max_workers: int = 3,
                 output_dir: str = "data/processed"):
        """
        初始化批量处理器
        
        Args:
            processor_config: 文档处理器配置
            max_workers: 最大并发工作线程数
            output_dir: 输出目录
        """
        self.processor_config = processor_config or {}
        self.max_workers = max_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.processor = create_processor(**self.processor_config)
        self.logger = logging.getLogger("BatchProcessor")

    def process_single(
            self,
            file_path:str,
            save_to_file: bool = False
    ) -> Dict[str, Any]:
        """
        处理单个文档并返回详细结果
        
        Args:
            file_path: 文档路径
            save_to_file: 是否将结果保存到文件
            
        Returns:
            处理结果字典
        """
        result = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "status": "pending",
            "start_time": datetime.now().isoformat(),
            "chunks": [],
            "error": None
        }

        try:
            chunks: List[Document] = self.processor.procecss_cocument(file_path=file_path)

            result.update({
                "status": "success",
                "end_time": datetime.now().isoformat(),
                "chunk_count": len(chunks),
                "chunks_preview": [
                    {
                        "content_preview": chunk.metadata.get("content_preview", ""),
                        "chunk_size": len(chunk.page_content),
                        "metadata_keys": list(chunk.metadata.keys())
                    } for chunk in chunks[:3]
                ]
            })

            if save_to_file:
                self._save_processing_result(file_path, chunks)
            
            self.logger.info(f"✅ 处理完成: {file_path} ({len(chunks)} 个块)")
        
        except Exception as e:
            result.update({
                'status': "error",
                "end_time": datetime.now().isoformat(),
                "error": str(e)
            })
            self.logger.error(f"❌ 处理失败: {file_path} - {e}")
        
        return result
    

    def _save_processing_result(self, file_path:str, chunks: List[Document]):
        """保存处理结果到Json文件"""
        file_name= Path(file_path).name
        output_file = self.output_dir / f"{file_name}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        save_data = {
            "source_file": file_path,
            "processed_time": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for chunk in chunks
            ]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        self.logger.debug(f"处理结果保存到: {output_file}")

    def process_batch(
            self,
            file_paths: List[str],
            save_results: bool = True
    ) -> Dict[str, Any]:
        """
        批量处理多个文档
        
        Args:
            file_paths: 文档路径列表
            save_results: 是否保存处理结果
            
        Returns:
            批量处理报告
        """
        self.logger.info(f"开始批量处理 {len(file_paths)} 个文档")

        report = {
            "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "start_time": datetime.now().isoformat(),
            "total_files": len(file_paths),
            "processed_files": 0,
            "successful": 0,
            "failed": 0,
            "results": [],
            "summary": {}
        }

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single, fp, save_results): fp for fp in file_paths
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    report['results'].append(result)

                    if result['status'] == 'success':
                        report['successful'] += 1
                    else:
                        report['failed'] += 1
                
                except Exception as e:
                    error_result = {
                        "file_path": file_path,
                        "status": 'error',
                        "error": str(e)
                    }
                    report['results'].append(error_result)
                    report['failed'] += 1
        end_time = datetime.now().isoformat()
        report.update({
            "end_time": end_time,
            "processed_files": report["successful"] + report["failed"],
            "summary": {
                "success_rate": report["successful"] / len(file_paths) if len(file_paths) > 0 else 0,
                "average_chunks_per_file": self._calculate_average_chunks(report["results"]),
                "processing_time": self._calculate_processing_time(report, end_time)
            }
        })

        # 保存批处理报告
        self._save_batch_report(report)
        
        self.logger.info(
            f"批量处理完成。成功: {report['successful']}, 失败: {report['failed']}, "
            f"成功率: {report['summary']['success_rate']:.2%}"
        )
        
        return report
    
    def _calculate_average_chunks(self, results: List[Dict]) -> float:
        """计算平均每个文件的块数量"""
        successful_results = [r for r in results if r.get('status') == 'success']
        if not successful_results:
            return 0
        
        total_chunks = sum(r.get("chunk_count", 0) for r in successful_results)
        return total_chunks / len(successful_results)
    
    def _calculate_processing_time(self, report: Dict, end_time:str) -> str:
        """计算处理时间"""

        start= datetime.fromisoformat(report['start_time'])
        end =  datetime.fromisoformat(end_time)
        duration = end - start
        return str(duration)
    
    def _save_batch_report(self, report: Dict):
        """保存批处理报告"""

        report_file = self.output_dir / f"batch_report_{report['batch_id']}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"批处理报告保存到: {report_file}")


def main():
    """主函数，命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="智能文档处理系统")
    parser.add_argument('--input', default="data/documents", help="输入文件或目录路径")
    parser.add_argument("--output", "-o", default="data/processed", help="输出目录")
    parser.add_argument("--chunk-size", type=int, default=1000, help="文本块大小")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="文本块重叠大小")
    parser.add_argument("--workers", '-w', type=int, default=3, help="并发工作线程数")
    parser.add_argument("--test", action='store_true', help="运行测试模式")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.test:
        run_tests()
        return
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 输入路径不存在: {args.input}")
        return
    
    file_paths = []
    if input_path.is_file():
        file_paths = [str(input_path)]
    else:
        # 支持的文件扩展名
        supported_extensions = ['.pdf', '.docx', '.doc', '.pptx', '.ppt']
        for ext in supported_extensions:
             file_paths.extend(list(input_path.glob(f"**/*{ext}")))
        file_paths = [str(fp) for fp in file_paths]

    if not file_paths:
        print(f"❌ 未找到支持的文档文件: {args.input}")
        return
    
    print(f"📂 找到 {len(file_paths)} 个文档文件")

    # 创建批量处理器
    processor_config = {
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap
    }

    batch_processor = BatchDocumentProcessor(
        processor_config=processor_config,
        max_workers=args.workers,
        output_dir=args.output
    )

    # 开始处理
    print("🔄 开始处理文档...")
    report = batch_processor.process_batch(file_paths, save_results=True)
    
    # 显示总结
    print("\n" + "=" * 60)
    print("📊 处理总结")
    print("=" * 60)
    print(f"总文件数: {report['total_files']}")
    print(f"处理成功: {report['successful']}")
    print(f"处理失败: {report['failed']}")
    print(f"成功率: {report['summary']['success_rate']:.2%}")
    print(f"平均每个文件块数: {report['summary']['average_chunks_per_file']:.1f}")
    print(f"处理时间: {report['summary']['processing_time']}")
    print(f"输出目录: {args.output}")
    print("=" * 60)



# 集成测试函数

def run_tests():
    """运行集成测试"""
    print("🧪 运行集成测试...")

    test_dir = Path("data/test_integration")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_files = []

    # 创建测试文本文件并转换为其他格式（模拟）
    test_content = """这是一个集成测试文档。
    
        第1章：文档处理系统
        本系统支持多种文档格式，包括PDF、Word和PPT。

        第2章：功能特点
        1. 自动文本分割
        2. 元数据提取
        3. 批量处理
        4. 质量验证

        第3章：性能指标
        处理速度：约100页/分钟
        准确率：99.5%以上
        支持并发：最多10个文档同时处理
            
        这是一个较长的段落，用于测试文本分割器是否能正确地将长文本分割成适当大小的块。文本分割是自然语言处理中的基础任务，它直接影响后续的检索和生成效果。一个好的分割策略应该保持语义的完整性，同时控制块的大小以便于处理。"""
    
    # 创建不同格式的测试文件
    formats = [
        (".pdf", test_content),
        (".doc", f"# 测试文档\n\n{test_content}"),
    ]

    for ext, content in formats:
        test_file = test_dir / f"test_document{ext}"
        test_file.write_text(content, encoding='utf-8')
        test_files.append(str(test_file))

    print(f"📝 创建了 {len(test_files)} 个测试文件")

    print("\n1. 测试单个文件处理:")

    processor = create_processor(chunk_size=500, chunk_overlap=100)

    for test_file in test_files[:1]:
        try:
            chunks = processor.procecss_cocument(test_file)
            print(f"   ✅ {Path(test_file).name}: {len(chunks)} 个文本块")

            if chunks:
                print(f"      示例块大小: {len(chunks[0].page_content)} 字符")
                print(f"      元数据字段: {list(chunks[0].metadata.keys())}")
        except Exception as e:
            print(f"   ❌ {Path(test_file).name}: 失败 - {e}")
    

    print("\n2. 测试批量处理:")

    batch_processor = BatchDocumentProcessor(
        max_workers=2, output_dir='data/test_output'
    )

    report = batch_processor.process_batch(test_files, save_results=False)

    print(f"   总文件数: {report['total_files']}")
    print(f"   成功: {report['successful']}")
    print(f"   失败: {report['failed']}")
    print(f"   成功率: {report['summary']['success_rate']:.2%}")
    
    # 清理测试文件
    print("\n3. 清理测试文件...")
    for test_file in test_dir.glob("*"):
        test_file.unlink()
    test_dir.rmdir()
    
    print("✅ 集成测试完成")


# ============ 4. 模块导入接口 ============
def get_document_processor(config: Optional[Dict] = None) -> DocumentProcessor:
    """获取文档处理器实例（供其他模块导入）"""
    return create_processor(**(config or {}))


def get_batch_processor(
    max_workers: int = 3,
    output_dir: str = "data/processed"
) -> BatchDocumentProcessor:
    """获取批量处理器实例（供其他模块导入）"""
    return BatchDocumentProcessor(
        max_workers=max_workers,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()