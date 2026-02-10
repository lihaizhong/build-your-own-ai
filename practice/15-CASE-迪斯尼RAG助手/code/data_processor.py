"""
数据处理层模块
Step1: 文档处理和图像处理
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
import pytesseract
from PIL import Image
import docx
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from .config import config
from .utils import table_to_markdown, chunk_text, clean_text, get_file_hash, get_timestamp


@dataclass
class TextChunk:
    """文本块数据类"""
    text: str
    source: str
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.chunk_id is None:
            self.chunk_id = f"{self.source}_{get_timestamp()}"


@dataclass
class ImageData:
    """图像数据类"""
    image_path: Path
    ocr_text: str = ""
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """文档处理器"""
    
    def __init__(self, documents_dir: Optional[Path] = None):
        """初始化文档处理器"""
        self.documents_dir = documents_dir or config.documents_dir
        self.chunks: List[TextChunk] = []
    
    def process_directory(self, directory: Optional[Path] = None) -> List[TextChunk]:
        """
        处理目录中的所有文档
        
        Args:
            directory: 文档目录，默认使用配置中的目录
        
        Returns:
            处理后的文本块列表
        """
        target_dir = directory or self.documents_dir
        
        if not target_dir.exists():
            logger.warning(f"文档目录不存在: {target_dir}")
            return []
        
        logger.info(f"开始处理文档目录: {target_dir}")
        
        # 支持的文件扩展名
        supported_extensions = {'.docx', '.txt', '.md'}
        
        # 查找所有支持的文档文件
        doc_files = [
            f for f in target_dir.rglob('*') 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        logger.info(f"找到 {len(doc_files)} 个文档文件")
        
        self.chunks = []
        for doc_file in doc_files:
            try:
                chunks = self.process_file(doc_file)
                self.chunks.extend(chunks)
                logger.info(f"处理完成: {doc_file.name}，生成 {len(chunks)} 个文本块")
            except Exception as e:
                logger.error(f"处理文件失败 {doc_file.name}: {e}")
        
        logger.info(f"文档处理完成，共生成 {len(self.chunks)} 个文本块")
        return self.chunks
    
    def process_file(self, file_path: Path) -> List[TextChunk]:
        """
        处理单个文档文件
        
        Args:
            file_path: 文件路径
        
        Returns:
            文本块列表
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.docx':
            return self._process_docx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._process_text_file(file_path)
        else:
            logger.warning(f"不支持的文件格式: {file_extension}")
            return []
    
    def _process_docx(self, file_path: Path) -> List[TextChunk]:
        """
        处理Word文档
        
        Args:
            file_path: Word文档路径
        
        Returns:
            文本块列表
        """
        chunks = []
        doc = Document(file_path)
        
        # 计算文件哈希用于去重
        file_hash = get_file_hash(file_path)
        
        # 提取段落和表格
        content_parts = []
        
        for element in doc.element.body:
            if element.tag.endswith('p'):  # 段落
                paragraph = Paragraph(element, doc)
                if paragraph.text.strip():
                    content_parts.append({
                        'type': 'paragraph',
                        'text': paragraph.text
                    })
            elif element.tag.endswith('tbl'):  # 表格
                table = Table(element, doc)
                table_data = self._extract_table_data(table)
                if table_data:
                    table_md = table_to_markdown(table_data)
                    content_parts.append({
                        'type': 'table',
                        'text': table_md
                    })
        
        # 将内容分割成块
        current_text = ""
        page_num = 0
        
        for part in content_parts:
            part_text = clean_text(part['text'])
            
            if not part_text:
                continue
            
            # 如果是表格，单独作为一个块
            if part['type'] == 'table':
                if current_text:
                    chunks.extend(self._create_text_chunks(
                        current_text, file_path, page_num, file_hash
                    ))
                    current_text = ""
                
                chunk = TextChunk(
                    text=part_text,
                    source=str(file_path),
                    page=page_num,
                    metadata={
                        'type': 'table',
                        'file_hash': file_hash,
                        'file_name': file_path.name
                    }
                )
                chunks.append(chunk)
            else:
                # 段落内容，累积到当前文本
                if current_text:
                    current_text += "\n\n"
                current_text += part_text
                
                # 如果累积的文本足够长，分割成块
                if len(current_text) > 1000:
                    chunked = self._create_text_chunks(
                        current_text, file_path, page_num, file_hash
                    )
                    chunks.extend(chunked)
                    current_text = ""
            
            page_num += 1
        
        # 处理剩余的文本
        if current_text:
            chunks.extend(self._create_text_chunks(
                current_text, file_path, page_num, file_hash
            ))
        
        return chunks
    
    def _process_text_file(self, file_path: Path) -> List[TextChunk]:
        """
        处理文本文件
        
        Args:
            file_path: 文本文件路径
        
        Returns:
            文本块列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        file_hash = get_file_hash(file_path)
        return self._create_text_chunks(
            text, file_path, 0, file_hash
        )
    
    def _extract_table_data(self, table: Table) -> List[List[str]]:
        """
        从表格中提取数据
        
        Args:
            table: docx表格对象
        
        Returns:
            表格数据（二维列表）
        """
        data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            data.append(row_data)
        return data
    
    def _create_text_chunks(
        self, 
        text: str, 
        source: Path, 
        page: int,
        file_hash: str
    ) -> List[TextChunk]:
        """
        创建文本块
        
        Args:
            text: 原始文本
            source: 来源文件路径
            page: 页码
            file_hash: 文件哈希
        
        Returns:
            文本块列表
        """
        chunked_texts = chunk_text(text, chunk_size=500, overlap=50)
        chunks = []
        
        for i, chunk_text in enumerate(chunked_texts):
            chunk = TextChunk(
                text=chunk_text,
                source=str(source),
                page=page,
                chunk_id=f"{file_hash}_{page}_{i}",
                metadata={
                    'type': 'paragraph',
                    'file_hash': file_hash,
                    'file_name': source.name,
                    'chunk_index': i
                }
            )
            chunks.append(chunk)
        
        return chunks


class ImageProcessor:
    """图像处理器"""
    
    def __init__(self, images_dir: Optional[Path] = None):
        """初始化图像处理器"""
        self.images_dir = images_dir or config.images_dir
        self.images: List[ImageData] = []
    
    def process_directory(self, directory: Optional[Path] = None) -> List[ImageData]:
        """
        处理目录中的所有图像
        
        Args:
            directory: 图像目录，默认使用配置中的目录
        
        Returns:
            处理后的图像数据列表
        """
        target_dir = directory or self.images_dir
        
        if not target_dir.exists():
            logger.warning(f"图像目录不存在: {target_dir}")
            return []
        
        logger.info(f"开始处理图像目录: {target_dir}")
        
        # 支持的图像扩展名
        supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        # 查找所有图像文件
        image_files = [
            f for f in target_dir.rglob('*') 
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        logger.info(f"找到 {len(image_files)} 个图像文件")
        
        self.images = []
        for image_file in image_files:
            try:
                image_data = self.process_image(image_file)
                self.images.append(image_data)
                logger.info(f"处理完成: {image_file.name}")
            except Exception as e:
                logger.error(f"处理图像失败 {image_file.name}: {e}")
        
        logger.info(f"图像处理完成，共处理 {len(self.images)} 个图像")
        return self.images
    
    def process_image(self, image_path: Path) -> ImageData:
        """
        处理单个图像文件
        
        Args:
            image_path: 图像路径
        
        Returns:
            图像数据对象
        """
        # OCR提取文本
        ocr_text = self._extract_text_with_ocr(image_path)
        
        image_data = ImageData(
            image_path=image_path,
            ocr_text=ocr_text,
            metadata={
                'file_name': image_path.name,
                'file_size': image_path.stat().st_size,
                'file_hash': get_file_hash(image_path)
            }
        )
        
        return image_data
    
    def _extract_text_with_ocr(self, image_path: Path) -> str:
        """
        使用Tesseract OCR提取图像中的文本
        
        Args:
            image_path: 图像路径
        
        Returns:
            提取的文本
        """
        try:
            image = Image.open(image_path)
            
            # 使用中英文识别
            text = pytesseract.image_to_string(
                image,
                lang=config.ocr_language,
                config=config.tesseract_config
            )
            
            return clean_text(text)
        except Exception as e:
            logger.error(f"OCR识别失败 {image_path.name}: {e}")
            return ""
    
    def extract_text_from_image_base64(self, image_base64: str) -> str:
        """
        从base64编码的图像中提取文本
        
        Args:
            image_base64: base64编码的图像数据
        
        Returns:
            提取的文本
        """
        try:
            import base64
            from io import BytesIO
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_data))
            
            text = pytesseract.image_to_string(
                image,
                lang=config.ocr_language,
                config=config.tesseract_config
            )
            
            return clean_text(text)
        except Exception as e:
            logger.error(f"Base64图像OCR识别失败: {e}")
            return ""


def extract_images_from_docx(docx_path: Path, output_dir: Path) -> List[Path]:
    """
    从Word文档中提取图像
    
    Args:
        docx_path: Word文档路径
        output_dir: 输出目录
    
    Returns:
        提取的图像路径列表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_images = []
    
    try:
        doc = Document(docx_path)
        image_index = 0
        
        # 遍历文档中的所有关系
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    
                    # 确定图像格式
                    image = Image.open(BytesIO(image_data))
                    ext = image.format.lower()
                    
                    # 保存图像
                    output_path = output_dir / f"{docx_path.stem}_image_{image_index}.{ext}"
                    with open(output_path, "wb") as f:
                        f.write(image_data)
                    
                    extracted_images.append(output_path)
                    image_index += 1
                    
                except Exception as e:
                    logger.warning(f"提取图像失败: {e}")
                    continue
        
        logger.info(f"从 {docx_path.name} 提取了 {len(extracted_images)} 个图像")
        
    except Exception as e:
        logger.error(f"从文档提取图像失败: {e}")
    
    return extracted_images


if __name__ == "__main__":
    # 测试代码
    from loguru import logger
    
    logger.add("logs/data_processor.log", rotation="1 day")
    
    # 测试文档处理
    doc_processor = DocumentProcessor()
    chunks = doc_processor.process_directory()
    logger.info(f"处理了 {len(chunks)} 个文本块")
    
    # 测试图像处理
    img_processor = ImageProcessor()
    images = img_processor.process_directory()
    logger.info(f"处理了 {len(images)} 个图像")