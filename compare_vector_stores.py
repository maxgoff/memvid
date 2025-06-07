#!/usr/bin/env python3
"""
compare_vector_stores.py - Compare memvid with FAISS for vector storage

This script:
1. Ingests documents into both memvid (QR video) and standalone FAISS
2. Measures and compares creation times
3. Benchmarks search performance
4. Queries LLM with contexts from both systems
5. Provides detailed comparison statistics

Usage:
    python compare_vector_stores.py --input-dir /path/to/documents --provider google
    python compare_vector_stores.py --files file1.txt file2.pdf --provider openai
    python compare_vector_stores.py --test-queries "What is X?" "How does Y work?" --provider google
"""

import argparse
import os
import sys
import time
import json
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
import pickle

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memvid import MemvidEncoder, MemvidRetriever, MemvidChat
from memvid.config import get_default_config
from memvid.llm_client import LLMClient


class FAISSVectorStore:
    """Standalone FAISS vector store for comparison"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2', dimension: int = 384):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.dimension = dimension
        self.index = None
        self.chunks = []
        self.metadata = []
        
    def create_index(self, index_type: str = "Flat"):
        """Create FAISS index"""
        if index_type == "Flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif index_type == "IVF":
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = min(100, max(4, int(np.sqrt(len(self.chunks)))))
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
            
    def add_chunks(self, chunks: List[str], metadata: List[Dict[str, Any]] = None):
        """Add text chunks to the store"""
        self.chunks.extend(chunks)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"chunk_id": len(self.chunks) + i} for i in range(len(chunks))])
            
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            chunks,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Add to index
        if self.index is None:
            self.create_index()
            
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            # Train IVF index if needed
            if len(embeddings) >= self.index.nlist:
                self.index.train(embeddings.astype('float32'))
                
        self.index.add(embeddings.astype('float32'))
        
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar chunks"""
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((
                    self.chunks[idx],
                    float(dist),
                    self.metadata[idx] if idx < len(self.metadata) else {}
                ))
                
        return results
        
    def save(self, path: str):
        """Save the vector store"""
        base_path = Path(path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save index
        faiss.write_index(self.index, str(base_path.with_suffix('.faiss')))
        
        # Save chunks and metadata
        data = {
            'chunks': self.chunks,
            'metadata': self.metadata,
            'dimension': self.dimension
        }
        
        with open(base_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(data, f)
            
    def load(self, path: str):
        """Load the vector store"""
        base_path = Path(path)
        
        # Load index
        self.index = faiss.read_index(str(base_path.with_suffix('.faiss')))
        
        # Load chunks and metadata
        with open(base_path.with_suffix('.pkl'), 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']
            self.dimension = data['dimension']


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 16) -> List[str]:
    """Chunk text with overlap"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
            
        start = end - overlap
        
    return chunks


def process_files(files: List[str], chunk_size: int = 1024, overlap: int = 16) -> List[Tuple[str, Dict[str, Any]]]:
    """Process files and return chunks with metadata"""
    all_chunks = []
    
    for file_path in files:
        file_path = Path(file_path)
        print(f"Processing: {file_path.name}")
        
        try:
            content = None
            
            # Read file content based on type
            if file_path.suffix.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
            elif file_path.suffix.lower() == '.pdf':
                try:
                    import fitz  # PyMuPDF
                    doc = fitz.open(file_path)
                    content = ""
                    for page in doc:
                        content += page.get_text()
                    doc.close()
                except ImportError:
                    try:
                        from pypdf import PdfReader
                        reader = PdfReader(file_path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text()
                    except ImportError:
                        print(f"Warning: No PDF library available. Install PyMuPDF or pypdf to process PDFs.")
                        continue
                        
            elif file_path.suffix.lower() in ['.html', '.htm']:
                try:
                    from bs4 import BeautifulSoup
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        soup = BeautifulSoup(f.read(), 'html.parser')
                        for script in soup(["script", "style"]):
                            script.decompose()
                        text = soup.get_text()
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        content = ' '.join(chunk for chunk in chunks if chunk)
                except ImportError:
                    print(f"Warning: BeautifulSoup not available for HTML processing.")
                    continue
                    
            else:
                print(f"Skipping unsupported format: {file_path.suffix}")
                continue
                
            if not content or not content.strip():
                print(f"Warning: No content extracted from {file_path.name}")
                continue
                
            # Chunk the content
            chunks = chunk_text(content, chunk_size, overlap)
            
            # Add metadata
            for i, chunk in enumerate(chunks):
                metadata = {
                    'file': str(file_path),
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
                all_chunks.append((chunk, metadata))
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            
    return all_chunks


def create_comparison_report(stats: Dict[str, Any], output_dir: Path):
    """Create a detailed comparison report"""
    report_path = output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_path, 'w') as f:
        f.write("# Vector Store Comparison Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Creation times
        f.write("## Artifact Creation Performance\n\n")
        f.write(f"- **Memvid creation time**: {stats['memvid_creation_time']:.2f} seconds\n")
        f.write(f"- **FAISS creation time**: {stats['faiss_creation_time']:.2f} seconds\n")
        f.write(f"- **Speed difference**: {stats['creation_speed_ratio']:.2f}x\n\n")
        
        # Storage sizes
        f.write("## Storage Comparison\n\n")
        f.write(f"- **Memvid video size**: {stats['memvid_size_mb']:.2f} MB\n")
        f.write(f"- **FAISS index size**: {stats['faiss_size_mb']:.2f} MB\n")
        f.write(f"- **Size ratio**: {stats['size_ratio']:.2f}x\n\n")
        
        # Search performance
        f.write("## Search Performance\n\n")
        f.write("| Query | Memvid Time (s) | FAISS Time (s) | Speed Ratio |\n")
        f.write("|-------|-----------------|----------------|-------------|\n")
        
        for query_stat in stats['search_stats']:
            f.write(f"| {query_stat['query'][:30]}... | "
                   f"{query_stat['memvid_time']:.3f} | "
                   f"{query_stat['faiss_time']:.3f} | "
                   f"{query_stat['speed_ratio']:.2f}x |\n")
                   
        f.write(f"\n**Average search times:**\n")
        f.write(f"- Memvid: {stats['avg_memvid_search_time']:.3f}s\n")
        f.write(f"- FAISS: {stats['avg_faiss_search_time']:.3f}s\n\n")
        
        # Result comparison
        if 'result_comparison' in stats:
            f.write("## Result Quality Comparison\n\n")
            for comp in stats['result_comparison']:
                f.write(f"### Query: {comp['query']}\n\n")
                f.write(f"**Overlap in top-5 results**: {comp['overlap_ratio']*100:.1f}%\n\n")
                
        # LLM comparison
        if 'llm_comparison' in stats:
            f.write("## LLM Response Comparison\n\n")
            for i, comp in enumerate(stats['llm_comparison']):
                f.write(f"### Query {i+1}: {comp['query']}\n\n")
                
                f.write("#### Memvid Response:\n")
                f.write("```\n")
                f.write(comp['memvid_response'])
                f.write("\n```\n\n")
                
                f.write("#### FAISS Response:\n")
                f.write("```\n")
                f.write(comp['faiss_response'])
                f.write("\n```\n\n")
                
                f.write("---\n\n")
                
    print(f"\n📊 Detailed report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare memvid with FAISS vector storage",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-dir', help='Directory containing documents')
    input_group.add_argument('--files', nargs='+', help='Specific files to process')
    
    # Test options
    parser.add_argument('--test-queries', nargs='+', 
                       default=["What is the main topic?", "Summarize the key points"],
                       help='Queries to test search performance')
    
    # LLM options
    parser.add_argument('--provider', choices=['openai', 'google', 'anthropic'],
                       default='google', help='LLM provider for chat comparison')
    parser.add_argument('--model', help='Specific model to use')
    
    # Processing options
    parser.add_argument('--chunk-size', type=int, default=1024,
                       help='Chunk size for text splitting')
    parser.add_argument('--overlap', type=int, default=16,
                       help='Overlap between chunks')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of results to retrieve')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Collect files
    if args.input_dir:
        input_path = Path(args.input_dir)
        files = []
        for ext in ['.txt', '.md', '.pdf']:
            files.extend(input_path.rglob(f"*{ext}"))
        files = [str(f) for f in files if f.is_file()]
    else:
        files = args.files
        
    if not files:
        print("No files to process!")
        return 1
        
    print(f"Found {len(files)} files to process")
    
    # Process files to get chunks
    print("\n📝 Processing files...")
    chunks_with_metadata = process_files(files, args.chunk_size, args.overlap)
    chunks = [chunk for chunk, _ in chunks_with_metadata]
    print(f"Created {len(chunks)} chunks")
    
    if not chunks:
        print("\n❌ No chunks were created. Please check:")
        print("  - File formats are supported (.txt, .md, .pdf, .html)")
        print("  - Files contain extractable text")
        print("  - For PDFs: Install PyMuPDF (pip install PyMuPDF) or pypdf (pip install pypdf)")
        return 1
    
    stats = {
        'total_files': len(files),
        'total_chunks': len(chunks),
        'chunk_size': args.chunk_size,
        'overlap': args.overlap
    }
    
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Create Memvid artifact
    print("\n🎬 Creating Memvid artifact...")
    memvid_start = time.time()
    
    config = get_default_config()
    config["chunking"]["chunk_size"] = args.chunk_size
    config["chunking"]["overlap"] = args.overlap
    
    encoder = MemvidEncoder(config)
    
    # Add all chunks
    for chunk in chunks:
        encoder.chunks.append(chunk)
        
    # Build video
    video_path = output_dir / f"memvid_comparison_{timestamp}.mp4"
    index_path = output_dir / f"memvid_comparison_{timestamp}_index.json"
    
    try:
        build_stats = encoder.build_video(str(video_path), str(index_path))
        memvid_creation_time = time.time() - memvid_start
        stats['memvid_creation_time'] = memvid_creation_time
        stats['memvid_size_mb'] = video_path.stat().st_size / (1024 * 1024)
        print(f"✅ Memvid created in {memvid_creation_time:.2f}s")
    except Exception as e:
        print(f"❌ Memvid creation failed: {e}")
        return 1
        
    # 2. Create FAISS artifact
    print("\n🔍 Creating FAISS artifact...")
    faiss_start = time.time()
    
    faiss_store = FAISSVectorStore()
    faiss_store.add_chunks(chunks, [meta for _, meta in chunks_with_metadata])
    
    faiss_path = output_dir / f"faiss_comparison_{timestamp}"
    faiss_store.save(str(faiss_path))
    
    faiss_creation_time = time.time() - faiss_start
    stats['faiss_creation_time'] = faiss_creation_time
    stats['faiss_size_mb'] = (faiss_path.with_suffix('.faiss').stat().st_size + 
                             faiss_path.with_suffix('.pkl').stat().st_size) / (1024 * 1024)
    print(f"✅ FAISS created in {faiss_creation_time:.2f}s")
    
    # Calculate creation speed ratio
    stats['creation_speed_ratio'] = memvid_creation_time / faiss_creation_time
    stats['size_ratio'] = stats['memvid_size_mb'] / stats['faiss_size_mb']
    
    # 3. Test search performance
    print("\n⚡ Testing search performance...")
    
    # Initialize retrievers
    memvid_retriever = MemvidRetriever(str(video_path), str(index_path))
    
    search_stats = []
    result_comparison = []
    
    for query in args.test_queries:
        print(f"\nQuery: '{query}'")
        
        # Memvid search
        memvid_start = time.time()
        memvid_results = memvid_retriever.search(query, top_k=args.top_k)
        memvid_search_time = time.time() - memvid_start
        
        # FAISS search
        faiss_start = time.time()
        faiss_results = faiss_store.search(query, k=args.top_k)
        faiss_search_time = time.time() - faiss_start
        
        search_stats.append({
            'query': query,
            'memvid_time': memvid_search_time,
            'faiss_time': faiss_search_time,
            'speed_ratio': memvid_search_time / faiss_search_time
        })
        
        # Compare results
        # memvid_results is a list of strings
        memvid_texts = memvid_results
        faiss_texts = [r[0] for r in faiss_results]
        
        # Calculate overlap
        overlap = len(set(memvid_texts[:5]) & set(faiss_texts[:5]))
        
        result_comparison.append({
            'query': query,
            'overlap_ratio': overlap / 5.0,
            'memvid_results': memvid_texts[:3],
            'faiss_results': faiss_texts[:3]
        })
        
        print(f"  Memvid: {memvid_search_time:.3f}s")
        print(f"  FAISS: {faiss_search_time:.3f}s")
        print(f"  Result overlap: {overlap}/5")
        
    stats['search_stats'] = search_stats
    stats['result_comparison'] = result_comparison
    stats['avg_memvid_search_time'] = np.mean([s['memvid_time'] for s in search_stats])
    stats['avg_faiss_search_time'] = np.mean([s['faiss_time'] for s in search_stats])
    
    # 4. Test with LLM
    print("\n🤖 Testing LLM responses with both contexts...")
    
    # Initialize chat for memvid
    memvid_chat = MemvidChat(
        video_file=str(video_path),
        index_file=str(index_path),
        llm_provider=args.provider,
        llm_model=args.model
    )
    
    # Initialize LLM client for FAISS
    llm_client = LLMClient(
        provider=args.provider,
        model=args.model
    )
    
    llm_comparison = []
    
    for query in args.test_queries[:2]:  # Test first 2 queries with LLM
        print(f"\n💬 LLM Query: '{query}'")
        
        # Get Memvid response
        memvid_response = memvid_chat.chat(query, stream=False)
        
        # Get FAISS context and query LLM
        faiss_results = faiss_store.search(query, k=args.top_k)
        faiss_context = "\n---\n".join([r[0] for r in faiss_results])
        
        faiss_messages = [
            {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions."},
            {"role": "user", "content": f"Context:\n{faiss_context}\n\nQuestion: {query}"}
        ]
        
        faiss_response = llm_client.chat(faiss_messages, stream=False)
        
        llm_comparison.append({
            'query': query,
            'memvid_response': memvid_response,
            'faiss_response': faiss_response
        })
        
        print("✅ Responses generated")
        
    stats['llm_comparison'] = llm_comparison
    
    # 5. Generate report
    print("\n📊 Generating comparison report...")
    report_path = create_comparison_report(stats, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"\n📦 Artifact Creation:")
    print(f"  Memvid: {stats['memvid_creation_time']:.2f}s ({stats['memvid_size_mb']:.1f} MB)")
    print(f"  FAISS: {stats['faiss_creation_time']:.2f}s ({stats['faiss_size_mb']:.1f} MB)")
    print(f"  Creation speed: FAISS is {stats['creation_speed_ratio']:.1f}x faster")
    print(f"  Storage size: FAISS is {stats['size_ratio']:.1f}x smaller")
    
    print(f"\n⚡ Search Performance:")
    print(f"  Memvid avg: {stats['avg_memvid_search_time']:.3f}s")
    print(f"  FAISS avg: {stats['avg_faiss_search_time']:.3f}s")
    print(f"  Search speed: FAISS is {stats['avg_memvid_search_time']/stats['avg_faiss_search_time']:.1f}x faster")
    
    print(f"\n📄 Full report: {report_path}")
    
    # Save full stats
    stats_path = output_dir / f"comparison_stats_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
        
    return 0


if __name__ == "__main__":
    sys.exit(main())