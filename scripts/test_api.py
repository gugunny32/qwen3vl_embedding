#!/usr/bin/env python3
"""
API testing script.
Tests the multimodal RAG API endpoints with Thai PDF documents.
"""

import sys
import os
import time
import requests
import argparse
from pathlib import Path

# Base URL
BASE_URL = "http://localhost:8000"


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_health():
    """Test health check endpoint"""
    print_section("Testing Health Check")

    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    return response.status_code == 200


def test_status():
    """Test status endpoint"""
    print_section("Testing Status")

    response = requests.get(f"{BASE_URL}/status")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Service: {data['service']}")
        print(f"Version: {data['version']}")
        print(f"CUDA Available: {data['cuda_available']}")
        print(f"GPU Count: {data['gpu_count']}")
        if data['gpu_name']:
            print(f"GPU Name: {data['gpu_name']}")
        print(f"\nSettings:")
        for key, value in data['settings'].items():
            print(f"  {key}: {value}")

    return response.status_code == 200


def test_upload(pdf_path):
    """Test document upload"""
    print_section(f"Testing Document Upload: {pdf_path}")

    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return None

    with open(pdf_path, 'rb') as f:
        files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
        data = {'title': os.path.basename(pdf_path)}

        response = requests.post(
            f"{BASE_URL}/api/v1/documents/upload",
            files=files,
            data=data
        )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Document ID: {result['document_id']}")
        print(f"Filename: {result['filename']}")
        print(f"Status: {result['status']}")
        print(f"Message: {result['message']}")

        # Wait for processing
        print("\nWaiting for document processing (30 seconds)...")
        time.sleep(30)

        return result['document_id']
    else:
        print(f"Error: {response.text}")
        return None


def test_list_documents():
    """Test listing documents"""
    print_section("Testing List Documents")

    response = requests.get(f"{BASE_URL}/api/v1/documents/")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        documents = response.json()
        print(f"Total Documents: {len(documents)}")

        for doc in documents[:5]:  # Show first 5
            print(f"\n  ID: {doc['id']}")
            print(f"  Filename: {doc['filename']}")
            print(f"  Total Chunks: {doc['total_chunks']}")
            print(f"  Created: {doc['created_at']}")

        return documents
    else:
        print(f"Error: {response.text}")
        return []


def test_search(query, use_reranker=False):
    """Test search endpoint"""
    print_section(f"Testing Search: '{query}'")

    payload = {
        "query": query,
        "top_k": 5,
        "use_reranker": use_reranker
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/search/hybrid",
        json=payload
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"Query: {result['query']}")
        print(f"Search Type: {result['search_type']}")
        print(f"Number of Results: {result['num_results']}")

        print("\nTop Results:")
        for i, res in enumerate(result['results'][:3], 1):
            print(f"\n  Result {i}:")
            print(f"  Document ID: {res['document_id'][:16]}...")
            print(f"  Page: {res['page_number']}")
            print(f"  Hybrid Score: {res['hybrid_score']:.4f}")
            if res['rerank_score']:
                print(f"  Rerank Score: {res['rerank_score']:.4f}")
            print(f"  Content: {res['content'][:200]}...")

        return result['results']
    else:
        print(f"Error: {response.text}")
        return []


def test_rag_query(question):
    """Test RAG query endpoint"""
    print_section(f"Testing RAG Query: '{question}'")

    payload = {
        "question": question,
        "top_k": 5,
        "use_reranker": True,
        "temperature": 0.7,
        "max_tokens": 512
    }

    response = requests.post(
        f"{BASE_URL}/api/v1/rag/query",
        json=payload
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nNumber of Sources: {result['num_sources']}")

        print("\nSource Documents:")
        for i, source in enumerate(result['source_documents'][:3], 1):
            print(f"\n  Source {i}:")
            print(f"  Document: {source['document_id'][:16]}...")
            print(f"  Page: {source['page_number']}")
            print(f"  Score: {source['score']:.4f}")
            print(f"  Preview: {source['content_preview'][:150]}...")

        return result
    else:
        print(f"Error: {response.text}")
        return None


def run_full_test_suite():
    """Run full test suite"""
    print_section("MULTIMODAL RAG API - FULL TEST SUITE")

    results = {
        'health': False,
        'status': False,
        'upload': False,
        'list': False,
        'search': False,
        'rag': False
    }

    # Test health
    results['health'] = test_health()
    if not results['health']:
        print("\n❌ Health check failed! Aborting tests.")
        return

    # Test status
    results['status'] = test_status()

    # Find test PDFs
    test_pdf_dir = Path(__file__).parent.parent / "test_file" / "pdf"
    pdf_files = list(test_pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"\n⚠️  No PDF files found in {test_pdf_dir}")
        print("Skipping upload test.")
    else:
        # Test upload with first PDF
        doc_id = test_upload(str(pdf_files[0]))
        results['upload'] = doc_id is not None

    # Test list documents
    docs = test_list_documents()
    results['list'] = len(docs) > 0

    # Test search
    search_queries = [
        "ระบบจัดซื้อ",
        "FAQ",
        "ขั้นตอน"
    ]

    for query in search_queries[:1]:  # Test with first query
        search_results = test_search(query, use_reranker=True)
        results['search'] = len(search_results) > 0
        if results['search']:
            break

    # Test RAG
    rag_questions = [
        "อธิบายขั้นตอนการจัดซื้อในระบบ",
        "FAQ คืออะไร",
    ]

    for question in rag_questions[:1]:  # Test with first question
        rag_result = test_rag_query(question)
        results['rag'] = rag_result is not None
        if results['rag']:
            break

    # Summary
    print_section("TEST SUMMARY")
    total = len(results)
    passed = sum(1 for v in results.values() if v)

    print(f"Tests Passed: {passed}/{total}\n")
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test.upper()}: {status}")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")


def main():
    parser = argparse.ArgumentParser(description="Test Multimodal RAG API")
    parser.add_argument(
        "--test",
        choices=["all", "health", "status", "upload", "list", "search", "rag"],
        default="all",
        help="Which test to run"
    )
    parser.add_argument("--pdf", help="Path to PDF file for upload test")
    parser.add_argument("--query", help="Query for search test")
    parser.add_argument("--question", help="Question for RAG test")

    args = parser.parse_args()

    if args.test == "all":
        run_full_test_suite()
    elif args.test == "health":
        test_health()
    elif args.test == "status":
        test_status()
    elif args.test == "upload":
        if not args.pdf:
            print("Error: --pdf required for upload test")
            sys.exit(1)
        test_upload(args.pdf)
    elif args.test == "list":
        test_list_documents()
    elif args.test == "search":
        query = args.query or "ระบบจัดซื้อ"
        test_search(query, use_reranker=True)
    elif args.test == "rag":
        question = args.question or "อธิบายขั้นตอนการจัดซื้อในระบบ"
        test_rag_query(question)


if __name__ == "__main__":
    main()
