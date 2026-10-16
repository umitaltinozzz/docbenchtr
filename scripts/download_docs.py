from docbenchtr.documents import download_documents


if __name__ == "__main__":
    for record in download_documents():
        print(f"{record['document_id']}  sha256={record['sha256']}  {record['source_url']}")
