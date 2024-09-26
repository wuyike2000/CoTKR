#!/bin/sh

# Build the index for the general knowledge base using pyserini.

Freebase="../../Freebase/processed"

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ../../Freebase/processed/document \
  --index index \
  --generator DefaultLuceneDocumentGenerator \
  --threads 10 \
  --storePositions --storeDocvectors --storeRaw
