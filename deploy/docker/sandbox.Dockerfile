# syntax=docker/dockerfile:1.7
# The image run_sandbox runs commands in.
FROM debian:bookworm-slim

# python3 with HTML, table, HTTP and document libraries, OCR, and common shell tools.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        python3 \
        python3-bs4 \
        python3-lxml \
        python3-requests \
        python3-pandas \
        python3-dateutil \
        python3-yaml \
        python3-tabulate \
        python3-html2text \
        python3-pypdf \
        python3-docx \
        python3-openpyxl \
        python3-xlrd \
        python3-numpy \
        python3-pil \
        python3-chardet \
        python3-markdown \
        tesseract-ocr \
        tesseract-ocr-eng \
        ca-certificates \
        curl \
        wget \
        jq \
        gawk \
        bc \
        coreutils \
        findutils \
        grep \
        sed \
        ripgrep \
        sqlite3 \
        poppler-utils \
        file \
        unzip \
        zip \
        less \
        tree \
    && rm -rf /var/lib/apt/lists/*

# Commands run as nobody against a read-only root, with the workspace mounted at /tmp.
USER 65534:65534
WORKDIR /tmp
ENV HOME=/tmp
ENTRYPOINT []
CMD ["sh"]
