# EVM Hacker Bench - Docker Container
# Based on SCONE-bench evaluation framework requirements
# Provides sandboxed execution environment with Foundry + Slither

FROM python:3.11-slim-bookworm

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FOUNDRY_DIR=/root/.foundry
ENV PATH="${FOUNDRY_DIR}/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Foundry (forge, cast, anvil)
RUN curl -L https://foundry.paradigm.xyz | bash && \
    ${FOUNDRY_DIR}/bin/foundryup

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install solc-select and Slither
RUN pip install --no-cache-dir solc-select slither-analyzer

# Install common Solidity versions
RUN solc-select install 0.8.19 && \
    solc-select install 0.8.17 && \
    solc-select install 0.7.6 && \
    solc-select install 0.6.12 && \
    solc-select use 0.8.19

# Copy project files
COPY . .

# Create workspace directory
RUN mkdir -p /workdir

# Set working directory
WORKDIR /workdir

# Default command
CMD ["python", "-m", "evm_hacker_bench.run_hacker_bench"]

