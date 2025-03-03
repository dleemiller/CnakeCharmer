# CnakeCharmer

CnakeCharmer is a service that generates, analyzes, and optimizes code in multiple languages, with a primary focus on Python and Cython equivalency.

## Features

- **AI-Powered Code Generation**: Generate Python and Cython code from natural language descriptions
- **Equivalency Checking**: Verify that Python and Cython implementations produce the same results
- **Performance Analysis**: Analyze and optimize Cython code for maximum performance
- **Distributed Task Processing**: Handle code generation and analysis jobs via Celery
- **Comprehensive Monitoring**: Track tasks, database, and message queues through dedicated UIs

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Docker Setup](#docker-setup)
  - [Manual Setup](#manual-setup)
- [Architecture](#architecture)
- [API Reference](#api-reference)
- [Development](#development)
  - [Project Structure](#project-structure)
  - [Adding New Analyzers](#adding-new-analyzers)
  - [Adding New Builders](#adding-new-builders)
- [Monitoring](#monitoring)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Docker and Docker Compose (for containerized setup)
- Python 3.9+ (for manual setup)
- PostgreSQL 13+ (for manual setup)
- Redis 6+ (for manual setup)

### Docker Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cnake-charmer.git
   cd cnake-charmer