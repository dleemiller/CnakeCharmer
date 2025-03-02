# CnakeCharmer - Python/Cython Code Generator

CnakeCharmer is a service that generates, analyzes, and optimizes both Python and Cython code from natural language prompts. It leverages large language models (Claude 3.7 Sonnet via OpenRouter) to produce efficient, equivalent implementations in both languages.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **AI-Powered Code Generation**: Generate Python and Cython code from natural language descriptions
- **Equivalency Checking**: Verify that Python and Cython implementations produce the same results
- **Performance Analysis**: Analyze and optimize Cython code for maximum performance
- **Distributed Task Processing**: Handle code generation and analysis jobs via Celery
- **Comprehensive Monitoring**: Track tasks, database, and message queues through dedicated UIs

## 📋 Table of Contents

- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Monitoring Tools](#-monitoring-tools)
- [Development](#-development)
- [Administration Commands](#-administration-commands)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🔧 Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/)
- [OpenRouter](https://openrouter.ai/) API key for Claude 3.7 Sonnet access

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/cnake-charmer.git
   cd cnake-charmer
   ```

2. **Create environment variables file**:
   Create a `.env` file in the project root:
   ```bash
   echo "OPENROUTER_API_KEY=your_openrouter_api_key_here" > .env
   ```

3. **Build and start services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify installation**:
   ```bash
   curl http://localhost:8000/health
   ```
   You should receive a JSON response with status `"healthy"`.

## 📁 Project Structure

```
cnake_charmer/
├── generate/                  # Core code generation modules
│   ├── code_generator.py      # AI-based code generation
│   ├── database.py            # Database operations
│   ├── equivalency_checker.py # Verification of code equivalence
│   ├── ephemeral_runner/      # Code execution environment
│   │   ├── builders.py        # Language-specific builders
│   │   └── core.py            # Core execution logic
│   ├── fastapi_service/       # API service
│   │   ├── main.py            # FastAPI initialization
│   │   ├── routes.py          # API endpoints
│   │   └── tasks.py           # Celery task definitions
│   └── worker/                # Celery worker configuration
├── docker-compose.yml         # Service definitions
├── Dockerfile                 # Container build instructions
└── requirements.txt           # Python dependencies
```

## 🔍 Usage

### Generating Code

You can generate code by making a POST request to the API:

```bash
curl -X POST "http://localhost:8000/generate?prompt=Write%20an%20efficient%20Fibonacci%20function%20in%20Cython"
```

This will return a task ID that you can use to check the status:

```json
{
  "task_id": "38d56bc2-b5bd-441a-ab4b-b5c41a712389"
}
```

### Checking Results

You can get the generated code by making a GET request with the original prompt:

```bash
curl "http://localhost:8000/results/Write%20an%20efficient%20Fibonacci%20function%20in%20Cython"
```

This will return the generated Python and Cython code:

```json
{
  "prompt_id": "Write an efficient Fibonacci function in Cython",
  "python_code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "cython_code": "# cython: boundscheck=False\n# cython: wraparound=False\n\ndef fibonacci(int n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "build_status": "generated",
  "created_at": "2025-03-02T17:59:31.350Z"
}
```

## 📚 API Documentation

The API is documented with OpenAPI and accessible at [http://localhost:8000/docs](http://localhost:8000/docs)

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate Python and Cython code from a prompt |
| `/results/{prompt}` | GET | Get results for a specific prompt |
| `/health` | GET | Check service health |

For full API details, see the [API Documentation](http://localhost:8000/docs).

## 🖥️ Monitoring Tools

CnakeCharmer includes several monitoring tools to help you manage and debug your services.

### Flower (Celery Monitoring)

[Flower](https://flower.readthedocs.io/) provides real-time monitoring of Celery tasks and workers.

- **Access URL**: [http://localhost:5555](http://localhost:5555)
- **Key Features**:
  - Task monitoring and management
  - Worker status tracking
  - Performance statistics
  - Task history and filtering

![Flower Dashboard](https://flower.readthedocs.io/en/latest/_images/dashboard.png)

#### Using Flower

- **View Tasks**: Navigate to [http://localhost:5555/tasks](http://localhost:5555/tasks)
- **Monitor Workers**: Check [http://localhost:5555/workers](http://localhost:5555/workers)
- **Revoke a Task**: Go to the Tasks tab, find the task and click "Revoke"
- **Filter Tasks**: Use the search bar to filter by task name or state

### PgAdmin (PostgreSQL Management)

[PgAdmin](https://www.pgadmin.org/) is a comprehensive PostgreSQL database management tool.

- **Access URL**: [http://localhost:5050](http://localhost:5050)
- **Login**:
  - Email: `admin@example.com`
  - Password: `admin`

#### Setting Up PgAdmin

1. After logging in, right-click on "Servers" and select "Create" → "Server"
2. In the "General" tab, name it "CnakeCharmer"
3. In the "Connection" tab, enter:
   - Host: `db`
   - Port: `5432`
   - Database: `cnake_charmer`
   - Username: `user`
   - Password: `password`
4. Click "Save"

#### Useful SQL Queries

View recent code generations:
```sql
SELECT 
    prompt_id, 
    build_status, 
    created_at,
    substring(python_code, 1, 50) as python_snippet
FROM 
    generated_code 
ORDER BY 
    created_at DESC 
LIMIT 10;
```

Find failed generations:
```sql
SELECT 
    prompt_id, 
    created_at
FROM 
    generated_code 
WHERE 
    python_code LIKE '%Error%'
ORDER BY 
    created_at DESC;
```

### Redis Commander

[Redis Commander](https://github.com/joeferner/redis-commander) is a web management tool for Redis.

- **Access URL**: [http://localhost:8081](http://localhost:8081)
- **Key Features**:
  - Browse Redis keys and data
  - View Celery queues
  - Execute Redis commands

#### Using Redis Commander

- **View Celery Tasks**: Look for keys starting with `celery-task-meta-`
- **Examine Queues**: Check `celery` keys to see pending tasks
- **Execute Commands**: Use the built-in Redis CLI to run commands

## 🛠️ Development

### Project Setup for Development

1. **Create a development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

2. **Run service with hot reloading**:
   ```bash
   docker-compose up -d
   ```

### Making Code Changes

The Docker setup mounts the local directory, so any changes to Python files will be immediately reflected in the running container with hot reloading.

### Testing

Run tests with:
```bash
pytest tests/
```

## 🔐 Administration Commands

### Celery Administration

CnakeCharmer uses [Celery](https://docs.celeryq.dev/) for task processing. Here are common commands for managing Celery:

#### Worker Management

```bash
# Check worker status
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker status

# Start a worker with debug logging
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker worker --loglevel=DEBUG

# Restart all workers
docker-compose restart worker

# Scale to multiple workers
docker-compose up -d --scale worker=3
```

#### Task Management

```bash
# View active tasks
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect active

# View queued tasks
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect reserved

# List registered tasks
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect registered

# Purge all waiting tasks
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker purge

# Revoke a specific task
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker revoke <task_id>

# Terminate a running task
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker revoke <task_id> --terminate
```

#### Inspection and Monitoring

```bash
# Get worker statistics
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect stats

# Show worker memory usage
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect memsample

# View scheduler ETA tasks
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect scheduled

# Check worker queue configuration
docker-compose exec worker celery -A cnake_charmer.generate.worker.worker inspect conf
```

### Database Administration

```bash
# Connect to database directly
docker-compose exec db psql -U user -d cnake_charmer

# Create a database backup
docker-compose exec db pg_dump -U user cnake_charmer > backup.sql

# Restore from backup
cat backup.sql | docker-compose exec -T db psql -U user -d cnake_charmer

# Run a SQL query
docker-compose exec db psql -U user -d cnake_charmer -c "SELECT count(*) FROM generated_code;"
```

### Redis Administration

```bash
# Connect to Redis CLI
docker-compose exec redis redis-cli

# View all keys
docker-compose exec redis redis-cli KEYS "*"

# Monitor Redis commands in real-time
docker-compose exec redis redis-cli MONITOR

# Clear all data
docker-compose exec redis redis-cli FLUSHALL

# Get information about server
docker-compose exec redis redis-cli INFO
```

### Docker Administration

```bash
# View all running containers
docker-compose ps

# View service logs
docker-compose logs

# Follow logs from a specific service
docker-compose logs -f worker

# Restart all services
docker-compose restart

# Stop all services
docker-compose down

# Update and rebuild services
docker-compose build --no-cache
docker-compose up -d
```

## 🐛 Troubleshooting

### Common Issues

#### Authentication Error with OpenRouter

If you see an error like:
```
litellm.AuthenticationError: AuthenticationError: OpenrouterException - {"error":{"message":"No auth credentials found","code":401}}
```

**Solution**: 
1. Ensure your `.env` file contains the correct `OPENROUTER_API_KEY`
2. Rebuild the container to ensure environment variables are updated:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

#### Database Table Not Found

If you see an error like:
```
relation "generated_code" does not exist
```

**Solution**:
1. Connect to the database and create the table:
   ```bash
   docker-compose exec db psql -U user -d cnake_charmer
   ```
   
   ```sql
   CREATE TABLE generated_code (
       id SERIAL PRIMARY KEY,
       prompt_id TEXT NOT NULL,
       python_code TEXT,
       cython_code TEXT,
       build_status TEXT DEFAULT 'pending',
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

#### Celery Worker Isn't Processing Tasks

**Solution**:
1. Check if the worker is running:
   ```bash
   docker-compose ps worker
   ```
2. Restart the worker:
   ```bash
   docker-compose restart worker
   ```
3. Check the logs for errors:
   ```bash
   docker-compose logs worker
   ```

### Getting Help

If you encounter issues not covered here, try:

1. Checking the logs for all services:
   ```bash
   docker-compose logs
   ```
2. Inspecting specific service logs:
   ```bash
   docker-compose logs fastapi
   ```
3. Opening an issue on the GitHub repository

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Celery Documentation](https://docs.celeryq.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Cython Documentation](https://cython.org/)
- [OpenRouter API](https://openrouter.ai/docs)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Redis Documentation](https://redis.io/documentation)