# CLAUDE.md - AI Agent Guidelines for RNZ Audiobook Archiver

## Build/Test Commands
- Run the application: `python main.py`
- Run with custom options: `python main.py --output-dir ./my_archive --max-workers 3`
- Type check: `python -m mypy main.py` (requires mypy installation)
- Lint: `python -m ruff check main.py` (requires ruff installation)

## Code Style Guidelines
- **Python Version**: >=3.13
- **Imports**: Group imports: stdlib first, then third-party, then local. Sort alphabetically.
- **Docstrings**: Every function/class needs docstrings with Args and Returns sections.
- **Error Handling**: Use try/except with specific exceptions. Log errors with proper context.
- **Naming**: snake_case for variables/functions, PascalCase for classes, UPPER_CASE for constants.
- **Type Hints**: Add type hints to function parameters and return values.
- **Concurrency**: Use ThreadPoolExecutor for I/O-bound operations.
- **Logging**: Use the existing logger with appropriate level (info, warning, error).
- **Path Handling**: Use pathlib.Path instead of os.path for file operations.
