# RNZ Audiobook Archiver (rnzdl)

A Python tool that archives audiobooks from Radio New Zealand's Storytime website, preserving them with full metadata.


## Features

- 📚 Downloads all audiobooks from RNZ Storytime categorized by age group
- 🖼️ Preserves cover artwork for each book
- 🎵 Enriches MP3 files with complete metadata (title, author, album art, etc.)
- 📁 Organizes files by reading age category
- 🔄 Supports graceful interruption with resume capability
- 🚀 Uses multi-threading for efficient downloading

## Installation

### Prerequisites
- Python 3.13 or newer

### Using uv (recommended)

```bash
uv sync
```

### Using standard pip

```bash
pip install -e .
```

## Usage

### Basic Usage

```bash
uv run main.py
```

This will download all audiobooks to the default directory `./rnz_storytime_archive`.

### Advanced Options

```bash
uv run main.py --output-dir /path/to/my/library --max-workers 8 --timeout 60
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--base-url` | Base URL of the RNZ Storytime website | https://storytime.rnz.co.nz |
| `--output-dir` | Directory to save downloaded files | ./rnz_storytime_archive |
| `--max-retries` | Maximum number of retries for failed downloads | 3 |
| `--timeout` | Timeout for HTTP requests in seconds | 30 |
| `--max-workers` | Maximum number of concurrent downloads | 5 |

## Output Structure

The archiver organizes files by reading age and book title:

```
rnz_storytime_archive/
├── Kids/
│   ├── Book Title 1/
│   │   ├── Book Title 1.mp3
│   │   └── Book Title 1_cover.jpg
│   └── Book Title 2/
│       ├── Book Title 2_E01.mp3
│       ├── Book Title 2_E02.mp3
│       └── Book Title 2_cover.jpg
├── Little Kids/
│   └── ...
└── Young Adult/
    └── ...
```

## MP3 Metadata

The archiver enriches each MP3 file with:
- **Title**: Episode or clip title
- **Artist**: Book author
- **Album**: Book title
- **Genre**: Reading age category
- **Cover Art**: Book cover image
- **Comment**: Book or episode synopsis
- **Track Number**: For multi-episode books


## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Radio New Zealand for providing these audiobooks
- All the authors, narrators, and producers who create this content

## Disclaimer

This tool is meant for personal archiving and preservation. Please respect copyright and usage restrictions for the downloaded content.
