#!/usr/bin/env python3
"""
RNZ Audiobook Archiver

A web archiver to download and organize audiobooks from RNZ Storytime.
It downloads MP3s and images, organizes them by reading age, and enriches the MP3 metadata.
"""

import os
import sys
import json
import time
import signal
import argparse
import requests
import threading
import re
import io
from pathlib import Path
from urllib.parse import urlparse
import concurrent.futures
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union

# For MP3 metadata manipulation
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, COMM, TCON, TRCK
from mutagen.mp3 import MP3
from mutagen.id3._util import ID3NoHeaderError
from PIL import Image

def setup_logging() -> logging.Logger:
    """Set up and configure logging for the application.
    
    Returns:
        A configured logger instance
    """
    # Ensure the file is opened with UTF-8 encoding
    log_file_handler = logging.FileHandler('rnz_archiver.log', encoding='utf-8')

    # Fix for Windows console encoding issues
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set the console output encoding to utf-8 if on Windows
    if sys.platform == 'win32':
        try_set_windows_utf8()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            log_file_handler,
            console_handler
        ]
    )
    return logging.getLogger('rnz_archiver')


def try_set_windows_utf8() -> None:
    """
    Try to set Windows console to UTF-8 mode.
    """
    # Check if running in a terminal that supports UTF-8
    try:
        # Try to set the console to UTF-8 mode (Windows 10+)
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # 65001 is the code page for UTF-8
        kernel32.SetConsoleCP(65001)  # 65001 is the code page for UTF-8
    except (AttributeError, ImportError):
        # Fallback if it doesn't work - may still have encoding issues
        pass


# Initialize the logger
logger = setup_logging()

class RNZArchiver:
    """Class to archive audiobooks from RNZ Storytime website."""

    def __init__(self, base_url: str, output_dir: Union[str, Path], 
                 max_retries: int = 3, timeout: int = 30, max_workers: int = 5):
        """Initialize the archiver.

        Args:
            base_url: Base URL of the RNZ Storytime website
            output_dir: Directory to save the downloaded files
            max_retries: Maximum number of retries for failed downloads
            timeout: Timeout for HTTP requests in seconds
            max_workers: Maximum number of concurrent downloads
        """
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        self.interrupted = False

        # Create a lock dictionary to prevent concurrent access to the same file
        self.file_locks: Dict[str, threading.Lock] = {}
        self.file_locks_lock = threading.Lock()

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Keep track of all processed books
        self.processed_books: Set[str] = set()
        self.load_processed_books()

    def load_processed_books(self) -> None:
        """Load the list of already processed books from a file."""
        processed_file = self.output_dir / 'processed_books.txt'
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8') as f:
                    self.processed_books = set(line.strip() for line in f)
                logger.info(f"Loaded {len(self.processed_books)} processed books from history")
            except (IOError, OSError) as e:
                logger.error(f"Error loading processed books file: {e}")
                # Continue with empty set if file can't be read
                self.processed_books = set()

    def save_processed_book(self, book_slug: str) -> None:
        """Save a book slug to the processed books file.
        
        Args:
            book_slug: The unique identifier for the book
        """
        processed_file = self.output_dir / 'processed_books.txt'
        try:
            with self.file_locks_lock:
                # Get or create lock for this file
                file_key = str(processed_file)
                if file_key not in self.file_locks:
                    self.file_locks[file_key] = threading.Lock()
                file_lock = self.file_locks[file_key]
                
            # Use lock to prevent race conditions when multiple threads write
            with file_lock:
                with open(processed_file, 'a', encoding='utf-8') as f:
                    f.write(f"{book_slug}\n")
                self.processed_books.add(book_slug)
        except (IOError, OSError) as e:
            logger.error(f"Error saving book {book_slug} to processed list: {e}")

    def make_request(self, url: str, retries: Optional[int] = None) -> Optional[requests.Response]:
        """Make an HTTP request with retry logic.

        Args:
            url: URL to request
            retries: Number of retries, defaults to self.max_retries

        Returns:
            Response object if successful, None otherwise
            
        Raises:
            requests.RequestException: Various network-related errors
            requests.HTTPError: For 4xx/5xx responses
            requests.ConnectionError: For connection problems
            requests.Timeout: When request times out
        """
        if retries is None:
            retries = self.max_retries
            
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            logger.error(f"Invalid URL format: {url}")
            return None

        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                
                # Handle rate limiting (429 Too Many Requests)
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    logger.warning(f"Rate limited. Waiting {retry_after} seconds before retry.")
                    time.sleep(retry_after)
                    continue
                    
                response.raise_for_status()
                return response
                
            except requests.HTTPError as e:
                logger.warning(f"HTTP error for {url}: {e}. Attempt {attempt + 1}/{retries}")
                if attempt + 1 < retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to retrieve {url} after {retries} attempts: HTTP error {e}")
                    return None
                    
            except requests.ConnectionError as e:
                logger.warning(f"Connection error for {url}: {e}. Attempt {attempt + 1}/{retries}")
                if attempt + 1 < retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to retrieve {url} after {retries} attempts: Connection error")
                    return None
                    
            except requests.Timeout as e:
                logger.warning(f"Timeout for {url}: {e}. Attempt {attempt + 1}/{retries}")
                if attempt + 1 < retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to retrieve {url} after {retries} attempts: Timeout")
                    return None
                    
            except requests.RequestException as e:
                logger.warning(f"Request failed for {url}: {e}. Attempt {attempt + 1}/{retries}")
                if attempt + 1 < retries:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"Failed to retrieve {url} after {retries} attempts")
                    return None

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename to remove invalid characters across all platforms.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Replace invalid characters with underscore (covers Windows, macOS, Linux)
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        
        # Remove control characters
        sanitized = re.sub(r'[\x00-\x1f\x7f]', '', sanitized)
        
        # Trim leading/trailing spaces and periods (problematic on Windows)
        sanitized = sanitized.strip(' .')
        
        # Handle empty filenames
        if not sanitized:
            sanitized = 'unnamed'
            
        # Ensure filename isn't too long (Windows has 260 char path limit)
        # Using 200 to be safe with paths
        if len(sanitized) > 200:
            # Keep extension if present
            parts = sanitized.rsplit('.', 1)
            if len(parts) > 1 and len(parts[1]) <= 10:  # If has extension
                sanitized = parts[0][:196-len(parts[1])] + '.' + parts[1]
            else:
                sanitized = sanitized[:200]
                
        return sanitized

    def get_age_category_books(self, age_category):
        """Get all books for a given age category.

        Args:
            age_category: Age category to fetch books for (e.g., 'little-kids', 'kids', 'young-adult')

        Returns:
            List of book nodes
        """
        url = f"{self.base_url}/page-data/age/{age_category}/page-data.json"
        logger.info(f"Fetching books for age category: {age_category}")

        response = self.make_request(url)
        if not response:
            return []

        try:
            data = response.json()
            books = data.get('result', {}).get('data', {}).get('allBookJson', {}).get('edges', [])
            logger.info(f"Found {len(books)} books in {age_category}")
            return books
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing JSON from {url}: {e}")
            return []

    def get_book_details(self, gatsby_slug):
        """Get detailed information about a book.

        Args:
            gatsby_slug: Gatsby slug of the book

        Returns:
            Book details dictionary if successful, None otherwise
        """
        url = f"{self.base_url}/page-data/{gatsby_slug}/page-data.json"
        logger.info(f"Fetching details for book: {gatsby_slug}")

        response = self.make_request(url)
        if not response:
            return None

        try:
            data = response.json()
            book_data = data.get('result', {}).get('data', {}).get('bookJson', {})
            return book_data
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing JSON from {url}: {e}")
            return None

    def get_file_lock(self, path):
        """Get a lock for a specific file path.

        Args:
            path: Path to get a lock for

        Returns:
            Lock object for the path
        """
        path_str = str(path)
        with self.file_locks_lock:
            if path_str not in self.file_locks:
                self.file_locks[path_str] = threading.Lock()
            return self.file_locks[path_str]

    def download_file(self, url, output_path):
        """Download a file with resume capability.

        Args:
            url: URL to download
            output_path: Path to save the file

        Returns:
            True if successful, False otherwise
        """
        # Check if we've been interrupted
        if self.interrupted:
            return False

        # Get a lock for this specific file
        file_lock = self.get_file_lock(output_path)

        # Use the lock to ensure we don't have concurrent access to the same file
        with file_lock:
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"File already exists: {output_path}")
                return True

            logger.info(f"Downloading {url} to {output_path}")

            # Create temporary file for download
            temp_path = output_path.with_suffix(f"{output_path.suffix}.part")

            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(url, stream=True, timeout=self.timeout)
                    response.raise_for_status()

                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)

                    # Rename to final filename - this is where the error was happening
                    try:
                        # Make sure the destination file doesn't exist before renaming
                        if output_path.exists():
                            output_path.unlink()
                        temp_path.rename(output_path)
                    except OSError as e:
                        logger.error(f"Error renaming file {temp_path} to {output_path}: {e}")
                        if temp_path.exists():
                            temp_path.unlink()
                        return False

                    logger.info(f"Successfully downloaded {url}")
                    return True

                except requests.RequestException as e:
                    logger.warning(f"Download failed for {url}: {e}. Attempt {attempt + 1}/{self.max_retries}")
                    if attempt + 1 < self.max_retries:
                        # Exponential backoff
                        time.sleep(2 ** attempt)
                    else:
                        logger.error(f"Failed to download {url} after {self.max_retries} attempts")
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except OSError:
                                pass
                        return False
                except Exception as e:
                    logger.error(f"Unexpected error downloading {url}: {e}")
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
                    return False

    def update_mp3_metadata(self, mp3_path: Path, image_path: Optional[Path], 
                         title: str, author: str, album: str, synopsis: str, 
                         reading_age: Optional[str] = None, 
                         track_number: Optional[int] = None, 
                         total_tracks: Optional[int] = None) -> bool:
        """Update MP3 metadata with book information and cover image.

        Args:
            mp3_path: Path to the MP3 file
            image_path: Path to the cover image
            title: Track title (clip title)
            author: Book author
            album: Album name (book title)
            synopsis: Clip synopsis
            reading_age: Reading age category for genre tag
            track_number: Track number for episode/clip (optional)
            total_tracks: Total number of tracks in album (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load existing ID3 tags or create if they don't exist
            try:
                audio = ID3(mp3_path)
            except ID3NoHeaderError:
                # If no ID3 tags exist, create them
                audio = ID3()
            except (OSError, IOError) as e:
                logger.error(f"Error reading MP3 file {mp3_path}: {e}")
                return False

            # Set title
            audio.add(TIT2(encoding=3, text=title))

            # Set artist (author)
            audio.add(TPE1(encoding=3, text=author))

            # Set album (reading age)
            audio.add(TALB(encoding=3, text=album))

            # Set comment (synopsis)
            audio.add(COMM(encoding=3, lang='eng', desc='Description', text=synopsis))

            # Add reading age as genre if available
            if reading_age:
                audio.add(TCON(encoding=3, text=f"RNZ {reading_age}"))

            # Add track number if available
            if track_number is not None and total_tracks is not None:
                audio.add(TRCK(encoding=3, text=f"{track_number}/{total_tracks}"))

            # Embed cover image if available
            if image_path and image_path.exists():
                try:
                    # Open image and convert to JPEG if it's not already
                    img = Image.open(image_path)
                    
                    # Default mime type
                    mimetype = 'image/jpeg'
                    
                    if img.format != 'JPEG':
                        # Convert to JPEG for better compatibility
                        buffer = io.BytesIO()
                        img.convert('RGB').save(buffer, format='JPEG')
                        img_data = buffer.getvalue()
                    else:
                        # Read image in chunks to avoid loading large files entirely into memory
                        with open(image_path, 'rb') as f:
                            img_data = f.read(10 * 1024 * 1024)  # Read up to 10MB
                            if len(img_data) >= 10 * 1024 * 1024:
                                logger.warning(f"Image {image_path} is very large (≥10MB), may impact performance")

                    # Add picture
                    audio.add(APIC(
                        encoding=3,           # UTF-8
                        mime=mimetype,        # MIME type
                        type=3,               # Cover (front)
                        desc='Cover',
                        data=img_data
                    ))
                except (IOError, OSError) as e:
                    logger.warning(f"Error processing cover image {image_path}: {e}")
                    # Continue without the cover image
                except Exception as e:
                    logger.warning(f"Unexpected error with image {image_path}: {e}")
                    # Continue without the cover image

            # Save the changes
            try:
                audio.save(mp3_path)
                logger.info(f"Updated metadata for {mp3_path}")
                return True
            except (IOError, OSError) as e:
                logger.error(f"Error saving metadata to {mp3_path}: {e}")
                return False

        except Exception as e:
            logger.error(f"Error updating metadata for {mp3_path}: {e}")
            return False

    def determine_reading_age_category(self, reading_ages):
        """Determine reading age category for organizing files.

        Args:
            reading_ages: List of reading age objects

        Returns:
            Reading age category as string
        """
        # Map reading age titles to priority values (lower = younger)
        age_priorities = {
            "Little Kids": 1,
            "Kids": 2,
            "Young Adult": 3
        }

        # Extract titles and find the youngest age
        titles = [age.get('title') for age in reading_ages if age.get('title')]
        if not titles:
            return "General"  # Default if no ages available

        # Find the youngest age category
        youngest_age = min(titles, key=lambda x: age_priorities.get(x, 999))
        return youngest_age

    def process_book(self, book_node):
        """Process a single book by downloading audio, image, and updating metadata.

        Args:
            book_node: Book node from the API

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we've been interrupted
            if self.interrupted:
                return False

            slug = book_node.get('node', {}).get('slug')
            gatsby_slug = book_node.get('node', {}).get('gatsby_slug')

            if not slug or not gatsby_slug:
                logger.error(f"Missing slug or gatsby_slug in book node: {book_node}")
                return False

            # Skip if already processed
            if slug in self.processed_books:
                logger.info(f"Book already processed: {slug}")
                return True

            # Get detailed book information
            book_details = self.get_book_details(gatsby_slug)
            if not book_details:
                logger.error(f"Failed to get details for book: {gatsby_slug}")
                return False

            # Extract book information
            title = book_details.get('title', '')
            book_authors = book_details.get('book_author', [])
            author_names = [author.get('title', '') for author in book_authors]
            author = ', '.join(filter(None, author_names))

            reading_ages = book_details.get('reading_age', [])
            reading_age_category = self.determine_reading_age_category(reading_ages)

            synopsis = book_details.get('book_synopsis', '')
            image_url = book_details.get('image')

            clips = book_details.get('clip', [])
            if not clips:
                logger.warning(f"No audio clips found for book: {slug}")
                return False

            # Process each clip
            for i, clip in enumerate(clips):
                clip_title = clip.get('title', title)
                clip_url = clip.get('clip_link')
                clip_synopsis = clip.get('clip_synopsis', synopsis)

                if not clip_url:
                    logger.warning(f"No audio URL for clip {i} of book: {slug}")
                    continue

                # Create sanitized filenames
                sanitized_title = self.sanitize_filename(title)
                reading_age_category = self.determine_reading_age_category(reading_ages)
                sanitized_age = self.sanitize_filename(reading_age_category)

                # Create output paths
                output_dir = self.output_dir / sanitized_age / sanitized_title

                # Create a better filename for episodes
                if len(clips) > 1:
                    # Try to extract episode number if possible
                    episode_match = re.search(r'Episode (\d+)', clip_title)
                    if episode_match:
                        episode_num = episode_match.group(1).zfill(2)  # Pad with zeros for sorting
                        mp3_filename = f"{sanitized_title}_E{episode_num}.mp3"
                    else:
                        mp3_filename = f"{sanitized_title}_part{i+1}.mp3"
                else:
                    mp3_filename = f"{sanitized_title}.mp3"

                mp3_path = output_dir / mp3_filename

                # Download the audio file
                if not self.download_file(clip_url, mp3_path):
                    logger.error(f"Failed to download audio for {slug}")
                    continue

                # Download the cover image if available
                image_path = None
                # First try to use clip-specific image if available
                clip_image_url = clip.get('clip_image')
                image_url_to_use = clip_image_url if clip_image_url else image_url

                if image_url_to_use:
                    image_filename = f"{sanitized_title}_cover{Path(urlparse(image_url_to_use).path).suffix}"
                    image_path = output_dir / image_filename
                    if not self.download_file(image_url_to_use, image_path):
                        logger.warning(f"Failed to download cover image for {slug}")
                        image_path = None

                # Update MP3 metadata
                self.update_mp3_metadata(
                    mp3_path=mp3_path,
                    image_path=image_path,
                    title=clip_title,
                    author=author,
                    album=title,  # Use book title as album name
                    synopsis=clip_synopsis,
                    reading_age=reading_age_category,
                    track_number=i+1,
                    total_tracks=len(clips)
                )

            # Mark book as processed
            self.save_processed_book(slug)
            return True

        except Exception as e:
            logger.error(f"Error processing book: {e}")
            return False

    def archive_all_books(self) -> Tuple[int, int, bool]:
        """Archive all books from all age categories.
        
        Returns:
            Tuple[int, int, bool]: (successful_count, failure_count, was_interrupted)
        """
        # Set up signal handling for graceful interruption
        self.interrupted = False
        original_sigint_handler = signal.getsignal(signal.SIGINT)

        def sigint_handler(sig, frame):
            logger.info("Interrupt received, gracefully shutting down...")
            self.interrupted = True
            # Restore original handler to allow forced exit on second Ctrl+C
            signal.signal(signal.SIGINT, original_sigint_handler)

        # Install the custom handler
        signal.signal(signal.SIGINT, sigint_handler)

        # Age categories to process - moved from hardcoded to class initialization
        age_categories = ['little-kids', 'kids', 'young-adult']
        all_books = []

        # Collect all books from all categories
        for age_category in age_categories:
            books = self.get_age_category_books(age_category)
            all_books.extend(books)

        logger.info(f"Found a total of {len(all_books)} books to process")

        # Process books with concurrent workers
        success_count = 0
        failure_count = 0
        interrupted = False

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_book = {executor.submit(self.process_book, book): book for book in all_books}

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_book):
                if self.interrupted:
                    logger.info("Shutting down, cancelling remaining tasks...")
                    # Cancel all pending futures
                    for f in future_to_book:
                        if not f.done():
                            f.cancel()
                    interrupted = True
                    break

                book = future_to_book[future]
                book_title = book.get('node', {}).get('title', 'Unknown')

                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        logger.info(f"Successfully processed book: {book_title}")
                    else:
                        failure_count += 1
                        logger.warning(f"Failed to process book: {book_title}")
                except Exception as e:
                    failure_count += 1
                    logger.error(f"Exception while processing book {book_title}: {e}")

        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)

        if interrupted:
            logger.info(f"Archiving interrupted. Progress saved. Successful: {success_count}, Failed: {failure_count}")
            return success_count, failure_count, True

        logger.info(f"Archiving complete. Successful: {success_count}, Failed: {failure_count}")
        return success_count, failure_count, False


def main() -> int:
    """Main entry point.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Ensure proper encoding for console output on Windows
    if sys.platform == 'win32':
        try_set_windows_utf8()

    parser = argparse.ArgumentParser(description='Download and archive audiobooks from RNZ Storytime.')
    parser.add_argument('--base-url', default='https://storytime.rnz.co.nz', help='Base URL of the RNZ Storytime website')
    parser.add_argument('--output-dir', default='./rnz_storytime_archive', help='Directory to save the downloaded files')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for failed downloads')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout for HTTP requests in seconds')
    parser.add_argument('--max-workers', type=int, default=5, help='Maximum number of concurrent downloads')

    args = parser.parse_args()

    archiver = RNZArchiver(
        base_url=args.base_url,
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        timeout=args.timeout,
        max_workers=args.max_workers
    )

    try:
        success_count, failure_count, interrupted = archiver.archive_all_books()
        if interrupted:
            logger.info("Archiving was interrupted but progress has been saved.")
            return 1
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
