import os
import csv
import logging
from typing import List, Dict, Any
from pathlib import Path
import argparse

logger = logging.getLogger(__name__)


def create_sample_manifest(output_path: str, num_samples: int = 100):
    """Create a sample manifest for testing purposes."""
    logger.info(f"Creating sample manifest with {num_samples} tracks")
    
    # Sample data for testing
    sample_tracks = []
    artists = ["The Beatles", "Queen", "Led Zeppelin", "Pink Floyd", "Radiohead", 
               "Nirvana", "David Bowie", "Bob Dylan", "The Rolling Stones", "AC/DC"]
    genres = ["Rock", "Pop", "Jazz", "Blues", "Country", "Electronic", "Hip Hop", "Classical"]
    
    for i in range(num_samples):
        track = {
            "track_id": f"track_{i:06d}",
            "audio_path": f"data/raw/audio/track_{i:06d}.wav",
            "lyrics_path": f"data/raw/lyrics/track_{i:06d}.txt",
            "artist": artists[i % len(artists)],
            "title": f"Sample Song {i+1}",
            "genre": genres[i % len(genres)],
            "duration": 180 + (i % 120),  # 3-5 minutes
            "year": 1970 + (i % 50)
        }
        sample_tracks.append(track)
    
    # Write manifest
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        if sample_tracks:
            writer = csv.DictWriter(f, fieldnames=sample_tracks[0].keys())
            writer.writeheader()
            writer.writerows(sample_tracks)
    
    logger.info(f"Sample manifest created: {output_path}")
    return output_path


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load track manifest from CSV."""
    tracks = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tracks.append(row)
    
    logger.info(f"Loaded {len(tracks)} tracks from manifest")
    return tracks


def validate_manifest(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate manifest entries and filter out invalid ones."""
    valid_tracks = []
    
    for track in tracks:
        # Check required fields
        required_fields = ["track_id", "audio_path", "lyrics_path", "artist", "title"]
        if not all(field in track for field in required_fields):
            logger.warning(f"Skipping track {track.get('track_id', 'unknown')}: missing required fields")
            continue
        
        # Check if files exist (optional validation)
        audio_path = track["audio_path"]
        lyrics_path = track["lyrics_path"]
        
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file not found: {audio_path}")
            # For demo purposes, we'll still include it
        
        if not os.path.exists(lyrics_path):
            logger.warning(f"Lyrics file not found: {lyrics_path}")
            # For demo purposes, we'll still include it
        
        valid_tracks.append(track)
    
    logger.info(f"Validated {len(valid_tracks)} tracks out of {len(tracks)}")
    return valid_tracks


def create_sample_files(manifest_path: str):
    """Create sample audio and lyrics files for testing."""
    tracks = load_manifest(manifest_path)
    
    for track in tracks:
        # Create sample lyrics file
        lyrics_path = track["lyrics_path"]
        os.makedirs(os.path.dirname(lyrics_path), exist_ok=True)
        
        sample_lyrics = f"""Sample lyrics for {track['title']} by {track['artist']}

Verse 1:
This is a sample song
With some sample lyrics
For testing purposes
In the Lyric2Vec system

Chorus:
Sample chorus here
Repeated multiple times
To make it longer
And more realistic

Verse 2:
Another verse follows
With different words
But similar structure
To the first verse

[Instrumental break]

Chorus:
Sample chorus here
Repeated multiple times
To make it longer
And more realistic

Outro:
This is the end
Of our sample song
Thank you for listening
To this test track
"""
        
        with open(lyrics_path, 'w', encoding='utf-8') as f:
            f.write(sample_lyrics)
        
        # Note: We don't create actual audio files here as they would be large
        # In a real implementation, you would download or process actual audio files
        logger.info(f"Created sample lyrics: {lyrics_path}")


def main():
    """Main function for data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest music data")
    parser.add_argument("--out", default="data/raw/manifest.csv", help="Output manifest path")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of sample tracks")
    parser.add_argument("--create-samples", action="store_true", help="Create sample files")
    
    args = parser.parse_args()
    
    # Create sample manifest
    manifest_path = create_sample_manifest(args.out, args.num_samples)
    
    if args.create_samples:
        create_sample_files(manifest_path)
    
    logger.info("Data ingestion completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
